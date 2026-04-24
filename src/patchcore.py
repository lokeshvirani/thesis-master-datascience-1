#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, Wide_ResNet50_2_Weights
from torch.utils.data import DataLoader

from coreset import CoresetSampler
from torchvision import transforms
from PIL import ImageFilter
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)

class PatchCoreSingle(torch.nn.Module):
    """
    PatchCore implementation adapted for KSDD2 dataset with the ability to work with
    arbitrary image dimensions without upsampling to fixed sizes.

    NOTE:
      - For baseline PatchCore, use memory_type='negative' and fit() on normal images only.
      - For dual PatchCore, you can also build a positive memory bank from defect crops
        (memory_type='positive'), but Dual scoring should use *raw distances* (s_star / dist_map),
        not the weighted final score (s_final). This file returns s_star for that purpose.
    """
    def __init__(self, device='cuda', backbone='resnet50', memory_type='negative', subsampling_share=0.01):
        super(PatchCoreSingle, self).__init__()
        self.k_nearest = 3
        self.device = device
        self.memory_bank = None
        self.extracted_features = []
        self.memory_type = memory_type
        self.subsampling_share = subsampling_share

        # Load the appropriate backbone
        if backbone == 'resnet50':
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        elif backbone == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Set model to evaluation mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hooks to extract features
        def hook(module, input, output):
            self.extracted_features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, sample):
        """Extract features from the input image."""
        self.extracted_features = []
        with torch.no_grad():
            _ = self.model(sample)
        return self.extracted_features

    def fit(self, dataloader: DataLoader):
        """Build memory bank from samples in dataloader (normal or defect crops)."""
        memory_items = []

        for sample, _, _, _ in tqdm(dataloader, desc=f"Building {self.memory_type.capitalize()} Memory Bank"):
            sample = sample.to(self.device)
            features = self(sample)

            self.avg = torch.nn.AvgPool2d(3, stride=1)
            fmap_size_h = features[0].shape[-2]
            fmap_size_w = features[0].shape[-1]

            self.resize = torch.nn.AdaptiveAvgPool2d((fmap_size_h, fmap_size_w))
            resized_maps = [self.resize(self.avg(fmap)) for fmap in features]

            sample_patch_collection = torch.cat(resized_maps, dim=1)
            sample_patch_collection = sample_patch_collection.reshape(sample_patch_collection.shape[1], -1).T
            memory_items.append(sample_patch_collection)

        self.memory_bank = torch.cat(memory_items, dim=0).to(self.device)
        N, C = self.memory_bank.shape
        print(f"Memory Bank: {N} patch embeddings collected with {C} dimensions")

        # Apply coreset subsampling to reduce memory bank size
        target = max(1000, int(N * self.subsampling_share))

        self.memory_bank, indices = self.coreset_subsampling(
            self.memory_bank, target, epsilon=0.1, device=self.device
        )
        print(f"Memory Bank reduced to {len(indices)} patch embeddings")

    def get_anomaly_score(self, sample):
        """
        Compute:
          - s_final: weighted anomaly score used by PatchCoreSingle (baseline)
          - dist_map: per-patch min distance map (before upsampling)
          - s_star: raw max min-distance over patches (recommended for Dual combination)
        """
        feature_maps = self(sample)

        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size_h = feature_maps[0].shape[-2]
        fmap_size_w = feature_maps[0].shape[-1]

        self.resize = torch.nn.AdaptiveAvgPool2d((fmap_size_h, fmap_size_w))
        resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]

        patch = torch.cat(resized_maps, dim=1)
        patch = patch.reshape(patch.shape[1], -1).T

        # Calculate distances to memory bank
        distances = torch.cdist(patch, self.memory_bank, p=2.0)
        dist_score, dist_score_idx = torch.min(distances, dim=1)

        # Find the patch with maximum distance (most anomalous w.r.t. this bank)
        s_idx = torch.argmax(dist_score)
        s_star = dist_score[s_idx]
        m_test_star = patch[s_idx]

        # Neighborhood-based weight (same as your original code)
        m_star = self.memory_bank[dist_score_idx[s_idx]].unsqueeze(0)
        knn_dists = torch.cdist(m_star, self.memory_bank, p=2.0)
        _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)
        m_neighborhood = self.memory_bank[nn_idxs[0, 1:]]

        w_denominator = torch.linalg.norm(m_test_star - m_neighborhood, dim=1)
        norm = torch.sqrt(torch.tensor(patch.shape[1], device=self.device))

        if self.memory_type == 'negative':
            w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))
        else:  # positive (kept as-is; Dual uses s_star/dist_map instead)
            w = torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm))

        s_final = w * s_star

        # Create distance map at feature-map resolution
        dist_map = dist_score.view(1, 1, fmap_size_h, fmap_size_w)

        # IMPORTANT: return s_star for Dual ratio scoring
        return s_final, dist_map, s_star

    def predict(self, sample):
        """Predict anomaly score and segmentation map (single-bank)."""
        score, dist_map, _ = self.get_anomaly_score(sample)

        original_size = (sample.shape[2], sample.shape[3])
        segm_map = torch.nn.functional.interpolate(dist_map, size=original_size, mode='bilinear')

        segm_map = self.gaussian_blur(segm_map)

        return score, segm_map

    def evaluate_single(self, test_dataloader):
        """Evaluate the model on a test dataset for single memory bank case."""
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for sample, label, mask, _ in tqdm(test_dataloader, desc="Evaluating PatchCoreSingle"):
            sample = sample.to(self.device)
            mask = mask.to(self.device)

            image_labels.append(label.item())
            pixel_labels.extend((mask.flatten().cpu().numpy() > 0).astype(np.uint8))

            score, segm_map = self.predict(sample)

            image_preds.append(float(score.detach().cpu().numpy()))
            pixel_preds.extend(segm_map.flatten().detach().cpu().numpy())

        image_auc = roc_auc_score(image_labels, image_preds)
        pixel_auc = roc_auc_score(pixel_labels, pixel_preds)

        return image_auc, pixel_auc

    def bilinear_upsample(self, lower_spatial_block, target_size):
        """Upsample feature map to target size using bilinear interpolation."""
        if lower_spatial_block.shape[2:] == target_size:
            return lower_spatial_block
        return F.interpolate(lower_spatial_block, size=target_size, mode='bilinear', align_corners=False)

    def coreset_subsampling(self, embeddings, target_samples, epsilon=0.1, device=None, use_projection=True):
        """Perform coreset subsampling to reduce memory bank size."""
        if device is None:
            device = embeddings.device

        N, C = embeddings.shape

        # Apply random projection if beneficial
        if use_projection and C > 10:
            d_star = 128
            if d_star < C:
                print(f"Projecting from {C} to {d_star} dimensions")
                embeddings_for_sampling = self.random_projection(embeddings, d_star, epsilon)
            else:
                embeddings_for_sampling = embeddings
        else:
            embeddings_for_sampling = embeddings

        target_samples = min(target_samples, N)

        sampler = CoresetSampler(n_samples=target_samples, device=str(device), tqdm_disable=False, verbose=1)
        selected_indices = sampler.sample(embeddings_for_sampling.cpu().numpy())

        return embeddings[selected_indices], selected_indices

    def random_projection(self, embeddings, target_dim, epsilon=0.1, seed=0):
        """Apply random projection to reduce embedding dimensionality."""
        N, C = embeddings.shape
        torch.manual_seed(seed)

        projection_matrix = torch.randn(C, target_dim, device=embeddings.device)
        projection_matrix /= torch.sqrt(torch.sum(projection_matrix**2, dim=0, keepdim=True))

        return torch.matmul(embeddings, projection_matrix)

    def gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to a tensor."""
        blur_kernel = ImageFilter.GaussianBlur(radius=2)
        tensor_to_pil = transforms.ToPILImage()
        pil_to_tensor = transforms.ToTensor()

        max_value = img.max()
        if max_value.item() == 0:
            return img

        blurred_pil = tensor_to_pil(img[0] / max_value).filter(blur_kernel)
        blurred_map = pil_to_tensor(blurred_pil).to(img.device)

        return blurred_map * max_value


class PatchCoreDual:
    """
    Dual PatchCore model that combines information from negative (normal) and positive (defect) memory banks.

    IMPORTANT FIX:
      Use *raw distances* (s_star / dist_map) for combination:
        - neg_s_star: far from normal => large
        - pos_s_star: far from defects => large, close to defects => small
      Ratio scoring:
        score = neg_s_star / (pos_s_star + eps)
        map   = neg_map   / (pos_map   + eps)
    """
    def __init__(self, device='cuda', backbone='wide_resnet50_2', negative_subsampling=0.01, positive_subsampling=0.10):
        self.negative_model = PatchCoreSingle(device, backbone, memory_type='negative', subsampling_share=negative_subsampling)
        self.positive_model = PatchCoreSingle(device, backbone, memory_type='positive', subsampling_share=positive_subsampling)
        self.device = device

    def fit(self, negative_dataloader, positive_dataloader):
        """Build both negative and positive memory banks."""
        print("Training negative model...")
        self.negative_model.fit(negative_dataloader)

        print("Training positive model...")
        self.positive_model.fit(positive_dataloader)

    def predict(self, sample):
        """
        Predict anomaly score and map for a sample using both memory banks.
        Final score is ratio of raw distances: neg_s_star / (pos_s_star + eps).
        """
        _, neg_map, neg_sstar = self.negative_model.get_anomaly_score(sample)
        _, pos_map, pos_sstar = self.positive_model.get_anomaly_score(sample)

        eps = 1e-6

        ratio_score = neg_sstar / (pos_sstar + eps)

        ratio_map = neg_map / (pos_map + eps)

        # Upsample to original image size
        original_size = (sample.shape[2], sample.shape[3])
        segm_map = self.negative_model.bilinear_upsample(ratio_map, original_size)

        segm_map = self.negative_model.gaussian_blur(segm_map)

        return ratio_score, segm_map

    def evaluate(self, test_dataloader):
        """Evaluate the model on a test dataset."""
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for sample, label, mask, _ in tqdm(test_dataloader, desc="Evaluating PatchCoreDual"):
            sample = sample.to(self.device)
            mask = mask.to(self.device)

            image_labels.append(label.item())
            pixel_labels.extend((mask.flatten().cpu().numpy() > 0).astype(np.uint8))

            score, segm_map = self.predict(sample)

            image_preds.append(float(score.detach().cpu().numpy()))
            pixel_preds.extend(segm_map.flatten().detach().cpu().numpy())

        image_auc = roc_auc_score(image_labels, image_preds)
        pixel_auc = roc_auc_score(pixel_labels, pixel_preds)

        return image_auc, pixel_auc
