# Method

## 1. Motivation

Earlier experiments compared real anomalies against *synthetic* anomalies that
were produced from **random mask shapes and generic, human-written prompts**.
Under that setting real and synthetic patches did not overlap in feature space,
but the comparison was not fair: the generator was never actually asked to
reproduce a *specific* real defect.

This method removes that confound with a **grounded reconstruction** test. For
every real defect we condition the generator on (i) a description of the defect
produced by a vision LLM and (ii) the defect's own location (its ground-truth
mask), and ask it to paint that defect onto a clean object. The synthetic output
is therefore a deliberate attempt to reproduce a known real defect. We then
compare real, synthetic, and normal samples in the PatchCore feature space with
t-SNE. If grounded generation still fails to overlap the real defects, the gap
is attributable to the **generative model**, not to prompt engineering.

## 2. Pipeline overview

![Grounded reconstruction pipeline](method_pipeline.png)

```
                 ┌──────────────────────────────┐
   real BAD ────►│  Stage 1: Vision-LLM caption │────►  "diamond holographic
   image         └──────────────────────────────┘        foil patch on label"
      │                                                          │
      │ (its ground-truth mask = WHERE)                          │ (WHAT)
      ▼                                                          ▼
   ┌──────┐   good (normal) image (BASE)   ┌───────────────────────────────┐
   │  GT  │ ─────────────────────────────► │ Stage 2: SDXL inpainting      │
   │ mask │                                │ base + mask + caption         │──► GENERATED
   └──────┘                                └───────────────────────────────┘     defect
                                                                                    │
   real defects ─┐                                                                  │
   normals ──────┤                                                                  │
   generated ────┴──►  ┌─────────────────────────────────┐                         │
                       │ Stage 3: PatchCore patch features│ ◄───────────────────────┘
                       │  WideResNet50 layer2+layer3 (1536-D)
                       └─────────────────────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────────┐
                       │ Stage 4: PCA(50) → t-SNE(2-D)    │──►  comparison figure
                       │  real vs generated vs normal     │     (overlap = match)
                       └─────────────────────────────────┘
```

## 3. Data

We use the MVTec AD 2 dataset, pre-processed per category into
`data/preprocessed/<category>/` with `train/`, `test/`, and `train.csv` /
`test.csv` listing each image as `negative` (normal) or `positive` (defective).
Each defective image `X.png` has a paired ground-truth mask `X_GT.png`.

- **Good (normal) images** — `negative` rows; used as the clean base for
  inpainting and as the source of normal patches.
- **Real (bad) images** — `positive` rows with their `_GT` masks; provide both
  the LLM caption input (Stage 1) and the defect location (Stage 2).

The `can` category contains 90 defective images that correspond to **15 distinct
defects** (each captured under several lighting/shift variants). We caption the
15 distinct defects, which covers the full variety of defect *types* (holographic
foil patches, wrong overprinted text panels, mis-registered labels).

Images are resized to **256×256** for generation and **384×384** for feature
extraction (the finer grid yields more patches per small defect).

## 4. Stage 1 — Defect captioning (vision LLM)

Each real defect image is shown to a vision-language model (VLM) with an
instruction to return a single concise sentence describing the defect's
appearance, type, and location, phrased as an image-generation prompt. The
captions are stored paired with their source image paths so that each caption
can later be matched to the same defect's mask.

- Output: `llm_captions_can_full.json` — `{"captions": {cat: [...]},
  "sources": {cat: [...]}}` with `caption[i] ↔ source[i]`.
- Captions are produced by a vision-language model and stored paired with their
  source image paths, so each caption maps to the same defect's mask downstream.

## 5. Stage 2 — Grounded regeneration (diffusion inpainting)

For each `(caption, source)` pair we resolve the source to its preprocessed bad
image and its `_GT` mask, then run **Stable Diffusion XL inpainting**:

- **base image** = a randomly selected good (normal) image — the clean object;
- **mask** = the real defect's own GT mask (soft-blurred edges) — *where* the
  defect goes;
- **prompt** = the LLM caption wrapped in a photo-realistic grounding template,
  plus a per-category negative prompt that suppresses the wrong defect family
  (e.g. for `can`, suppress metal-damage terms; do **not** suppress the
  holographic/printed-label terms that describe the true defect).

Several reconstructions are produced per defect (different good bases) to sample
the generator's behaviour. Each output is written so it plugs directly into the
t-SNE step.

- Output: `data/preprocessed/<cat>/recon_llmcaption_full/{imgs,masks,compare}/`
  plus `manifest.json`. `compare/` holds `good | real | generated` triptychs for
  qualitative inspection.
- Code: `src/reconstruct_real_defects.py`.
- Settings: 256×256, 30 inference steps, guidance 7.0, strength 0.92.

## 6. Stage 3 — Patch-descriptor extraction (PatchCore features)

Real, generated, and normal images are passed through a frozen
**WideResNet50_2** backbone; we take `layer2` and `layer3` feature maps, local
average-pool them, align them to a common grid, and concatenate to a **1536-D**
descriptor per grid cell — exactly the representation used by PatchCore
(`src/patchcore.py`). A grid cell counts as an **anomaly patch** when it lies
inside the (dilated) GT mask.

- **Normal patches** — random grid cells from good images.
- **Real anomaly patches** — mask-selected cells from real defects.
- **Generated anomaly patches** — mask-selected cells from the reconstructions.

Working at patch level (rather than whole-image) keeps all three groups in one
feature space and avoids the crop-vs-full-image scale artefact that distorted an
earlier image-level t-SNE.

- Code: `src/analysis/tsne_real_vs_gen.py` (`PatchExtractor`,
  `extract_pos_patches`, `extract_neg_patches`).

## 7. Stage 4 — Embedding comparison (t-SNE)

The three patch groups are stacked into one matrix, reduced with **PCA to 50
dimensions** (denoising / speed), then projected to **2-D with t-SNE**
(`perplexity=30`, `init="pca"`, fixed seed). t-SNE preserves local
neighbourhoods, so patches with similar descriptors form clusters. The figure is
drawn with normals as a faint background, generated patches in the mid-layer, and
real patches on top.

**Reading the figure:** where generated (orange) patches land on real (green)
clusters, the synthetic defect matches a real one; real clusters with no
generated points are defect types the generator failed to reproduce. t-SNE is
read **qualitatively** (overlap vs no-overlap) — absolute distances and cluster
sizes are not quantitatively meaningful.

- Output: `results/recon_tsne_full/tsne_<cat>.png`.
- Code: `src/analysis/tsne_real_vs_gen.py` (`main`).

## 8. Implementation map

| Stage | Role | File |
|---|---|---|
| 1 | Vision-language-model defect captioning | external VLM |
| 2 | Grounded SDXL inpainting | `src/reconstruct_real_defects.py` |
| 3 | PatchCore patch descriptors | `src/analysis/tsne_real_vs_gen.py` (extractor) |
| 4 | PCA → t-SNE comparison + plot | `src/analysis/tsne_real_vs_gen.py` (`main`) |
| — | Backbone reference (PatchCore) | `src/patchcore.py` |

## 9. Hyperparameters

| Item | Value |
|---|---|
| Generation resolution | 256×256 |
| Inpainting model | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` |
| Inference steps / guidance / strength | 30 / 7.0 / 0.92 |
| Reconstructions per defect | 4 |
| Feature backbone | WideResNet50_2, `layer2`+`layer3` (1536-D) |
| Feature extraction resolution | 384×384 |
| Dim. reduction | PCA → 50, then t-SNE (perplexity 30, PCA init, seed 42) |

## 10. First experiment (can)

The pipeline was run end-to-end on `can`: 15 distinct defects captioned, 60
reconstructions generated, and a single shared t-SNE produced (300 generated vs
450 real vs 1800 normal patches). The grounded reconstructions overlap the real
defects only partially — texture-like defects (holographic foil) are reproduced,
while structured defects (wrong text panels, mis-registration) are not — so the
real–synthetic gap narrows but does not close, indicating a generative-model
limitation rather than a prompting problem.
