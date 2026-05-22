"""Regenerate a defect with a diffusion model (step 2 of the pipeline).

Inpainting: take a GOOD (normal) image, a MASK marking where the defect goes
(the real defect's GT mask), and a PROMPT (the caption from step 1). The model
repaints only the masked area, producing a synthetic defect at the real
location. Goal: the result resembles the original BAD image.
"""
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline


MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

IMAGE_SIZE = (256, 256)   # all images resized to this square canvas
MASK_BLUR = 2             # soften mask edges so the defect blends in
NUM_STEPS = 30      # how many denoising steps (more = better quality, slower)
GUIDANCE = 7.0      # how strongly to follow the prompt
STRENGTH = 0.92     # how much to change the masked area (0 = none, 1 = replace fully)

# Wraps the caption so SDXL produces a realistic photo, not a stylised drawing.
PROMPT_TEMPLATE = (
    "photorealistic close-up macro photo of {caption}, "
    "industrial inspection image, natural lighting, subtle defect"
)

# What the model must NOT draw (keeps the defect realistic, not cartoonish).
NEGATIVE_PROMPT = (
    "text, watermark, logo, letters, numbers, "
    "cartoon, drawing, illustration, painting, "
    "extra objects, oversized defect, unrealistic defect shape, "
    "heavy blur, distortion, glare"
)

# Some categories need a different list. `can` defects ARE printed-label issues
# (holographic foil, overprinted text), so we must NOT forbid text/letters here;
# instead we forbid physical metal damage, which is the WRONG kind of defect.
NEGATIVE_PROMPT_BY_CATEGORY = {
    "can": (
        "metal dent, scratch, rust, corrosion, hole, puncture, "
        "cartoon, drawing, illustration, painting, "
        "extra objects, oversized defect, unrealistic defect shape, "
        "heavy blur, distortion, glare"
    ),
}


def negative_prompt_for(category):
    """Per-category negative prompt, falling back to the default."""
    return NEGATIVE_PROMPT_BY_CATEGORY.get(category, NEGATIVE_PROMPT)


def load_model(low_vram=False):
    """Load the SDXL inpainting model onto the GPU (or CPU if no GPU)."""
    use_gpu = torch.cuda.is_available()                 # is a GPU available?
    dtype = torch.float16 if use_gpu else torch.float32  # GPU likes half precision

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

    if low_vram and use_gpu:
        pipe.enable_model_cpu_offload()   # keep idle parts on CPU -> less GPU memory
        pipe.enable_vae_slicing()         # decode the image in slices, not all at once
        pipe.enable_vae_tiling()          # decode in tiles -> even less GPU memory
    else:
        pipe = pipe.to("cuda" if use_gpu else "cpu")    # load whole model on device

    return pipe


def load_good_image(path):
    """Load a normal/good image as RGB, resized to the canvas."""
    return Image.open(path).convert("RGB").resize(IMAGE_SIZE)


def load_mask(path):
    """Load a GT defect mask as grayscale, resized, with softened edges."""
    mask = Image.open(path).convert("L").resize(IMAGE_SIZE, Image.NEAREST)
    if MASK_BLUR > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=MASK_BLUR))
    return mask


def generate_defect(pipe, good_image, mask, caption, category=None, seed=42):
    """Inpaint the defect described by `caption` into the masked area of `good_image`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)  # fixed seed = repeatable

    full_prompt = PROMPT_TEMPLATE.format(caption=caption)  # wrap for photorealism

    result = pipe(
        prompt=full_prompt,             # what defect to paint
        negative_prompt=negative_prompt_for(category),  # what to avoid (per category)
        image=good_image,         # the base good image
        mask_image=mask,          # white area = where to paint
        guidance_scale=GUIDANCE,
        strength=STRENGTH,
        num_inference_steps=NUM_STEPS,
        generator=generator,
        height=IMAGE_SIZE[1], width=IMAGE_SIZE[0],  # keep output at 256x256
    )
    return result.images[0]       # the generated image


def save_comparison(out_path, good_image, real_bad_path, generated):
    """Save a side-by-side strip: good base | real defect | generated defect."""
    real = Image.open(real_bad_path).convert("RGB").resize(IMAGE_SIZE)
    w, h = IMAGE_SIZE
    strip = Image.new("RGB", (w * 3, h), (255, 255, 255))  # blank canvas, 3 panels wide
    strip.paste(good_image.resize(IMAGE_SIZE), (0, 0))     # left panel
    strip.paste(real, (w, 0))                              # middle panel
    strip.paste(generated.resize(IMAGE_SIZE), (w * 2, 0))  # right panel
    strip.save(out_path)

