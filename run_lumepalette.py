#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

# NOTE: Keep heavyweight imports (torch/transformers/diffsynth) out of module import time so
# `python run_lumepalette.py --help` works even in environments with mismatched deps.

DEFAULT_STAGE1_LORA_LEFT = "hf://cjeen/lumepalette/stage1/left_adapter_model.safetensors"
DEFAULT_STAGE1_LORA_MIDDLE = "hf://cjeen/lumepalette/stage1/middle_adapter_model.safetensors"
DEFAULT_STAGE1_LORA_RIGHT = "hf://cjeen/lumepalette/stage1/right_adapter_model.safetensors"
DEFAULT_STAGE1_LORA_TOP = "hf://cjeen/lumepalette/stage1/top_adapter_model.safetensors"
DEFAULT_STAGE2_LORA = "hf://cjeen/lumepalette/stage2/adapter_model.safetensors"
DEFAULT_BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"


def is_comfy_flux2(keys: Iterable[str]) -> bool:
    # Comfy-style Flux2 LoRA exported by the training repo uses diffusion_model.double_blocks/... keys.
    return any(
        k.startswith("diffusion_model.")
        or k.startswith("model.diffusion_model.")
        or "double_blocks." in k
        or "single_blocks." in k
        for k in keys
    )


def build_pipe(device: str):
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    import torch

    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=DEFAULT_BASE_MODEL_ID,
                origin_file_pattern="text_encoder/*.safetensors",
            ),
            ModelConfig(
                model_id=DEFAULT_BASE_MODEL_ID,
                origin_file_pattern="transformer/*.safetensors",
            ),
            ModelConfig(
                model_id=DEFAULT_BASE_MODEL_ID,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id=DEFAULT_BASE_MODEL_ID, origin_file_pattern="tokenizer/"
        ),
    )
    return pipe


def _resolve_lora_path(spec: Optional[str]) -> Optional[Path]:
    if spec is None or spec == "":
        return None
    path = Path(spec)
    if path.exists():
        return path
    if spec.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo_spec = spec[len("hf://") :]
        parts = repo_spec.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid HF lora spec: {spec}. Expected hf://repo_id/path/to/file.")
        if len(parts) == 2:
            repo_id = parts[0]
            file_path = parts[1]
        else:
            repo_id = "/".join(parts[:2])
            file_path = "/".join(parts[2:])
        downloaded = hf_hub_download(repo_id=repo_id, filename=file_path)
        return Path(downloaded)
    raise FileNotFoundError(
        f"LoRA source not found: {spec}. Use a local file path or hf://repo_id/path/in/repo."
    )


def maybe_load_lora(pipe, lora_spec: Optional[str], lora_scale: float, enable: bool):
    from safetensors import safe_open
    from diffsynth.utils.state_dict_converters.flux2_comfy_to_diffusers_klein import (
        Flux2ComfyKleinToDiffusers,
    )

    if not enable:
        print("LoRA loading skipped (--no_lora).")
        return
    if lora_spec is None:
        print("No LoRA provided (--lora not set); running base DiT.")
        return

    lora_path = _resolve_lora_path(lora_spec)
    print(f"Loading LoRA: {lora_spec} -> {lora_path} (scale={lora_scale})")
    with safe_open(str(lora_path), framework="pt") as f:
        keys = list(f.keys())
        if is_comfy_flux2(keys):
            state_dict = Flux2ComfyKleinToDiffusers(f)
            pipe.load_lora(pipe.dit, state_dict=state_dict, alpha=lora_scale)
        else:
            pipe.load_lora(pipe.dit, str(lora_path), alpha=lora_scale)
    print("LoRA loaded.")


def load_lora_into_pipe(pipe, lora_spec: str, lora_scale: float):
    from safetensors import safe_open
    from diffsynth.utils.state_dict_converters.flux2_comfy_to_diffusers_klein import (
        Flux2ComfyKleinToDiffusers,
    )

    lora_path = _resolve_lora_path(lora_spec)
    with safe_open(str(lora_path), framework="pt") as f:
        keys = list(f.keys())
        if is_comfy_flux2(keys):
            state_dict = Flux2ComfyKleinToDiffusers(f)
            pipe.load_lora(pipe.dit, state_dict=state_dict, alpha=lora_scale)
        else:
            pipe.load_lora(pipe.dit, str(lora_path), alpha=lora_scale)


# Keep side references in the order expected by stage-2 training/inference.
STAGE1_LIGHTING_ORDER = ("left", "middle", "right", "top")
# Match the old stage-1 batch runner's process order. Since fused LoRA cannot be
# cleared, each direction below gets a fresh pipe and this order only affects logs.
STAGE1_GENERATION_ORDER = ("left", "right", "middle", "top")
STAGE1_LIGHTING_PROMPTS = {
    "left": "light from the left",
    "middle": "light from the middle",
    "right": "light from the right",
    "top": "light from the top",
}


def _round_size(width: int, height: int):
    import math

    return math.ceil(width / 16) * 16, math.ceil(height / 16) * 16


def _cleanup_device_cache(device: str):
    import gc
    import torch

    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _run_stage1_edit_old_pipe(
    pipe,
    prompt: str,
    edit_image,
    seed: Optional[int],
    rand_device: str,
    num_inference_steps: int,
    cfg_scale: float,
    embedded_guidance: float,
    height: int,
    width: int,
    edit_image_auto_resize: bool,
):
    import torch

    with torch.inference_mode():
        return pipe(
            prompt,
            edit_image=edit_image,
            seed=seed,
            rand_device=rand_device,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            embedded_guidance=embedded_guidance,
            height=height,
            width=width,
            edit_image_auto_resize=edit_image_auto_resize,
        )


def generate_stage1_lighting_refs(
    source_images,
    lora_paths: dict[str, str],
    lora_scale: float,
    steps: int,
    cfg_scale: float,
    embedded_guidance: float,
    seed: Optional[int],
    rand_device: str,
    width: int,
    height: int,
    edit_image_auto_resize: bool,
    device: str,
    save_dir: Optional[Path] = None,
):
    """
    Generate stage-1 lighting references with old-stage1 semantics:
    a fresh full pipe per direction, fused LoRA, and direct pipe(...) execution.

    Returns side_images grouped by view:
      [[view1_left, view1_middle, view1_right, view1_top], ...]
    """
    source_images = [_resize(img, width=width, height=height) for img in source_images]
    by_direction: dict[str, list] = {direction: [] for direction in STAGE1_LIGHTING_ORDER}
    target_w, target_h = _round_size(width, height)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for direction in STAGE1_GENERATION_ORDER:
        lora_path = lora_paths[direction]
        print(f"[stage1] generating {direction} lighting refs with fused LoRA: {lora_path}")
        pipe = build_pipe(device)
        load_lora_into_pipe(pipe, lora_path, lora_scale)

        prompt = STAGE1_LIGHTING_PROMPTS[direction]
        for idx, img in enumerate(source_images, start=1):
            edited = _run_stage1_edit_old_pipe(
                pipe,
                prompt=prompt,
                edit_image=[img],
                seed=seed,
                rand_device=rand_device,
                num_inference_steps=steps,
                cfg_scale=cfg_scale,
                embedded_guidance=embedded_guidance,
                height=target_h,
                width=target_w,
                edit_image_auto_resize=edit_image_auto_resize,
            )
            by_direction[direction].append(edited)
            if save_dir is not None:
                out_path = save_dir / f"view{idx}_light_{direction}_klein.jpg"
                edited.save(out_path)
                print(f"[stage1] saved: {out_path}")
            _cleanup_device_cache(rand_device)

        del pipe
        _cleanup_device_cache(rand_device)

    _cleanup_device_cache(rand_device)
    return [
        [by_direction[direction][view_idx] for direction in STAGE1_LIGHTING_ORDER]
        for view_idx in range(len(source_images))
    ]


def _resize(img, width: int, height: int):
    from PIL import Image

    if img.size == (width, height):
        return img
    return img.resize((width, height), resample=Image.BICUBIC)


def load_images(paths: List[Path], width: int, height: int):
    from PIL import Image

    images = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGB")
        images.append(_resize(img, width=width, height=height))
    return images


def _group_side_paths(paths: List[Path], num_views: int, per_view: int) -> List[List[Path]]:
    if per_view > 0:
        expected = num_views * per_view
        if len(paths) != expected:
            raise ValueError(
                f"--side_ref expects {expected} images (num_views={num_views}, per_view={per_view}), got {len(paths)}"
            )
        k = per_view
    else:
        if len(paths) % num_views != 0:
            raise ValueError(
                f"--side_ref count {len(paths)} not divisible by num_views={num_views}; set --side_ref_per_view"
            )
        k = len(paths) // num_views
    return [paths[i * k : (i + 1) * k] for i in range(num_views)]


def encode_images_to_tokens(pipe, images):
    """
    Encode PIL images into VAE latents tokens: list of (1, S, C) tensors on pipe.device.
    """
    from einops import rearrange

    pipe.load_models_to_device(["vae"])
    tokens = []
    for img in images:
        x = pipe.preprocess_image(img)
        lat = pipe.vae.encode(x)  # (1, C, H, W)
        tok = rearrange(lat, "B C H W -> B (H W) C")
        tokens.append(tok)
    return tokens


def encode_prompt(pipe, prompt: str) -> dict:
    """
    Returns a dict with:
      - prompt_embeds: (1, L, D)
      - text_ids: (1, L, 4)
    """
    from diffsynth.pipelines.flux2_image import Flux2Unit_PromptEmbedder, Flux2Unit_Qwen3PromptEmbedder

    # Prefer Qwen3 text encoder when available (klein configs normally provide it).
    out = Flux2Unit_Qwen3PromptEmbedder().process(pipe, prompt)
    if out:
        return out
    out = Flux2Unit_PromptEmbedder().process(pipe, prompt)
    if out:
        return out
    raise RuntimeError("No text encoder available in the pipeline (both text_encoder and text_encoder_qwen3 are None).")


def build_image_ids(modality_idx: int, latent_h: int, latent_w: int, device, ref_index_scale: float = 10.0):
    """
    Build Flux2 DiT image ids for one modality.
    Shape: (1, latent_h*latent_w, 4), float32.

    IMPORTANT: This follows ComfyUI Flux2 `process_img` behavior (index ref method):
      - ids[..., 0] is a constant "index" = modality_idx * ref_index_scale
      - ids[..., 1] is the H coordinate (linspace 0..H-1)
      - ids[..., 2] is the W coordinate (linspace 0..W-1)
      - ids[..., 3] stays 0

    The stage2 training repo distinguishes modalities via:
      index = modality_idx * self.params.ref_index_scale
    where ref_index_scale is typically 10.0 for Flux2.
    """
    import torch

    index = float(modality_idx) * float(ref_index_scale)
    ids = torch.zeros((latent_h, latent_w, 4), device=device, dtype=torch.float32)
    # ComfyUI fills axis 0 as (axis1 + index). Since axis1 is 0 here, it's constant == index.
    ids[:, :, 0] = ids[:, :, 1] + index
    ids[:, :, 1] = torch.linspace(0, latent_h - 1, steps=latent_h, device=device, dtype=torch.float32).unsqueeze(1)
    ids[:, :, 2] = torch.linspace(0, latent_w - 1, steps=latent_w, device=device, dtype=torch.float32).unsqueeze(0)
    return ids.view(1, latent_h * latent_w, 4)


def save_views(images: List[Image.Image], output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = output.stem, output.suffix
    for i, im in enumerate(images, start=1):
        out_path = output.with_name(f"{stem}_view{i}{suffix}")
        im.save(out_path)
        print(f"Saved: {out_path}")


def run_multiview(
    pipe,
    prompt: str,
    negative_prompt: str,
    ref_images,
    extra_images,
    num_views: int,
    height: int,
    width: int,
    steps: int,
    embedded_guidance: float,
    cfg_scale: float,
    seed: Optional[int],
    rand_device: str,
    side_images=None,
    context_seed: Optional[int] = None,
    return_context: bool = False,
):
    import torch
    from einops import rearrange
    from diffsynth.pipelines.flux2_image import model_fn_flux2
    from tqdm import tqdm

    # Inference only.
    torch.set_grad_enabled(False)

    if num_views < 4:
        # Training convention uses 3 "other views" (modality ids 3..8).
        # Keep at least 4 views so we can select 3 context views.
        raise ValueError("This script currently supports --num_views >= 4 (fixed 3-context-view tokens).")

    height, width = pipe.check_resize_height_width(height, width)
    # If the pipeline rounded shapes up to multiples of 16, make sure conditioning images match.
    ref_images = [_resize(img, width=width, height=height) for img in ref_images]
    extra_images = [_resize(img, width=width, height=height) for img in extra_images]
    latent_h, latent_w = height // 16, width // 16
    tokens_per_view = latent_h * latent_w

    # Encode prompts
    pos = encode_prompt(pipe, prompt)
    neg = encode_prompt(pipe, negative_prompt)

    # Encode conditioning images
    ref_tokens = encode_images_to_tokens(pipe, ref_images)  # list[(1, S, C)]
    extra_tokens = encode_images_to_tokens(pipe, extra_images)
    if side_images is None:
        side_images = [[] for _ in range(num_views)]
    if len(side_images) != num_views:
        raise ValueError(f"side_images expects {num_views} view lists, got {len(side_images)}")
    side_k = len(side_images[0])
    if any(len(v) != side_k for v in side_images):
        raise ValueError("side_images must have the same number of items per view")
    side_images = [[_resize(img, width=width, height=height) for img in imgs] for imgs in side_images]
    side_tokens = [encode_images_to_tokens(pipe, imgs) for imgs in side_images] if side_k > 0 else []

    # Select 3 context views per main view (fixed token length to match training).
    context_k = 3
    if context_k > (num_views - 1):
        raise ValueError(f"context_k={context_k} exceeds available other views (num_views={num_views})")
    import random

    rng = random.Random(context_seed) if context_seed is not None else random.Random()
    context_by_view: List[List[int]] = []
    for v in range(num_views):
        others = [o for o in range(num_views) if o != v]
        if len(others) == context_k:
            chosen = others
        else:
            chosen = rng.sample(others, context_k)
        context_by_view.append(chosen)

    # Precompute ids once on device (modalities 0..(base+side_k-1)).
    base_modalities = 3 + 2 * context_k  # fixed 9
    total_modalities = base_modalities + side_k
    ids_by_mod = {
        m: build_image_ids(m, latent_h, latent_w, device=pipe.device, ref_index_scale=10.0)
        for m in range(total_modalities)
    }

    # Init per-view noise latents: (1, V, S, C)
    noise = pipe.generate_noise(
        (1, num_views, 128, latent_h, latent_w),
        seed=seed,
        rand_device=rand_device,
        rand_torch_dtype=pipe.torch_dtype,
    )
    latents = rearrange(noise, "B V C H W -> B V (H W) C")

    # Scheduler
    pipe.scheduler.set_timesteps(
        steps,
        denoising_strength=1.0,
        dynamic_shift_len=tokens_per_view,
    )

    # Denoise
    pipe.load_models_to_device(["dit"])

    for progress_id, timestep_scalar in enumerate(
        tqdm(pipe.scheduler.timesteps, desc="Denoising", total=len(pipe.scheduler.timesteps))
    ):
        timestep = timestep_scalar.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        noise_pred_views: List[torch.Tensor] = []
        for v in range(num_views):
            others = context_by_view[v]

            # Token sequence: main | main_control | main_extra | other_noisy | other_extra | side_controls
            tok_parts = [
                latents[:, v],  # modality 0
                ref_tokens[v],  # modality 1 (main only)
                extra_tokens[v],  # modality 2
                *(latents[:, o] for o in others),  # modality 3..5
                *(extra_tokens[o] for o in others),  # modality 6..8
            ]
            if side_k > 0:
                tok_parts.extend(side_tokens[v])  # modality 9..(9+K-1), main view only
            hidden_states = torch.cat(tok_parts, dim=1)

            id_parts = [ids_by_mod[i] for i in range(base_modalities)]
            if side_k > 0:
                id_parts.extend(ids_by_mod[base_modalities + i] for i in range(side_k))
            image_ids = torch.cat(id_parts, dim=1)

            # CFG (Flux2 is fast enough at cfg_scale=1.0, but keep parity with other scripts)
            noise_pos = model_fn_flux2(
                dit=pipe.dit,
                latents=hidden_states,
                timestep=timestep,
                embedded_guidance=embedded_guidance,
                prompt_embeds=pos["prompt_embeds"],
                text_ids=pos["text_ids"],
                image_ids=image_ids,
            )
            if cfg_scale != 1.0:
                noise_neg = model_fn_flux2(
                    dit=pipe.dit,
                    latents=hidden_states,
                    timestep=timestep,
                    embedded_guidance=embedded_guidance,
                    prompt_embeds=neg["prompt_embeds"],
                    text_ids=neg["text_ids"],
                    image_ids=image_ids,
                )
                noise = noise_neg + cfg_scale * (noise_pos - noise_neg)
            else:
                noise = noise_pos

            # Keep only main tokens (first S)
            noise_pred_views.append(noise[:, :tokens_per_view])

        noise_pred = torch.stack(noise_pred_views, dim=1)  # (1, V, S, C)
        latents = pipe.scheduler.step(noise_pred, timestep_scalar, latents)

    # Decode each view
    pipe.load_models_to_device(["vae"])
    latents_4d = rearrange(latents[0], "V (H W) C -> V C H W", H=latent_h, W=latent_w)
    decoded = pipe.vae.decode(latents_4d)  # (V, 3, H, W)
    images = [pipe.vae_output_to_image(decoded[v], pattern="C H W") for v in range(num_views)]
    pipe.load_models_to_device([])
    if return_context:
        return images, {
            "context_views": context_by_view,
            "context_k": context_k,
            "context_seed": context_seed,
        }
    return images


def main():
    parser = argparse.ArgumentParser(description="Run FLUX.2-klein multi-view inference.")
    parser.add_argument("--prompt", default="Turn it to photo", help="Prompt text.")
    parser.add_argument("--negative-prompt", "--negative_prompt", dest="negative_prompt", default="", help="Negative prompt text.")
    parser.add_argument(
        "--lighting-map",
        "--lighting_map",
        dest="ref_image",
        type=Path,
        nargs="+",
        required=True,
        help="4 lighting-map images, one per view.",
    )
    parser.add_argument(
        "--view-image",
        "--view_image",
        dest="extra_ref",
        type=Path,
        nargs="+",
        required=True,
        help="4 original-view images, one per view. Required for the default two-stage demo.",
    )
    parser.add_argument(
        "--side-ref",
        "--side_ref",
        dest="side_ref",
        type=Path,
        nargs="*",
        default=None,
        help="Optional side control images (grouped by view: view0 K imgs, view1 K imgs, ...).",
    )
    parser.add_argument(
        "--side-ref-per-view",
        "--side_ref_per_view",
        dest="side_ref_per_view",
        type=int,
        default=0,
        help="Number of side control images per view (optional; inferred if 0).",
    )
    parser.add_argument("--num-views", "--num_views", dest="num_views", type=int, default=4, help="Number of views (must be >=4).")
    parser.add_argument("--height", type=int, default=512, help="Output height (multiple of 16).")
    parser.add_argument("--width", type=int, default=768, help="Output width (multiple of 16).")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps.")
    parser.add_argument("--embedded-guidance", "--embedded_guidance", dest="embedded_guidance", type=float, default=1.0, help="Flux2 embedded guidance scale.")
    parser.add_argument("--cfg-scale", "--cfg_scale", dest="cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    parser.add_argument("--rand-device", "--rand_device", dest="rand_device", default="cuda", help="Device used for random sampling (noise init).")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:1")
    parser.add_argument(
        "--context-seed",
        type=int,
        default=None,
        help="Optional seed for selecting 3 context views when num_views > 4 (default: random).",
    )
    parser.add_argument(
        "--lora",
        default=DEFAULT_STAGE2_LORA,
        help="Stage-2 LoRA checkpoint. Defaults to the Lume-Palette stage-2 LoRA on Hugging Face.",
    )
    parser.add_argument("--lora-scale", "--lora_scale", dest="lora_scale", type=float, default=1.0, help="LoRA scaling factor (alpha).")
    parser.add_argument("--no-lora", "--no_lora", dest="no_lora", action="store_true", help="Disable LoRA loading.")
    parser.add_argument(
        "--stage1-lora-left",
        default=DEFAULT_STAGE1_LORA_LEFT,
        help="Stage-1 LoRA for left lighting refs. Defaults to the Lume-Palette stage-1 LoRA on Hugging Face.",
    )
    parser.add_argument(
        "--stage1-lora-middle",
        default=DEFAULT_STAGE1_LORA_MIDDLE,
        help="Stage-1 LoRA for middle lighting refs. Defaults to the Lume-Palette stage-1 LoRA on Hugging Face.",
    )
    parser.add_argument(
        "--stage1-lora-right",
        default=DEFAULT_STAGE1_LORA_RIGHT,
        help="Stage-1 LoRA for right lighting refs. Defaults to the Lume-Palette stage-1 LoRA on Hugging Face.",
    )
    parser.add_argument(
        "--stage1-lora-top",
        default=DEFAULT_STAGE1_LORA_TOP,
        help="Stage-1 LoRA for top lighting refs. Defaults to the Lume-Palette stage-1 LoRA on Hugging Face.",
    )
    parser.add_argument(
        "--stage1-source",
        type=Path,
        nargs="+",
        default=None,
        help="Original-view images for stage-1 lighting generation. Defaults to --view-image.",
    )
    parser.add_argument("--stage1-lora-scale", type=float, default=1.0, help="Stage-1 LoRA scaling factor.")
    parser.add_argument("--stage1-steps", type=int, default=50, help="Stage-1 lighting ref denoising steps.")
    parser.add_argument("--stage1-cfg-scale", type=float, default=1.0, help="Stage-1 CFG scale.")
    parser.add_argument("--stage1-embedded-guidance", type=float, default=1.0, help="Stage-1 embedded guidance scale.")
    parser.add_argument("--stage1-seed", type=int, default=1000, help="Stage-1 seed.")
    parser.add_argument(
        "--stage1-edit-auto-resize",
        action="store_true",
        help="Enable Flux2 internal auto resize/crop for stage-1 edit images. Off by default.",
    )
    parser.add_argument(
        "--stage1-save-dir",
        type=Path,
        default=None,
        help="Optional directory to save generated stage-1 lighting refs.",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Generate and save stage-1 lighting refs, then exit before stage 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/multiview_demo_real_klein/room.png"),
        help="Output base path; views are saved as *_view{i}.png",
    )
    args = parser.parse_args()

    if args.num_views < 4:
        raise SystemExit("--num_views must be >= 4 (fixed 3-context-view tokens).")
    if len(args.ref_image) != args.num_views:
        raise SystemExit(f"--lighting-map expects exactly {args.num_views} images, got {len(args.ref_image)}")
    if len(args.extra_ref) != args.num_views:
        raise SystemExit(f"--view-image expects exactly {args.num_views} images, got {len(args.extra_ref)}")
    if args.stage1_source is not None and len(args.stage1_source) != args.num_views:
        raise SystemExit(f"--stage1-source expects exactly {args.num_views} images, got {len(args.stage1_source)}")

    # Download FLUX.2-klein base models from Hugging Face by default.
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    # Xet can fail for large gated HF model files in some environments.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    ref_imgs = load_images(list(args.ref_image), width=args.width, height=args.height)
    extra_imgs = load_images(list(args.extra_ref), width=args.width, height=args.height)
    stage1_source_imgs = None
    if args.stage1_source is not None:
        stage1_source_imgs = load_images(list(args.stage1_source), width=args.width, height=args.height)

    pipe = None
    side_imgs = None
    if args.side_ref:
        side_paths = _group_side_paths(list(args.side_ref), args.num_views, args.side_ref_per_view)
        side_imgs = [load_images(paths, width=args.width, height=args.height) for paths in side_paths]
    stage1_loras = {
        "left": args.stage1_lora_left,
        "middle": args.stage1_lora_middle,
        "right": args.stage1_lora_right,
        "top": args.stage1_lora_top,
    }
    provided_stage1_loras = {k: v for k, v in stage1_loras.items() if v is not None}
    if provided_stage1_loras and side_imgs is None:
        missing = [name for name in STAGE1_LIGHTING_ORDER if stage1_loras[name] is None]
        if missing:
            raise SystemExit(
                "Stage-1 lighting generation requires all four LoRAs. "
                f"Missing: {', '.join(missing)}"
            )
        if stage1_source_imgs is None:
            stage1_source_imgs = extra_imgs
        side_imgs = generate_stage1_lighting_refs(
            source_images=stage1_source_imgs,
            lora_paths=stage1_loras,
            lora_scale=args.stage1_lora_scale,
            steps=args.stage1_steps,
            cfg_scale=args.stage1_cfg_scale,
            embedded_guidance=args.stage1_embedded_guidance,
            seed=args.stage1_seed if args.stage1_seed is not None else args.seed,
            rand_device=args.rand_device,
            width=args.width,
            height=args.height,
            edit_image_auto_resize=args.stage1_edit_auto_resize,
            device=args.device,
            save_dir=args.stage1_save_dir,
        )
        if args.stage1_only:
            if args.stage1_save_dir is None:
                raise SystemExit("--stage1-only requires --stage1-save-dir so the generated refs are available to stage 2.")
            print(f"[stage1] done. Lighting refs saved to: {args.stage1_save_dir}")
            return
    elif args.stage1_only:
        raise SystemExit("--stage1-only requires all four stage-1 LoRAs.")

    if pipe is None:
        pipe = build_pipe(args.device)
    maybe_load_lora(pipe, args.lora, args.lora_scale, enable=not args.no_lora)

    images = run_multiview(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        ref_images=ref_imgs,
        extra_images=extra_imgs,
        side_images=side_imgs,
        num_views=args.num_views,
        height=args.height,
        width=args.width,
        steps=args.steps,
        embedded_guidance=args.embedded_guidance,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        rand_device=args.rand_device,
        context_seed=args.context_seed,
    )

    save_views(images, args.output)


if __name__ == "__main__":
    main()
