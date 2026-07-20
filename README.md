<h1 align="center">Lume-Palette</h1>

<h3 align="center">
Decoupled Illumination Priors for Spatially Controllable Multi-View Indoor Scene Relighting
</h3>

<p align="center">
Chenjian Gao, Linning Xu, Tianfan Xue
</p>

<p align="center">
<strong>ECCV 2026</strong>
</p>

## Overview

Lume-Palette uses two stages:

- **Stage 1: illumination distillation**
  Generates canonical lighting reference images for each input view.

- **Stage 2: illumination casting**
  Uses lighting-map images, original-view images, and Stage-1 lighting references to produce the final multi-view relighting results.

The provided demo runs on the prepared ScanNet++ example in `demo_inputs/`.

## Environment

Create a Python environment with CUDA PyTorch:

```bash
conda create -n lumepalette python=3.10 -y
conda activate lumepalette

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install \
  transformers safetensors Pillow numpy einops tqdm \
  sentencepiece protobuf accelerate peft huggingface_hub ftfy pandas datasets imageio
```

## Run

The demo contains four input views and four lighting maps:

- `--view-image`: original indoor views.
- `--lighting-map`: target lighting controls for the same views.
- `--side-ref`: Stage-1 lighting references used by Stage 2. When omitted, the script generates them automatically with Stage 1.

### Run Stage 1 + Stage 2

Run the full pipeline on the prepared demo scene:

```bash
python run_lumepalette.py \
  --lighting-map \
  demo_inputs/lighting_map/lighting_map_1.png \
  demo_inputs/lighting_map/lighting_map_2.png \
  demo_inputs/lighting_map/lighting_map_3.png \
  demo_inputs/lighting_map/lighting_map_4.png \
  --view-image \
  demo_inputs/view_1.png \
  demo_inputs/view_2.png \
  demo_inputs/view_3.png \
  demo_inputs/view_4.png \
  --stage1-save-dir \
  output/stage1_refs \
  --output \
  output/demo.png
```

This first generates Stage-1 lighting references in:

```text
output/stage1_refs/view1_light_left_klein.jpg
output/stage1_refs/view1_light_middle_klein.jpg
output/stage1_refs/view1_light_right_klein.jpg
output/stage1_refs/view1_light_top_klein.jpg
...
output/stage1_refs/view4_light_top_klein.jpg
```

Then Stage 2 writes the relit multi-view outputs to:

```text
output/demo_view1.png
output/demo_view2.png
output/demo_view3.png
output/demo_view4.png
```

### Run Stage 1 Only

Stage 1 generates four canonical lighting references for each input view: left, middle, right, and top.

```bash
python run_lumepalette.py \
  --stage1-only \
  --stage1-save-dir \
  output/stage1_refs \
  --lighting-map \
  demo_inputs/lighting_map/lighting_map_1.png \
  demo_inputs/lighting_map/lighting_map_2.png \
  demo_inputs/lighting_map/lighting_map_3.png \
  demo_inputs/lighting_map/lighting_map_4.png \
  --view-image \
  demo_inputs/view_1.png \
  demo_inputs/view_2.png \
  demo_inputs/view_3.png \
  demo_inputs/view_4.png
```

By default, Stage 1 uses `--view-image` as the source images. To use a different set of source images for Stage 1, pass them with `--stage1-source`.

The expected Stage-1 output files are:

```text
output/stage1_refs/view1_light_left_klein.jpg
output/stage1_refs/view1_light_middle_klein.jpg
output/stage1_refs/view1_light_right_klein.jpg
output/stage1_refs/view1_light_top_klein.jpg
output/stage1_refs/view2_light_left_klein.jpg
output/stage1_refs/view2_light_middle_klein.jpg
output/stage1_refs/view2_light_right_klein.jpg
output/stage1_refs/view2_light_top_klein.jpg
output/stage1_refs/view3_light_left_klein.jpg
output/stage1_refs/view3_light_middle_klein.jpg
output/stage1_refs/view3_light_right_klein.jpg
output/stage1_refs/view3_light_top_klein.jpg
output/stage1_refs/view4_light_left_klein.jpg
output/stage1_refs/view4_light_middle_klein.jpg
output/stage1_refs/view4_light_right_klein.jpg
output/stage1_refs/view4_light_top_klein.jpg
```

### Run Stage 2 Only

After Stage 1 has produced the lighting references, pass them to Stage 2 with `--side-ref`.

`--side-ref` must be grouped by view. For each view, pass the four references in this order:

```text
left, middle, right, top
```

For the demo, run:

```bash
python run_lumepalette.py \
  --lighting-map \
  demo_inputs/lighting_map/lighting_map_1.png \
  demo_inputs/lighting_map/lighting_map_2.png \
  demo_inputs/lighting_map/lighting_map_3.png \
  demo_inputs/lighting_map/lighting_map_4.png \
  --view-image \
  demo_inputs/view_1.png \
  demo_inputs/view_2.png \
  demo_inputs/view_3.png \
  demo_inputs/view_4.png \
  --side-ref-per-view 4 \
  --side-ref \
  output/stage1_refs/view1_light_left_klein.jpg \
  output/stage1_refs/view1_light_middle_klein.jpg \
  output/stage1_refs/view1_light_right_klein.jpg \
  output/stage1_refs/view1_light_top_klein.jpg \
  output/stage1_refs/view2_light_left_klein.jpg \
  output/stage1_refs/view2_light_middle_klein.jpg \
  output/stage1_refs/view2_light_right_klein.jpg \
  output/stage1_refs/view2_light_top_klein.jpg \
  output/stage1_refs/view3_light_left_klein.jpg \
  output/stage1_refs/view3_light_middle_klein.jpg \
  output/stage1_refs/view3_light_right_klein.jpg \
  output/stage1_refs/view3_light_top_klein.jpg \
  output/stage1_refs/view4_light_left_klein.jpg \
  output/stage1_refs/view4_light_middle_klein.jpg \
  output/stage1_refs/view4_light_right_klein.jpg \
  output/stage1_refs/view4_light_top_klein.jpg \
  --output \
  output/demo_stage2.png
```

Stage 2 writes:

```text
output/demo_stage2_view1.png
output/demo_stage2_view2.png
output/demo_stage2_view3.png
output/demo_stage2_view4.png
```

## Acknowledgements

This codebase is built on top of [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio). We thank the DiffSynth-Studio authors and contributors for their open-source work.
