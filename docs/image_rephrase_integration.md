# Image Rephrase Attack

`ImageRephraseAttack` is a framework-native black-box attack evaluator for
MLLMU image-text questions. It reads the MLLMU `Test_Set`, restricts rows to
the IDs in `forget_10` by default, and evaluates several image variants
directly. It does not run a clean-image baseline. Results and transformed
images are stored below the evaluation output directory:

```text
<output_dir>/
  ImageRephrase_EVAL.json
  ImageRephrase_SUMMARY.json
  checkpoint/image_rephrase/<variant>/*.png
```

## Variants

The default, model-free variants are:

- `noise`: seeded Gaussian noise with `noise_sigma` measured in `[0, 1]` pixel
  intensity units (`0.02` is a small perturbation).
- `brightness`: multiplies image brightness by `brightness_factor` (`0.85` by
  default).
- `contrast`: multiplies image contrast by `contrast_factor` (`0.85` by
  default).
- `resize`: downsizes by `resize_scale` and resizes back to the original
  dimensions.
- `crop`: takes a centered crop covering `crop_ratio` and resizes it back to
  the original dimensions.
- `blur`: applies a light Gaussian blur using `rephrase_blur_radius`.
- `occlusion`: a seeded rectangle covering `occlusion_ratio` of the image,
  filled with the image mean, black, white, or a blurred image.
- `remove_irrelevant`: fills a supplied `remove_mask`; when no mask is present,
  the configured normalized corner region is used as a conservative proxy for
  an irrelevant background object.

The `regenerate` variant is also supported, but it requires an explicit source:

1. `record["regenerated_image"]` supplied by a caller;
2. `image_rephrase.regeneration_dir` containing `<record-id>.png` (or jpg,
   jpeg, webp); or
3. a `regeneration_fn` passed by Python code.

This keeps the benchmark reproducible and prevents an unconfigured diffusion
model from changing the subject identity. To enable it in a Hydra override:

```yaml
eval:
  image_rephrase:
    image_rephrase:
      variants: [noise, occlusion, remove_irrelevant, regenerate]
      regeneration_dir: /path/to/precomputed/generated/images
```

## Running with LLaVA-7B

From the repository root, in the `unlearning` environment:

```bash
conda run -n unlearning env PYTHONPATH=src python src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/mllmubench/image_rephrase_llava7b \
  paths.output_dir=/path/to/checkpoint
```

The evaluator reports per-variant recovery and `attack_success_at_b`, which is
the fraction of records recovered by at least one configured variant.
