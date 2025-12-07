# hybrid_dynamic_filter_eq_nn

Prototype implementation of a hybrid dynamic filter architecture that uses moving frames and differential invariants to encourage equivariance to the non-compact projective group PGL(3,R).

## What is here
- Projective geometry utilities (`src/projective.py`) for constructing differentiable homographies and warping inputs.
- A moving-frame module (`src/moving_frame.py`) that predicts control points and canonicalizes inputs through a projective warp.
- Invariant feature mixing blocks (`src/invariant_features.py`) that combine gradient-based invariants with learnable filters.
- A compact classifier backbone (`src/model.py`) that exposes logits plus diagnostic homography and warped patches.
- A smoke test (`tests/test_model.py`) to verify the forward pass and tensor shapes.

## Quickstart
```bash
pip install -r requirements.txt
pytest

# run the small MNIST experiment with projective augmentation
python scripts/train_mnist.py --epochs 2 --train-size 5000 --test-size 1000
```

## Notes
Attempts to convert the reference paper to markdown with `marker` were hindered by heavy OCR runtime, but the pipeline now includes the geometry and normalization pieces needed to experiment with projective-equivariant models.
