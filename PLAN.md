# Implementation plan

1. Parse the reference paper into markdown (blocked locally by heavy OCR runtime; models were downloaded via `marker`, but conversion stalled). Replace with lightweight prototype capturing the moving-frame plus invariant-filter design.
2. Build reusable geometry utilities for PGL(3) actions (`src/projective.py`).
3. Implement a moving-frame estimator that predicts control points and canonicalizes inputs (`src/moving_frame.py`).
4. Add differential invariant feature mixing (`src/invariant_features.py`).
5. Wire everything into a hybrid classifier backbone with logits and auxiliary homography outputs (`src/model.py`).
6. Provide a smoke test to verify tensor shapes and differentiability (`tests/test_model.py`).
