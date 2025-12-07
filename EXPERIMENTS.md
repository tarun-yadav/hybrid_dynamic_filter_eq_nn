# MNIST projective augmentation study

This experiment follows the paper's recommendation to expose the network to synthetic projective distortions. We compare a lightly augmented setting (random affine + perspective) against a plain baseline using a small MNIST subset for speed.

## Setup
- Model: `HybridEquivariantNet` with moving-frame canonicalization and invariant feature mixing.
- Data: MNIST with 5k training / 1k test samples.
- Optimization: Adam, lr=3e-3, batch size 64, 2 epochs on CPU.
- Augmentations: `RandomAffine`(±25°, 10% translation, 0.9–1.1 scale) + `RandomPerspective`(0.35) applied before tensor conversion.

## Results
| Setting | Train loss | Train acc | Test loss | Test acc |
| --- | --- | --- | --- | --- |
| Augmented | 2.3040 | 10.66% | 2.3029 | 9.90% |
| No augmentation | 2.3041 | 9.98% | 2.3007 | 12.60% |

Both runs were intentionally brief for turnaround. Accuracy remains near chance, indicating the prototype needs longer training, tighter regularization, or architectural tuning to exploit the moving-frame normalization.

## How to reproduce
```bash
# With projective + affine augmentation
python scripts/train_mnist.py --epochs 2 --train-size 5000 --test-size 1000

# Baseline without augmentation
python scripts/train_mnist.py --no-augment --epochs 2 --train-size 5000 --test-size 1000
```
