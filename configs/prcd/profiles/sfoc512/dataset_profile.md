# Paired OBB dataset profile

Profile hash: `ec5e5e40997390e89bb689f11280cfcf9257f0ee7582b83222fcdf553825528d`.

Pairing is based on explicit filename stems or the configured manifest; directory order is never used.

## sfoc-1.0-100k-soa-512

### train

- Complete pairs: 4252 (SAR 4252, optical 4252, labels 4252).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 0 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1]` against 2 declared slots; out-of-range `[]`.
- Instances: 24126; per image mean 5.674; class frequencies `{'0': 9176, '1': 14950}`.
- Equivalent side: median 36.27px, P90 113.34px; aspect-ratio median 2.44.
- Neighbor within two equivalent sides: 79.8%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 10.6% | 19.7% | 21.9% | 38.6% |
| P4/16 | 24.8% | 34.1% | 38.6% | 62.4% |
| P5/32 | 45.0% | 57.4% | 62.4% | 82.4% |

### test

- Complete pairs: 1822 (SAR 1822, optical 1822, labels 1822).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 0 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1]` against 2 declared slots; out-of-range `[]`.
- Instances: 10687; per image mean 5.866; class frequencies `{'0': 4013, '1': 6674}`.
- Equivalent side: median 43.66px, P90 117.22px; aspect-ratio median 2.42.
- Neighbor within two equivalent sides: 79.4%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 16.4% | 28.3% | 29.9% | 39.6% |
| P4/16 | 31.1% | 35.8% | 39.6% | 55.2% |
| P5/32 | 42.2% | 51.2% | 55.2% | 75.0% |

Split identity leakage: `{'train<->test': {'count': 0, 'examples': []}}`.

### Recommended initial candidate

- Levels `['P3', 'P4']` with `scale_assigned` assignment: next-level area-under-one-cell ratio is 24.8%.
- Adaptive support radius 1–4 and grid 7: P3 P90 max dimension is 17.87 cells.
- ICD enabled `True`: largest candidate-batch per-class fractions with at least two instances are `[1.0, 1.0]`.
- Prototype momentum `0.99` and class-balanced sampling recommendation `False`.
- CRC/SPD/ICD loss weights are explicitly provisional candidates. Run the loss-scale probe before choosing them; dataset geometry alone cannot determine gradient balance.

## Interpretation constraints

Feature-cell statistics describe spatial support after the configured letterbox scale. They do not imply that high-frequency SAR response or local context is noise. The recommendation favors region-level support because cross-modal pixel responses are not assumed to match.
