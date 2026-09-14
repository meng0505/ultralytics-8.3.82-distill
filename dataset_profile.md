# Paired OBB dataset profile

> Historical combined 256-profile artifact. Its SFOC section predates the
> audited `0/3 -> 0/1` label remap and is not used by `train.py`. The current
> native-resolution two-class SFOC profile is
> `configs/prcd/profiles/sfoc512/dataset_profile.md`.

Profile hash: `18884c48b7f40f38d726f02eda884f2bfdc3f7e5f8ced3a9b01b4853a7a3102d`.

Pairing is based on explicit filename stems or the configured manifest; directory order is never used.

## ogsod-2.0

### train

- Complete pairs: 14250 (SAR 14250, optical 14250, labels 14250).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 11 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1, 2]` against 3 declared slots; out-of-range `[]`.
- Instances: 37929; per image mean 2.662; class frequencies `{'0': 23552, '1': 3614, '2': 10763}`.
- Equivalent side: median 8.81px, P90 23.84px; aspect-ratio median 2.05.
- Neighbor within two equivalent sides: 40.9%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 39.7% | 68.8% | 74.9% | 91.5% |
| P4/16 | 82.3% | 89.3% | 91.5% | 95.0% |
| P5/32 | 92.7% | 94.6% | 95.0% | 98.4% |

### val

- Complete pairs: 2035 (SAR 2035, optical 2035, labels 2035).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 1 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1, 2]` against 3 declared slots; out-of-range `[]`.
- Instances: 5430; per image mean 2.668; class frequencies `{'0': 3425, '1': 465, '2': 1540}`.
- Equivalent side: median 8.83px, P90 24.64px; aspect-ratio median 2.03.
- Neighbor within two equivalent sides: 39.8%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 40.1% | 67.8% | 76.2% | 91.1% |
| P4/16 | 81.3% | 89.1% | 91.1% | 95.0% |
| P5/32 | 92.6% | 94.6% | 95.0% | 98.3% |

### test

- Complete pairs: 4074 (SAR 4074, optical 4074, labels 4074).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 6 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1, 2]` against 3 declared slots; out-of-range `[]`.
- Instances: 10716; per image mean 2.630; class frequencies `{'0': 6725, '1': 1014, '2': 2977}`.
- Equivalent side: median 8.95px, P90 24.83px; aspect-ratio median 2.08.
- Neighbor within two equivalent sides: 39.5%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 38.3% | 67.4% | 73.9% | 91.1% |
| P4/16 | 81.6% | 88.6% | 91.1% | 94.4% |
| P5/32 | 92.2% | 94.1% | 94.4% | 98.2% |

Split identity leakage: `{'train<->val': {'count': 0, 'examples': []}, 'train<->test': {'count': 0, 'examples': []}, 'val<->test': {'count': 0, 'examples': []}}`.

### Recommended initial candidate

- Levels `['P3']` with `single` assignment: next-level area-under-one-cell ratio is 82.3%.
- Adaptive support radius 1–4 and grid 7: P3 P90 max dimension is 4.91 cells.
- ICD enabled `True`: largest candidate-batch per-class fractions with at least two instances are `[1.0, 1.0, 0.9982142857142857]`.
- Prototype momentum `0.99` and class-balanced sampling recommendation `True`.
- CRC/SPD/ICD loss weights are explicitly provisional candidates. Run the loss-scale probe before choosing them; dataset geometry alone cannot determine gradient balance.

## ogsod-1.0

### train

- Complete pairs: 14664 (SAR 14664, optical 14664, labels 14664).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 9 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1, 2]` against 3 declared slots; out-of-range `[]`.
- Instances: 38966; per image mean 2.657; class frequencies `{'0': 25533, '1': 3306, '2': 10127}`.
- Equivalent side: median 8.83px, P90 23.45px; aspect-ratio median 2.15.
- Neighbor within two equivalent sides: 38.8%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 39.0% | 69.7% | 76.3% | 92.4% |
| P4/16 | 83.2% | 89.6% | 92.4% | 95.1% |
| P5/32 | 93.0% | 94.6% | 95.1% | 98.3% |

### val

- Complete pairs: 3667 (SAR 3667, optical 3667, labels 3667).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 5 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1, 2]` against 3 declared slots; out-of-range `[]`.
- Instances: 9609; per image mean 2.620; class frequencies `{'0': 6389, '1': 803, '2': 2417}`.
- Equivalent side: median 9.00px, P90 24.19px; aspect-ratio median 2.19.
- Neighbor within two equivalent sides: 37.3%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 37.0% | 68.2% | 74.6% | 92.0% |
| P4/16 | 82.6% | 89.1% | 92.0% | 94.5% |
| P5/32 | 92.5% | 94.2% | 94.5% | 98.2% |

Split identity leakage: `{'train<->val': {'count': 0, 'examples': []}}`.

### Recommended initial candidate

- Levels `['P3']` with `single` assignment: next-level area-under-one-cell ratio is 83.2%.
- Adaptive support radius 1–4 and grid 7: P3 P90 max dimension is 4.97 cells.
- ICD enabled `True`: largest candidate-batch per-class fractions with at least two instances are `[1.0, 1.0, 0.9956521739130435]`.
- Prototype momentum `0.99` and class-balanced sampling recommendation `True`.
- CRC/SPD/ICD loss weights are explicitly provisional candidates. Run the loss-scale probe before choosing them; dataset geometry alone cannot determine gradient balance.

## osprc

### train

- Complete pairs: 2640 (SAR 2640, optical 2640, labels 2640).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 0 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1, 2]` against 3 declared slots; out-of-range `[]`.
- Instances: 13096; per image mean 4.961; class frequencies `{'0': 5226, '1': 1151, '2': 6719}`.
- Equivalent side: median 6.36px, P90 17.26px; aspect-ratio median 1.33.
- Neighbor within two equivalent sides: 59.9%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 68.7% | 83.2% | 84.9% | 93.0% |
| P4/16 | 89.2% | 92.2% | 93.0% | 97.2% |
| P5/32 | 94.7% | 97.4% | 97.2% | 99.9% |

### val

- Complete pairs: 660 (SAR 660, optical 660, labels 660).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 0 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 1, 2]` against 3 declared slots; out-of-range `[]`.
- Instances: 3184; per image mean 4.824; class frequencies `{'0': 1350, '1': 277, '2': 1557}`.
- Equivalent side: median 6.24px, P90 16.89px; aspect-ratio median 1.31.
- Neighbor within two equivalent sides: 56.8%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 70.3% | 84.4% | 86.0% | 93.4% |
| P4/16 | 89.7% | 92.6% | 93.4% | 97.9% |
| P5/32 | 95.4% | 98.1% | 97.9% | 99.8% |

Split identity leakage: `{'train<->val': {'count': 0, 'examples': []}}`.

### Recommended initial candidate

- Levels `['P3']` with `single` assignment: next-level area-under-one-cell ratio is 89.2%.
- Adaptive support radius 1–3 and grid 7: P3 P90 max dimension is 2.63 cells.
- ICD enabled `True`: largest candidate-batch per-class fractions with at least two instances are `[1.0, 1.0, 1.0]`.
- Prototype momentum `0.9` and class-balanced sampling recommendation `True`.
- CRC/SPD/ICD loss weights are explicitly provisional candidates. Run the loss-scale probe before choosing them; dataset geometry alone cannot determine gradient balance.

## sfoc-1.0-100k-soa

### train

- Complete pairs: 4252 (SAR 4252, optical 4252, labels 4252).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 0 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 3]` against 4 declared slots; out-of-range `[]`.
- Instances: 24126; per image mean 5.674; class frequencies `{'0': 9176, '3': 14950}`.
- Equivalent side: median 36.27px, P90 113.34px; aspect-ratio median 2.44.
- Neighbor within two equivalent sides: 79.8%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 24.8% | 34.1% | 38.6% | 62.4% |
| P4/16 | 45.0% | 57.4% | 62.4% | 82.4% |
| P5/32 | 70.9% | 81.1% | 82.4% | 96.7% |

### test

- Complete pairs: 1822 (SAR 1822, optical 1822, labels 1822).
- Missing IDs: SAR 0, optical 0, labels 0.
- Duplicate IDs: SAR 0, optical 0, labels 0.
- Corrupt images: SAR 0, optical 0; unequal pair dimensions 0.
- Label warning records: 0 (degenerate OBBs are skipped; exact duplicate annotations are counted once).
- Class IDs: observed `[0, 3]` against 4 declared slots; out-of-range `[]`.
- Instances: 10687; per image mean 5.866; class frequencies `{'0': 4013, '3': 6674}`.
- Equivalent side: median 43.66px, P90 117.22px; aspect-ratio median 2.42.
- Neighbor within two equivalent sides: 79.4%.

| Level | Area <1 cell | Area <2 cells | min dimension <1 | min dimension <2 |
|---|---:|---:|---:|---:|
| P3/8 | 31.1% | 35.8% | 39.6% | 55.2% |
| P4/16 | 42.2% | 51.2% | 55.2% | 75.0% |
| P5/32 | 61.9% | 73.8% | 75.0% | 96.5% |

Split identity leakage: `{'train<->test': {'count': 0, 'examples': []}}`.

### Recommended initial candidate

- Levels `['P3', 'P4']` with `scale_assigned` assignment: next-level area-under-one-cell ratio is 45.0%.
- Adaptive support radius 1–4 and grid 7: P3 P90 max dimension is 8.93 cells.
- ICD enabled `True`: largest candidate-batch per-class fractions with at least two instances are `[1.0, 1.0]`.
- Prototype momentum `0.9` and class-balanced sampling recommendation `False`.
- CRC/SPD/ICD loss weights are explicitly provisional candidates. Run the loss-scale probe before choosing them; dataset geometry alone cannot determine gradient balance.

## Interpretation constraints

Feature-cell statistics describe spatial support after the configured letterbox scale. They do not imply that high-frequency SAR response or local context is noise. The recommendation favors region-level support because cross-modal pixel responses are not assumed to match.
