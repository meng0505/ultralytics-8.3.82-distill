# PRCD implementation plan

## Scope

This repository will gain an optically guided SAR OBB training path implementing
Prototype-Referenced Cross-Modal Distillation (PRCD). The ordinary Ultralytics
OBB trainer, model loss and inference path remain the baseline reference. A
distillation-disabled run must take that original path.

The implementation follows three principles:

1. pair optical and SAR samples by an explicit sample identity, never by sorted
   positional `zip`;
2. align teacher and student feature maps by their actual head stride, not by a
   YAML layer number or feature-list position;
3. make CRC, SPD and ICD independent switches and keep all teacher-only state
   outside the exported SAR detector.

## Phases

### 1. Repository and artefact audit

- Record the installed source version, OBB entry points, raw head structure,
  dataloader/augmentation flow and checkpoint behavior.
- Inspect teacher and baseline checkpoint metadata and run a 256-pixel shape
  probe on every relevant architecture.
- Record unavailable provenance instead of inventing a commit hash.

Exit criterion: `repository_audit.md` describes a minimally invasive design and
all known baseline-reproducibility boundaries.

### 2. Pair validation and dataset profiling

- Add a reusable pair index supporting an explicit CSV/JSON manifest, exact
  sample-ID matching and a registered filename mapper.
- Add `tools/validate_pair_alignment.py`.
- Add `tools/profile_paired_obb_dataset.py` and produce integrity, OBB geometry,
  feature-support, density and batch-feasibility statistics.
- Generate a finite, evidence-annotated recommended configuration. Values are
  initial candidates, not claims of optimality.

Exit criterion: all three in-repository paired datasets and the external SFOC
dataset can be audited without relying on directory order.

### 3. Paired OBB data path

- Subclass `YOLODataset` for paired training.
- Load the optical counterpart by sample ID and apply the same image-local
  geometric random state as SAR.
- Return the ordinary `img` key for the SAR branch plus `img_opt`,
  `sar_path`, `opt_path` and `sample_id`.
- Replay mix-transform sample selection through a modality-aware paired fetch,
  and assert that both branches used the same source sample IDs and transformed
  OBBs.
- Keep validation SAR-only.

The first production configuration still uses `mosaic=0`, `mixup=0` and
`copy_paste=0`: the full-data geometry statistics and loss probe were produced
without mix transforms, and the initial PRCD comparisons must keep augmentation
budgets controlled. Pair-aware mosaic is implemented and covered by an
identity/layout/OBB replay test, but remains an explicit ablation rather than a
silent default.

### 4. Feature and response capture

- Capture the input tuple of the final OBB head with a forward pre-hook.
- Copy the tensor references immediately so the head's in-place list rewrite
  cannot change the captured feature list.
- Parse raw classification logits from the last `nc` channels of each raw head
  map; keep DFL (`4 * reg_max`) and angle outputs separately.
- Match levels through the numeric stride intersection between teacher and
  student.

### 5. Object support

- Implement continuous-center grid sampling in feature coordinates.
- Implement `hard_gt_roi`, fixed 3/5/7, adaptive and scale-assigned modes with a
  common `K x K` result.
- Construct supersampled soft rotated-core masks and context masks, with tiny
  object fallback and optional neighbor-core exclusion.
- Make assignments and every skip reason observable.

### 6. CRC, SPD and ICD

- CRC transfers the region-level GT-class core/context distribution in float32.
- Student features pass through learnable lightweight adapters; teacher
  features use fixed normalization and non-trainable dimension matching.
- SPD maintains `(class, stride)` teacher EMA prototype buffers, updated from
  synchronized sums and counts under DDP.
- ICD compares deterministic, class-internal prototype-centered Gram matrices
  and skips groups with inadequate support.

### 7. Trainer, state and export

- Add a dedicated `DistillOBBTrainer` and `DistillOBBModel`.
- The trainer owns the frozen optical teacher; it is not a registered child of
  the student model and is not passed to the optimizer or checkpoint EMA.
- The student model owns adapters and prototype buffers so optimizer, DDP, EMA
  and resume use standard PyTorch state handling.
- The trainer computes detached teacher payloads in `preprocess_batch`; the
  model computes the baseline detection loss and optional distillation loss.
- Export copies only the SAR detection model state and numerically checks it.

### 8. Analysis, ablations and documentation

- Add loss-magnitude/gradient recommendation and optical-shift hooks.
- Generate controlled ablation YAMLs rather than an unbounded search.
- Add dataset/support/response/prototype/configuration visualizations.
- Provide smoke-test, baseline-equivalence, known-limitations and changed-file
  reports.

## Verification matrix

| Area | Verification |
|---|---|
| Pairing | exact IDs, duplicate/missing pairs, manifest parsing, split leakage |
| Geometry | identical transformed OBBs, rotations/flips, continuous centers |
| Support | fixed/adaptive shapes, soft core, neighbor exclusion, tiny/empty GT |
| CRC | normalized probabilities, finite AMP path, differentiable empty zero |
| SPD | EMA update/skip, DDP sum-count synchronization, checkpoint restore |
| ICD | single-instance skip, deterministic cap, diagonal exclusion |
| Teacher | eval mode, frozen parameters, no gradient, no optimizer/checkpoint |
| Baseline | outputs, detection loss and parameters identical when disabled |
| Export | no teacher/distiller/prototype keys and equal SAR predictions |

## Data-driven decisions to make after profiling

- P3-only versus P3+P4 or scale assignment;
- adaptive radius quantiles, output grid and context margin;
- whether the class/batch co-occurrence supports ICD;
- maximum instances per class and class-balanced sampling;
- prototype momentum candidates;
- loss-weight candidates based on raw magnitude and student-neck gradient norm.

No pilot score will be reported unless it is actually run with the named
checkpoint, data split, seed and budget.
