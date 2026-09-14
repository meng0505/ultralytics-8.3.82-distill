# Repository audit for PRCD

## Provenance

- Working tree: `/home/mengfanlong/ultralytics-8.3.82-distill`
- Source version: `ultralytics.__version__ == 8.3.82`
- Commit hash: **unavailable**. The mounted `.git` directory is empty and the
  discovered parent repository has no valid `HEAD`. Consequently this workspace
  cannot provide trustworthy commit provenance or create the requested staged
  commits. File-level phase reports are used instead.
- The source currently contains no executable KD/PRCD implementation. Historical
  `runs/train/*/args.yaml` files contain several old `kd_*` options, but those
  options are not present in the current source/config and are not treated as an
  implementation baseline.

## OBB entry points

| Concern | Actual location |
|---|---|
| Public OBB trainer | `ultralytics/models/yolo/obb/train.py::OBBTrainer` |
| OBB model | `ultralytics/nn/tasks.py::OBBModel` |
| OBB criterion | `ultralytics/utils/loss.py::v8OBBLoss` |
| OBB head | `ultralytics/nn/modules/head.py::OBB` |
| Task dispatch | `ultralytics/models/yolo/model.py` |
| Base detection trainer | `ultralytics/models/yolo/detect/train.py` |
| Main optimization loop | `ultralytics/engine/trainer.py::_do_train` |
| OBB label formatting | `ultralytics/data/dataset.py` and `data/augment.py::Format` |

`OBBTrainer` is a thin `DetectionTrainer` subclass. It sets `task=obb`, builds
an `OBBModel`, and uses `OBBValidator`. `OBBModel` differs from
`DetectionModel` only in selecting `v8OBBLoss`.

## Actual head representation

The final `OBB` module first computes an angle branch per feature map, then
calls `Detect.forward`. During training it returns:

```text
(
  raw_maps: list[B, 4*reg_max + nc, H_l, W_l],
  angle:    B, ne, sum(H_l*W_l)
)
```

For all inspected weights, `reg_max=16`, `ne=1`, and therefore `no=64+nc`.
There is no independent objectness output. Raw class logits are
`raw_map[:, 4*reg_max:, :, :]`; CRC must use the GT-class sigmoid response.
The DFL regression part is `raw_map[:, :4*reg_max]`. The angle branch maps its
sigmoid to `[-pi/4, 3pi/4]`.

`Detect.forward` replaces entries in the input feature list with concatenated
raw maps. A head forward-pre-hook must save `tuple(args[0])` immediately.
Feature matching must use the head's actual `stride`, not a hard-coded module
number or list index.

## Measured 256-pixel tensor shapes

The following probes used the actual `best.pt` files and a `1x3x256x256`
tensor in training mode.

| Weight | Strides | Captured neck features | Raw maps |
|---|---|---|---|
| OGSOD-1.0 optical YOLOv8s-P2 | 4/8/16/32 | 64x64x64, 128x32x32, 256x16x16, 512x8x8 | 67x64x64, 67x32x32, 67x16x16, 67x8x8 |
| OGSOD-2.0 optical YOLOv8s | 8/16/32 | 128x32x32, 256x16x16, 512x8x8 | 67x32x32, 67x16x16, 67x8x8 |
| OSPRC optical YOLOv8s | 8/16/32 | 128x32x32, 256x16x16, 512x8x8 | 67x32x32, 67x16x16, 67x8x8 |
| OGSOD-2.0 SAR YOLOv8n | 8/16/32 | 64x32x32, 128x16x16, 256x8x8 | 67x32x32, 67x16x16, 67x8x8 |

Shape notation in the table omits batch: `C x H x W`. The OGSOD-1.0 teacher
has an additional stride-4 level. A standard three-level student can therefore
distill only on the common 8/16/32 strides unless it also has P2.

## Data loading and OBB geometry

`YOLODataset` derives label paths through `img2label_paths`, verifies images and
labels into a cache, and represents polygon OBB labels as `Instances.segments`.
The standard transform chain is:

```text
Mosaic -> CopyPaste -> RandomPerspective/LetterBox -> MixUp
-> Albumentations -> RandomHSV -> vertical flip -> horizontal flip -> Format
```

`Format(return_obb=True)` converts the transformed quadrilateral to normalized
`xywhr`. The stock `collate_fn` stacks `img`, concatenates labels and offsets
`batch_idx`.

Resetting the random state and applying the image-local chain twice synchronizes
LetterBox, perspective/affine and flips for a SAR/optical pair. Mix transforms
also request additional dataset samples, so `PairedOBBDataset` switches its
sample fetch by modality, restores both RNG and mosaic-buffer state, traces the
requested sample IDs, and asserts identical transformed OBBs. A paired-mosaic
test covers the four source identities and layout replay. The safe main
configuration still disables Mosaic/CopyPaste/MixUp so the first comparisons
match the profiled, controlled augmentation policy.

## Detection loss details

`v8OBBLoss` concatenates raw maps, splits DFL and class logits, creates anchors
from the actual strides, filters GT whose pixel width or height is below two,
and applies the rotated task-aligned assigner. Its returned values are:

```text
scalar = batch_size * (weighted box + weighted cls + weighted dfl)
items  = [weighted box, weighted cls, weighted dfl]
```

PRCD must add its scalar to the first result without changing the three
detection items. An empty-GT distillation result must remain connected to the
student graph but have exact zero value.

## AMP, DDP, resume and checkpoint behavior

- AMP surrounds both `preprocess_batch` and `model(batch)` in the base loop.
  PRCD's masks, probability distributions, prototype normalization, EMA and
  Gram matrices must explicitly enter float32.
- DDP wraps only `self.model`. A trainer-owned frozen teacher is loaded and
  moved independently on each rank and has no gradient synchronization.
- The distributed dataloader samples paired dataset records atomically, so
  pairing remains valid as long as each record already contains both paths.
- The base checkpoint stores the EMA model, optimizer, training arguments and
  metrics. Registered prototype buffers are therefore recoverable through the
  model state. Teacher path/hash and profile hash belong in training arguments.
- The base EMA deep-copies the student model. Registering the teacher as a
  student child would incorrectly copy it into checkpoints and optimizer model
  traversal; the teacher must remain trainer-owned.

## Existing dataset and checkpoint inventory

| Dataset | SAR / optical / labels | Splits | Relevant optical teacher |
|---|---:|---|---|
| OGSOD-2.0 | 14,250 / 14,250 / 14,250 train; 2,035 each val; 4,074 each test | train/val/test | `opt-teacher-yolov8s-ogsod-2.0-256-nomosaic/weights/best.pt` |
| OGSOD-1.0 | 14,664 each train; 3,667 each val | train/val | `opt-teacher-yolov8s-ogsod-1.0-256-nomosaic/weights/best.pt` |
| OSPRC | 2,640 each train; 660 each val | train/val | `opt-teacher-yolov8s-osprc-256-nomosaic/weights/best.pt` |
| external SFOC-100K-SOA subset | 4,252 pairs train; 1,822 pairs test | scene-disjoint train/test; val aliases test by policy | none |

The external SFOC subset has an authoritative `manifests/tiles.csv`. Its
summary declares 512x512 non-empty, pure SO-A pairs, two occupied semantic
classes, 24,126 train instances and 10,687 test instances. The source IDs
originally followed the six-class parent table (`0/3`); they have now been
explicitly and auditably remapped to contiguous `0=plane, 1=ship`, so all
current model/data configs use `nc=2`. The profiler validates rather than
blindly trusts those figures.

The repository `data.yaml` and `data_opt.yaml` currently point only to the
OGSOD-2.0 directory names. Dataset-specific paired YAML files are required to
avoid silently training a named checkpoint on whatever data happens to occupy
those generic paths.

## Lowest-intrusion implementation boundary

1. Leave `v8OBBLoss`, `OBBTrainer`, `OBBModel`, `OBB.forward` and the base
   optimization loop unchanged.
2. Add `PairedOBBDataset` and override only the distillation trainer's dataset
   builder and preprocessing.
3. Add `DistillOBBModel.loss`, calling the ordinary criterion first.
4. Attach capture hooks from the dedicated model/trainer, and isolate all PRCD
   math under `ultralytics/nn/distillation`.
5. Store trainable student adapters and prototype buffers in the distillation
   student; keep the teacher outside it.
6. Use a separate training launcher/config rather than changing ordinary
   `yolo obb train` semantics.

## Baseline reproducibility boundaries

The following locations can change baseline results and require equivalence
tests or must remain untouched:

- image/label transforms and random-number consumption;
- OBB polygon-to-`xywhr` formatting;
- raw head list mutation and criterion input;
- GT `<2 px` filtering in `v8OBBLoss`;
- optimizer parameter traversal and grouping;
- model `.train()` propagation to a registered teacher;
- AMP dtype/autocast boundaries;
- DDP world-size scaling;
- model EMA and resume restoration;
- validation using the SAR-only dataset/model;
- export task dispatch.

When distillation is disabled, the dedicated model will bypass capture-based
losses and call the same forward/criterion pair. The ordinary `OBBTrainer`
remains entirely unchanged and is the strongest equivalence reference.

## Audit limitations

- A trustworthy Git commit hash and phase commits are impossible with the
  mounted empty Git metadata.
- No SFOC optical teacher exists, so an SFOC distillation pilot is blocked until
  that teacher is trained.
- Pair-aware mosaic now passes the automated identity/layout/OBB replay test,
  but remains disabled by default until an AP ablation establishes its effect.
- No PRCD AP result is claimed by this audit.
