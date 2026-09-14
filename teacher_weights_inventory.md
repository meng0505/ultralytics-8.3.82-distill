# Teacher and SAR baseline weight inventory

以下 SHA256 由当前磁盘文件计算。训练目录名表明数据集归属，但旧 `args.yaml`
引用的是可变的通用 `dataset/data*.yaml`，没有保存当时数据清单 hash；因此新
PRCD run 额外记录 teacher 文件 hash 和当前 profile hash。

| 角色 | 数据集 | checkpoint | SHA256 |
|---|---|---|---|
| optical teacher | OGSOD-1.0 | `runs/train/opt-teacher-yolov8s-ogsod-1.0-256-nomosaic/weights/best.pt` | `e3cb02eb48f9982360552ffa1d334a6dd8da30553040cdd64ff4b139366edb0b` |
| optical teacher | OGSOD-2.0 | `runs/train/opt-teacher-yolov8s-ogsod-2.0-256-nomosaic/weights/best.pt` | `89ab7f79914f2a7873f180a187e2337acf36a7cca526708f3b6b6c7b1480287b` |
| optical teacher | OSPRC | `runs/train/opt-teacher-yolov8s-osprc-256-nomosaic/weights/best.pt` | `77518b95b68fc9b25912b49d9e8d523c95cc0b0b6044cb1cbf3aad60e1760886` |
| SAR baseline | OGSOD-1.0 | `runs/train/sar-baseline-raw-yolov8n-ogsod-1.0-256-nomosaic/weights/best.pt` | `a650e6d9d7f16f350c412c970e199110b1067fdbe6873e5f08523de691688331` |
| SAR baseline | OGSOD-2.0 | `runs/train/sar-baseline-raw-nomosaic-yolov8n-ogsod-2.0-256/weights/best.pt` | `18cb41ae92e30b1c41bb7364a4846db18d889e396d1cef95141b19f0227d9ddb` |
| SAR baseline | OSPRC | `runs/train/sar-baseline-raw-nomosaic-yolov8n-osprc-256/weights/best.pt` | `0d929e7b62963d80ac21252438ca04a9074bd06162c861bc2f68b188d495e92a` |

OGSOD-1.0 teacher 是 P2/P3/P4/P5 head；标准 YOLOv8n student 是 P3/P4/P5，
所以按实际 stride 交集匹配。主配置仍因数据统计选择 P3，不会把 teacher 的
list index 0 错配到 student 的 list index 0。

SFOC 当前没有 optical teacher 或 SAR baseline，本仓库没有声称其蒸馏结果。
