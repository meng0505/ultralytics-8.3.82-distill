from ultralytics.data.converter import convert_dota_to_yolo_obb

convert_dota_to_yolo_obb(r'/media/mengfanlong/111E18BA111E18BA/数据集/OGSOD-1.0')
# 关于dataobb文件下的目录下面会详细说明
# 整成下面的格式
# - DOTA
# ├─ images
# │   ├─ train
# │   └─ val
# └─ labels
# │ ─ train_original
# └ ─ val_original
#
# After
# execution, the
# function
# will
# organize
# the
# labels
# into:
#
# - DOTA
# └─ labels
# ├─ train
# └─ val