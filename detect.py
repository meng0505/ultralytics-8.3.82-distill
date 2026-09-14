import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/mengfanlong/ultralytics-8.3.82-distill/runs/obb/osprc-opt-512-yolov8n/weights/best.pt') # select your model.pt path
    # model = YOLO('yolov8n-cls.pt')  # select your model.pt path
    model.predict(source=
                  r'/home/mengfanlong/ultralytics-8.3.82-distill/dataset/images/val',
                  imgsz=512,
                  project='/home/mengfanlong/ultralytics-8.3.82-distill/runs/detect',

                  name='exp',
                  save=True,              
                  line_width=2,
                  show_labels=True,
                  show_conf=True,     # ✅显示置信度


                )