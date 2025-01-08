from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data="dataset.yaml",imgsz=416,batch=16,workers=0,device=0,epochs=500,name="YOLO v11 500 epochs")


