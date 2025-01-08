import os

from ultralytics import YOLO

model = YOLO("C:\\Users\\Ahmet\\Desktop\\Python\\cv project YOLOV11\\runs\\detect\\YOLO v11 trhird trial 200 epochs\\weights\\best.pt")

files = [i for i in os.listdir(os.getcwd()) if i.endswith(".png")]

for i in files:
    model.predict(source=i,show=True,save=True,line_width=2,save_crop=False,show_labels=True,show_conf=True,classes=[0,1,2])


input()