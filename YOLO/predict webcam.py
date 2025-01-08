from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:\\Users\\Ahmet\\Desktop\\Python\\cv project YOLOV11\\runs\\detect\\YOLO v11 trhird trial 200 epochs\\weights\\best.pt")

# Start video capture from webcam (replace 0 with the webcam index if you have multiple)
results = model(0,show=True)


