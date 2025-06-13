from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\python.learm\shixun\day5\ultralytics-main\runs\detect\train\weights\best.pt")
    model.val(data="coco8.yaml",imgsz=640,epochs=20,batch=16)