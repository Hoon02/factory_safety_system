import os
from ultralytics import YOLO
from roboflow import Roboflow

def main():
    rf = Roboflow(api_key="HH0kncgFE5I88s5NRB5A")
    project = rf.workspace("aiproject1-6ursd").project("3_tr14_valid")
    version = project.version(1)
    dataset = version.download("yolov9")

    # rf = Roboflow(api_key="HH0kncgFE5I88s5NRB5A")
    # project = rf.workspace("aiproject1-6ursd").project("7_tr21_valid")
    # version = project.version(1)
    # dataset = version.download("yolov9")

    # rf = Roboflow(api_key="HH0kncgFE5I88s5NRB5A")
    # project = rf.workspace("aiproject1-6ursd").project("0_te11_valid")
    # version = project.version(1)
    # dataset = version.download("yolov9")

    # YOLO 모델 로드
    model = YOLO("./yolo_forklift/best.pt")

    # 테스트 데이터셋 경로
    test_data_path = os.path.join(dataset.location, "test", "images")

    # 모델 평가
    results = model.val(data=dataset.location + "/data.yaml", split="test", workers=0)

    # mAP 출력
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()