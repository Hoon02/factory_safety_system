import argparse
import cv2
import numpy as np
import os
import torch
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
import time
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Multiple YOLO models and Depth Anything V2')
    parser.add_argument('--video-path', type=str, default='../../dataset/custom/video/forklift/7_tr21.mp4')
    parser.add_argument('--input-size', type=int, default=384)
    parser.add_argument('--outdir', type=str, default='./result')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--yolo-model-1', type=str, default='./yolo_forklift/best.pt')
    parser.add_argument('--yolo-model-2', type=str, default='./yolo_person/best.pt')
    parser.add_argument('--yolo-model-3', type=str, default='./yolo_machine/best.pt')
    parser.add_argument('--process-every-n-frames', type=int, default=10)
    return parser.parse_args()

def load_models(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo_model_1 = YOLO(args.yolo_model_1, verbose=False)
    yolo_model_2 = YOLO(args.yolo_model_2, verbose=False)
    yolo_model_3 = YOLO(args.yolo_model_3, verbose=False)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_model = DepthAnythingV2(**model_configs[args.encoder])
    depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()

    return yolo_model_1, yolo_model_2, yolo_model_3, depth_model

def get_depth_value(depth, x1, y1, x2, y2, method='trimmed_mean'):
    height, width = depth.shape

    def get_valid_pixels(x1, y1, x2, y2):
        xx, yy = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        return xx[valid], yy[valid]

    xx, yy = get_valid_pixels(x1, y1, x2, y2)
    if len(xx) == 0:
        return None
    valid_depths = depth[yy, xx]
    sorted_depths = np.sort(valid_depths)
    trim_size = max(1, len(sorted_depths) // 10)
    trimmed = sorted_depths[trim_size:-trim_size] if len(sorted_depths) > 2 * trim_size else sorted_depths
    return np.mean(trimmed)

def calculate_proximity(objects, base_distance_threshold, depth_threshold):
    proximity_pairs = []
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects[i + 1:], start=i + 1):
            center1 = ((obj1['x1'] + obj1['x2']) / 2, (obj1['y1'] + obj1['y2']) / 2)
            center2 = ((obj2['x1'] + obj2['x2']) / 2, (obj2['y1'] + obj2['y2']) / 2)
            distance_2d = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

            avg_depth = (obj1['depth'] + obj2['depth']) / 2
            distance_threshold = base_distance_threshold * (100 - avg_depth) / 50

            depth_diff = abs(obj1['depth'] - obj2['depth'])

            if (distance_2d <= distance_threshold) and (depth_diff <= depth_threshold):
                label1, label2 = obj1['label'].split('_')[0], obj2['label'].split('_')[0]
                if ('person' in (label1, label2)) and ('machine' in (label1, label2) or 'forklift' in (label1, label2)):
                    proximity_pairs.append((obj1, obj2, distance_2d))

    return proximity_pairs

def process_frame(frame, yolo_model_1, yolo_model_2, yolo_model_3, depth_model, args, frame_number):
    results_1 = yolo_model_1(frame, conf=0.4, verbose=False)
    results_2 = yolo_model_2(frame, conf=0.4, verbose=False)
    results_3 = yolo_model_3(frame, conf=0.4, verbose=False)

    depth = depth_model.infer_image(frame, args.input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_color = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    class_counts = {}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    objects = []

    for i, results in enumerate([results_1, results_2, results_3]):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), colors[i], 2)

                class_name = r.names[int(box.cls)]

                if class_name not in class_counts:
                    class_counts[class_name] = 1
                else:
                    class_counts[class_name] += 1

                label = f"{class_name}_{class_counts[class_name]}_M{i + 1}"

                depth_value = get_depth_value(depth, x1, y1, x2, y2, method='trimmed_mean')
                if depth_value is not None:
                    depth_value = int(((255 - depth_value) / 255) * 100)
                    objects.append({
                        'label': label,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'depth': depth_value
                    })
                else:
                    continue

                cv2.putText(frame, f"{label} Depth: {depth_value}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                cv2.putText(depth_color, f"{label} Depth: {depth_value}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    base_distance_threshold = 370
    depth_threshold = 15
    proximity_pairs = calculate_proximity(objects, base_distance_threshold, depth_threshold)

    for obj1, obj2, distance_2d in proximity_pairs:
        center1 = (int((obj1['x1'] + obj1['x2']) / 2), int((obj1['y1'] + obj1['y2']) / 2))
        center2 = (int((obj2['x1'] + obj2['x2']) / 2), int((obj2['y1'] + obj2['y2']) / 2))

        cv2.line(frame, center1, center2, (255, 255, 0), 2)
        cv2.line(depth_color, center1, center2, (255, 255, 0), 2)

        mid_point = (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2))
        cv2.putText(frame, f"{distance_2d:.1f}px", mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(depth_color, f"{distance_2d:.1f}px", mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return frame, depth_color, len(proximity_pairs)

def main(args):
    yolo_model_1, yolo_model_2, yolo_model_3, depth_model = load_models(args)

    video = cv2.VideoCapture(args.video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    original_duration = total_frames / fps

    os.makedirs(args.outdir, exist_ok=True)
    output_path_detect = os.path.join(args.outdir, 'output_detection.mp4')
    output_path_depth = os.path.join(args.outdir, 'output_depth.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_detect = cv2.VideoWriter(output_path_detect, fourcc, fps // args.process_every_n_frames, (width, height))
    out_depth = cv2.VideoWriter(output_path_depth, fourcc, fps // args.process_every_n_frames, (width, height))

    # CSV 파일 생성
    csv_path = os.path.join(args.outdir, 'proximity_count_7tr21_machine.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'Count'])  # 헤더 작성

    frame_count = 0
    start_time = time.time()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % args.process_every_n_frames == 0:
            frame_filename = os.path.join('./frames', f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_filename, frame)

            processed_frame, depth_map, proximity_count = process_frame(frame, yolo_model_1, yolo_model_2, yolo_model_3,
                                                                        depth_model, args, frame_count)

            # CSV 파일에 결과 추가
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([frame_count, proximity_count])

            out_detect.write(processed_frame)
            out_depth.write(depth_map)

            cv2.imshow('Object Detection', processed_frame)
            cv2.imshow('Depth Map', depth_map)

            frame_filename = os.path.join('./result_frame', f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_filename, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    total_processing_time = end_time - start_time

    video.release()
    out_detect.release()
    out_depth.release()
    cv2.destroyAllWindows()

    print(f"원본 영상 길이: {original_duration:.2f} 초")
    print(f"총 처리 시간: {total_processing_time:.2f} 초")

if __name__ == '__main__':
    args = parse_args()
    main(args)