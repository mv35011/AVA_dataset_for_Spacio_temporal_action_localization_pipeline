import os
import json
import torch
import argparse

from tqdm import tqdm
from yolox_config.yolox_config import get_yolox_config
from mmdet.apis import init_detector, inference_detector

def main():
    parser = argparse.ArgumentParser(description="Run YOLOX detection on frame directories.")
    parser.add_argument('--frame_dir', type=str, required=True, help="Path to extracted frames directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save detection results.")
    args = parser.parse_args()

    frame_dir = args.frame_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config = get_yolox_config()
    model = init_detector(config['config_file'], config['checkpoint_file'], device=config['device'])
    score_thresh = config.get("score_thresh", 0.25)

    video_ids = [d for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))]
    print(f"🔍 Found {len(video_ids)} video clips for detection...")

    for video_id in tqdm(video_ids, desc="Running person detection"):
        input_path = os.path.join(frame_dir, video_id)
        frame_files = sorted([f for f in os.listdir(input_path) if f.endswith('.jpg')])
        results = []

        for frame_file in frame_files:
            frame_path = os.path.join(input_path, frame_file)
            result = inference_detector(model, frame_path)

            pred = result.pred_instances
            boxes = pred.bboxes
            labels = pred.labels
            scores = pred.scores

            # Keep only person class (label 0) and filter by confidence
            keep = (labels == 0) & (scores > score_thresh)
            person_boxes = boxes[keep]

            # 🔍 Debug: Show detection stats
            print(f"\n📸 Frame: {frame_file}")
            print(f"  - Total detections: {len(labels)}")
            print(f"  - Person boxes kept: {len(person_boxes)}")

            for bbox in person_boxes.cpu().numpy():
                results.append({
                    'video_id': video_id,
                    'frame': frame_file,
                    'bbox': bbox.tolist()
                })

        out_file = os.path.join(output_dir, f"{video_id}.json")
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)

    print("✅ YOLOX person detection completed.")

if __name__ == "__main__":
    main()
