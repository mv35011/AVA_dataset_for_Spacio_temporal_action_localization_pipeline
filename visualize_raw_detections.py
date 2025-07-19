import os
import json
import argparse
import cv2


def visualize_raw_boxes(json_dir, frame_dir, clip_id, frame_number):
    """
    Draws raw, absolute bounding boxes from a YOLOvX JSON file onto the corresponding frame.
    This script bypasses the .pkl file to verify the source detection data.
    """
    # 1. Construct paths
    json_filename = f"{clip_id}.json"
    json_path = os.path.join(json_dir, json_filename)

    frame_filename = f"{clip_id}_frame_{str(frame_number).zfill(4)}.jpg"
    image_path = os.path.join(frame_dir, clip_id, frame_filename)

    # 2. Load the detection JSON file
    try:
        with open(json_path, 'r') as f:
            all_detections = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Detection JSON file not found at '{json_path}'")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{json_path}'. It might be empty or corrupt.")
        return

    # 3. Load the image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image file.")
        return

    print(f"✅ Loaded image '{frame_filename}'")

    # 4. Find all detections for the specified frame
    frame_detections = [det for det in all_detections if det.get('frame') == frame_filename]

    if not frame_detections:
        print(f"⚠️ Warning: No detections found in '{json_filename}' for frame '{frame_filename}'.")
        return

    print(f"✅ Found {len(frame_detections)} detections for this frame.")

    # 5. Draw each bounding box using raw pixel values
    for i, det in enumerate(frame_detections):
        bbox = det.get('bbox')
        if not bbox or len(bbox) < 4:
            continue

        # The bbox is already in absolute pixel coordinates [x1, y1, x2, y2]
        abs_x1, abs_y1, abs_x2, abs_y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Draw rectangle on the image
        color = ((i * 60) % 255, (i * 100) % 255, (i * 140) % 255)
        cv2.rectangle(img, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 2)

        # Put a label on the box
        label = f"Raw Box {i + 1}"
        cv2.putText(img, label, (abs_x1, abs_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 6. Save the output image
    output_filename = f"DEBUG_RAW_{clip_id}_frame_{frame_number}.jpg"
    cv2.imwrite(output_filename, img)
    print(f"\n🎉 Success! Raw visualization saved to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RAW bounding box detections from a YOLOvX JSON file.")
    parser.add_argument('--json_dir', type=str, required=True,
                        help="Directory containing the original YOLOvX JSON detection files.")
    parser.add_argument('--frame_dir', type=str, required=True, help="Root directory containing frame subdirectories.")
    parser.add_argument('--clip_id', type=str, required=True,
                        help="The ID of the clip to visualize (e.g., '4_clip_001').")
    parser.add_argument('--frame_number', type=int, default=0,
                        help="The specific frame number to visualize (e.g., 0 for '..._frame_0000.jpg').")
    args = parser.parse_args()

    visualize_raw_boxes(args.json_dir, args.frame_dir, args.clip_id, args.frame_number)


if __name__ == "__main__":
    main()
