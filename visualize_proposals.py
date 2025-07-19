import os
import pickle
import argparse
import cv2


def visualize_boxes(pickle_path, frame_dir, clip_id, second, fps):
    """
    Draws bounding boxes from a dense proposals pickle file onto the corresponding keyframe.
    """
    # 1. Load the pickle file
    try:
        with open(pickle_path, 'rb') as f:
            proposals_data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Pickle file not found at {pickle_path}")
        return

    # 2. Find the specific proposal data
    proposal_key = f"{clip_id},{str(second).zfill(4)}"
    if proposal_key not in proposals_data:
        print(f"❌ Error: No proposals found in pickle for key '{proposal_key}'")
        return

    detections = proposals_data[proposal_key]
    print(f"✅ Found {len(detections)} detections for '{proposal_key}'.")

    # 3. Find the corresponding image file
    keyframe_idx = second * fps
    frame_filename = f"{clip_id}_frame_{str(keyframe_idx).zfill(4)}.jpg"
    image_path = os.path.join(frame_dir, clip_id, frame_filename)

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        return

    # 4. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image file.")
        return

    img_H, img_W = img.shape[:2]
    print(f"Image dimensions: {img_W}x{img_H}")

    # 5. Draw each bounding box
    for i, bbox_data in enumerate(detections):
        x1_norm, y1_norm, x2_norm, y2_norm = bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]

        # De-normalize coordinates to absolute pixel values
        abs_x1 = int(x1_norm * img_W)
        abs_y1 = int(y1_norm * img_H)
        abs_x2 = int(x2_norm * img_W)
        abs_y2 = int(y2_norm * img_H)

        # Draw rectangle on the image
        # Using a distinct color for each box
        color = ((i * 50) % 255, (i * 90) % 255, (i * 130) % 255)
        cv2.rectangle(img, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 2)

        # Put the score on the box
        score = bbox_data[4] if len(bbox_data) > 4 else 'N/A'
        label = f"Box {i + 1}: {score:.2f}"
        cv2.putText(img, label, (abs_x1, abs_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 6. Save the output image
    output_filename = f"DEBUG_{clip_id}_second_{second}.jpg"
    cv2.imwrite(output_filename, img)
    print(f"\n🎉 Success! Visualization saved to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Visualize bounding box proposals on a keyframe.")
    parser.add_argument('--pickle_path', type=str, required=True, help="Path to the dense_proposals.pkl file.")
    parser.add_argument('--frame_dir', type=str, required=True, help="Root directory containing frame subdirectories.")
    parser.add_argument('--clip_id', type=str, required=True,
                        help="The ID of the clip to visualize (e.g., '4_clip_001').")
    parser.add_argument('--second', type=int, default=0, help="The specific second within the clip to visualize.")
    parser.add_argument('--fps', type=int, default=25, help="Frame rate of the source videos.")
    args = parser.parse_args()

    visualize_boxes(args.pickle_path, args.frame_dir, args.clip_id, args.second, args.fps)


if __name__ == "__main__":
    main()
