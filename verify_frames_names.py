# verify_frame_names.py
import os
import pickle
PICKLE_PATH = r"D:\Compressed\AVA dataset\Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset\mywork\proposals\dense_proposals.pkl"
FRAME_DIR = r"D:\Compressed\AVA dataset\Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset\mywork\frames"
FPS = 25

print("--- Frame Filename Verification Script ---")

try:
    with open(PICKLE_PATH, 'rb') as f:
        proposals_data = pickle.load(f)

    first_video_id = sorted(list(set(key.split(',')[0] for key in proposals_data.keys())))[0]

    keys_for_first_video = [key for key in proposals_data if key.startswith(first_video_id)]
    first_second = sorted([int(key.split(',')[1]) for key in keys_for_first_video])[0]

    print(f"\nFound first video ID: '{first_video_id}'")
    print(f"Found its first keyframe at second: {first_second}")
    middle_frame_idx = (first_second * FPS) + (FPS // 2)
    expected_filename = f"{first_video_id}_frame_{str(middle_frame_idx).zfill(6)}.jpg"

    print(f"\nSCRIPT EXPECTS to find a filename like: '{expected_filename}'")
    video_folder_path = os.path.join(FRAME_DIR, first_video_id)
    if not os.path.isdir(video_folder_path):
        print(f"\n❌ ERROR: The directory '{video_folder_path}' does not exist.")
        exit()

    actual_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.jpg')])

    if not actual_files:
        print(f"\n❌ ERROR: No .jpg files found in '{video_folder_path}'.")
        exit()

    print(f"\nACTUAL filenames in the directory look like this:")
    for i in range(min(5, len(actual_files))):
        print(f"  - {actual_files[i]}")
    print("\n--- FINAL CHECK ---")
    if expected_filename in actual_files:
        print("✅ SUCCESS: The expected filename format matches the actual files!")
    else:
        print("❌ FAILURE: The expected filename format DOES NOT MATCH the actual files.")
        print("This is the reason no JSON files are being created.")

except Exception as e:
    print(f"\nAn error occurred: {e}")