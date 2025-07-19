# verify_paths.py
import os
import pickle
PICKLE_PATH = r"D:\Compressed\AVA dataset\Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset\mywork\proposals\dense_proposals.pkl"
FRAME_DIR = r"D:\Compressed\AVA dataset\Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset\mywork\frames"

print("--- Path Verification Script ---")
with open(PICKLE_PATH, 'rb') as f:
    proposals_data = pickle.load(f)
video_ids_from_pickle = sorted(list(set(key.split(',')[0] for key in proposals_data.keys())))
print(f"\n✅ Found {len(video_ids_from_pickle)} unique video IDs in the pickle file.")
if not video_ids_from_pickle:
    print("Error: No video IDs found in pickle.")
    exit()
subdirectories = sorted([d for d in os.listdir(FRAME_DIR) if os.path.isdir(os.path.join(FRAME_DIR, d))])
print(f"✅ Found {len(subdirectories)} subdirectories in the frame directory.")
if not subdirectories:
    print("Error: No subdirectories found in frame_dir.")
    exit()
print("\n--- COMPARING THE FIRST 5 ENTRIES ---")
print(f"{'Pickle Video ID':<25} | {'Folder Name in /frames/':<25}")
print("-" * 55)
for i in range(min(5, len(video_ids_from_pickle))):
    pickle_id = video_ids_from_pickle[i]
    folder_name = subdirectories[i]
    match_status = "✅ MATCH" if pickle_id == folder_name else "❌ MISMATCH"
    print(f"{pickle_id:<25} | {folder_name:<25} | {match_status}")
first_id = video_ids_from_pickle[0]
expected_path = os.path.join(FRAME_DIR, first_id)
print("\n--- FINAL CHECK ---")
print(f"Checking if path for first video ID exists...")
print(f"Expected path: {expected_path}")
if os.path.isdir(expected_path):
    print("✅ SUCCESS: The path exists.")
else:
    print("❌ FAILURE: The path does not exist. This is the reason no files are being created.")