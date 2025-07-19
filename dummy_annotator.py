import os
import json
import random
import argparse
from tqdm import tqdm


def dummy_annotate_file(json_path):
    """
    Opens a _via.json file, randomly assigns action labels, and saves it as a _finish.json file.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            via_json = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"⚠️ Warning: Skipping corrupted or empty file: {json_path} ({e})")
        return False

    attributes = via_json.get('attribute', {})
    metadata = via_json.get('metadata', {})

    if not attributes or not metadata:
        return False
    action_options_map = {}
    for attr_id, attr_info in attributes.items():
        if attr_id.isdigit() and int(attr_id) <= 8:
            options = list(attr_info.get('options', {}).keys())
            if options:
                action_options_map[attr_id] = options
    for meta_key, meta_value in metadata.items():
        av_dict = meta_value.get('av', {})
        if random.random() < 0.9:
            num_actions_to_apply = random.randint(1, 3)
            categories_to_annotate = random.sample(
                list(action_options_map.keys()),
                min(num_actions_to_apply, len(action_options_map))
            )

            for attr_id in categories_to_annotate:
                if attr_id in action_options_map:
                    av_dict[attr_id] = random.choice(action_options_map[attr_id])
    output_path = json_path.replace('_via.json', '_finish.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(via_json, f)
        return True
    except IOError as e:
        print(f"❌ Error writing to file {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automatically create dummy annotations for VIA JSON files to test the pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--frame_dir', type=str, required=True,
                        help="Root directory containing the frame subdirectories with '_via.json' files.")
    args = parser.parse_args()
    files_to_process = []
    for root, _, files in os.walk(args.frame_dir):
        for file in files:
            if file.endswith("_via.json"):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("❌ Error: No '_via.json' files found to process. Please run the previous script first.")
        return

    print(f"Found {len(files_to_process)} VIA files to auto-annotate.")

    successful_count = 0
    for json_path in tqdm(files_to_process, desc="Generating dummy annotations"):
        if dummy_annotate_file(json_path):
            successful_count += 1

    print(f"\n🎉 Operation complete. Successfully generated {successful_count} '_finish.json' annotation files.")


if __name__ == "__main__":
    main()
