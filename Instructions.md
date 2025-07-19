# Custom AVA-Style Dataset Creation Pipeline

This document outlines the complete step-by-step process to convert tracked person detections into a final `train.csv` file in the AVA dataset format.

## Prerequisites

Before starting the pipeline, ensure you have the following:

### Required Dependencies
- YOLOX and DeepSORT must be run to generate detection files
- All required Python dependencies installed
- VIA (VGG Image Annotator) tool for manual annotation (if not using dummy annotations)

### Required Directory Structure
1. **Frames Directory:** A root directory (`/frames/`) containing subdirectories for each video clip:
   ```
   /frames/
   ├── 1_clip_000/    # Contains extracted .jpg frames for clip 000
   ├── 1_clip_001/    # Contains extracted .jpg frames for clip 001
   └── ...
   ```

2. **Tracking Data Directory:** A directory (`/tracking_data/`) containing tracking output as `.json` files:
   ```
   /tracking_data/
   ├── 1_clip_000.json    # Tracking data for clip 000
   ├── 1_clip_001.json    # Tracking data for clip 001
   └── ...
   ```

### Tracking Data Format
Each tracking JSON file must contain:
- `video_id`: Identifier for the video clip
- `frame`: Frame number or timestamp
- `track_id`: Unique identifier for each tracked person
- `bbox`: Absolute pixel coordinates in format `[x1, y1, x2, y2]`

---

## Core Pipeline

### Step 1: Create Dense Proposals from Tracking Data

**Purpose:** Consolidates all tracking JSON files into a single, efficient `.pkl` file for downstream processing.

**Script:** `tools/create_proposals_from_tracks_absolute.py`

**Key Logic:** 
- Iterates through each tracking JSON file
- Groups bounding boxes by `video_id` and then by `frame_name`
- Stores **raw, absolute pixel coordinates** directly to ensure accuracy
- Avoids fragile normalization steps that could introduce errors

**Command:**
```bash
python tools/create_proposals_from_tracks.py \
  --tracking_dir "/path/to/your/tracking_data/" \
  --output_path "./proposals/dense_proposals.pkl"
```

**Output:** `dense_proposals.pkl` file containing all consolidated tracking data

---

### Step 2: Generate VIA Annotation Files

**Purpose:** Creates VIA-compatible JSON annotation files for each video clip, preparing them for the annotation phase.

**Script:** `tools/proposals_to_via.py`

**Key Logic:**
- Reads the `dense_proposals.pkl` file
- Creates a VIA project for each video clip
- Treats every frame with person detections as a separate, annotatable image
- Embeds `track_id` from tracking data as a non-editable `person_id` attribute
- Saves `_via.json` files directly into corresponding frame subdirectories

**Command:**
```bash
python tools/proposals_to_via.py \
  --pickle_path "./proposals/dense_proposals.pkl" \
  --frame_dir "/path/to/your/frames/"
```

**Output:** `_via.json` files in each clip subdirectory, ready for annotation

---

### Step 3: Annotate the Data

**Purpose:** Label actions for each detected person in the bounding boxes.

#### Method A: Manual Annotation (Recommended for Production)

1. **Setup:**
   - Open a clip's frame directory in the VIA tool
   - Load the images and corresponding `_via.json` file

2. **Annotation Process:**
   - Annotate actions for each person bounding box
   - Use the embedded `person_id` for context and consistency

3. **Completion:**
   - Save the project in VIA
   - Rename the output file to `_finish.json` (e.g., `1_clip_000_finish.json`)
   - The `_finish.json` suffix signals completion to the pipeline

#### Method B: Automatic Dummy Annotation (Testing Only)

**Purpose:** Automatically generates `_finish.json` files with random action labels for pipeline testing.

**Script:** `dummy_annotator.py`

**Key Logic:**
- Finds all `_via.json` files in the frame directory
- Iterates through every bounding box
- Randomly assigns valid action labels from predefined categories
- Saves results as `_finish.json` files

**Command:**
```bash
python dummy_annotator.py --frame_dir "/path/to/your/frames/"
```

**Note:** This method is only for testing the complete pipeline without manual annotation effort.

---

### Step 4: Convert Finished Annotations to Final CSV

**Purpose:** Processes all completed annotations and generates the final `train.csv` file in standard AVA format.

**Script:** `tools/via_to_ava_csv.py`

**Key Logic:**
- Scans all frame directories for `_finish.json` files
- Extracts annotated actions, bounding boxes, and embedded `person_id`
- Calculates cumulative AVA `action_id` values (e.g., "walking" → action 13)
- Formats data into official AVA CSV structure: `video_id, frame_timestamp, bbox_coords, action_id, person_id`

**Command:**
```bash
python tools/via_to_ava_csv.py \
  --frame_dir "/path/to/your/frames/" \
  --output_csv "./annotations/train.csv" \
  --fps 25
```

**Parameters:**
- `--frame_dir`: Directory containing frames with completed `_finish.json` files
- `--output_csv`: Path for the final training CSV file
- `--fps`: Frame rate for timestamp calculations (default: 25)

**Output:** Final `train.csv` file ready for model training

---

## Utility Scripts

### Reset VIA JSON Files

**Purpose:** Clean slate restart - removes all generated annotation files to restart from Step 2.

**Script:** `tools/reset_via_files_distributed.py`

**Use Case:** When you need to regenerate annotations or fix issues in the annotation pipeline.

**Command:**
```bash
python tools/reset_via_files_distributed.py --frame_dir "/path/to/your/frames/"
```

**Warning:** This deletes all `_via.json` and `_finish.json` files. Use with caution.

---

## Complete Directory Structure

After running the full pipeline, your project structure should look like:

```
project_root/
├── frames/                    # Video frame directories
│   ├── 1_clip_000/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   ├── ...
│   │   ├── 1_clip_000_via.json      # Generated in Step 2
│   │   └── 1_clip_000_finish.json   # Completed in Step 3
│   ├── 1_clip_001/
│   │   └── ...
│   └── ...
├── tracking_data/             # Input tracking JSON files
│   ├── 1_clip_000.json
│   ├── 1_clip_001.json
│   └── ...
├── proposals/                 # Intermediate files
│   └── dense_proposals.pkl    # Generated in Step 1
├── annotations/               # Final output
│   └── train.csv             # Generated in Step 4
└── tools/                    # Processing scripts
    ├── create_proposals_from_tracks_absolute.py
    ├── proposals_to_via.py
    ├── dummy_annotator.py
    ├── via_to_ava_csv.py
    └── reset_via_files_distributed.py
```

---

## Pipeline Flow Summary

```
Tracking Data (.json) 
    ↓ [Step 1]
Dense Proposals (.pkl)
    ↓ [Step 2] 
VIA Annotation Files (_via.json)
    ↓ [Step 3]
Completed Annotations (_finish.json)
    ↓ [Step 4]
Final AVA Dataset (train.csv)
```

---

## Important Notes

- **Coordinate System:** The pipeline uses absolute pixel coordinates throughout to maintain accuracy
- **Person ID Tracking:** The `track_id` from tracking data is preserved as `person_id` in annotations
- **File Naming Convention:** Completed annotation files must end with `_finish.json` for Step 4 to process them
- **Quality Control:** For production datasets, implement manual review of annotations after Step 3
- **Disk Space:** Ensure sufficient storage for intermediate files, especially `.pkl` files which can be large
- **Path Formats:** Adjust file paths according to your operating system (Windows/Linux/macOS)