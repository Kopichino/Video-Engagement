# Technical Report: COLER Dataset Structural Inspection & Standardization Strategy

**Date**: July 31, 2026  
**Target Directory**: `E:\Video_Engagement\Ver4-20260630T020728Z-3-005\Ver4`  
**Purpose**: Detailed structural audit of the COLER dataset, comparative analysis against OUC-CGE, and proposed standardization framework for cross-dataset experimentation.

---

## 1. Executive Summary

A deep-level inspection of the extracted COLER dataset (`Ver4`) was conducted without making code changes or running training procedures. 

Key quantitative findings:
- **Total Files**: 18,546 files
  - **PNG Images**: 14,978 files (RGB cropped student frames and emotion crops)
  - **JPG Images**: 3,117 files (Global classroom context view frames)
  - **CSV Files**: 451 files (`skeletons.csv` containing pose joint coordinates)
  - **Zero-byte Files**: 0
- **Total Directories**: 1,234 directories across 5 camera angles (`Goc1` to `Goc5`)
- **Empty Directories**: 14 directories (resulting from missing emotion or context mask samples)
- **Unique Video IDs**: 746 unique `Video{ID}` sequences

---

## 2. COLER Folder Hierarchy & Organization

The directory structure is organized hierarchically by camera angle (`Goc`), seating table layout (`Table`), modality/engagement class, student seat index (`X`), sample video ID (`Video`), and student tracking ID (`id_N`).

### 2.1 Top-Level Structure
```
Ver4/
├── Goc1/
│   ├── Table1/
│   ├── Table2/
│   └── Table3/
├── Goc2/
│   ├── Table1/
│   ├── Table2/
│   └── Table3/
├── Goc3/
│   ├── Table1/
│   └── Table2/
├── Goc4/
│   ├── Table1/
│   └── Table2/
└── Goc5/
    └── Table3/
```

### 2.2 Subdirectory Inventory Across Camera Angles & Tables

Not all camera angles contain all three table layouts, nor do all table folders contain all six subdirectories (`Context`, `Context_Mask`, `Disengagement`, `Engagement`, `Normal`, `Very engagement`):

| Camera Angle | Table | Available Subdirectories |
|---|---|---|
| **Goc1** | `Table1` | `Context`, `Context_Mask`, `Disengagement`, `Engagement`, `Normal`, `Very engagement` |
| **Goc1** | `Table2` | `Engagement`, `Normal`, `Very engagement` *(Missing Context & Disengagement)* |
| **Goc1** | `Table3` | `Context_Mask`, `Disengagement`, `Engagement`, `Normal`, **`Very Engagement`** *(Capitalized 'E')* |
| **Goc2** | `Table1` | `Context`, `Context_Mask`, `Disengagement`, `Engagement`, `Normal`, `Very engagement` |
| **Goc2** | `Table2` | `Very engagement` *(Only contains 1 class)* |
| **Goc2** | `Table3` | `Engagement`, `Normal`, `Very engagement` |
| **Goc3** | `Table1` | `Context`, `Context_Mask`, `Disengagement`, `Engagement`, `Normal`, `Very engagement` |
| **Goc3** | `Table2` | `Engagement`, `Normal`, `Very engagement` |
| **Goc4** | `Table1` | `Context`, `Context_Mask`, `Disengagement`, `Engagement`, `Normal`, `Very engagement` |
| **Goc4** | `Table2` | `Engagement`, `Normal`, `Very engagement` |
| **Goc5** | `Table3` | `Disengagement`, `Engagement`, `Normal`, `Very engagement` |

---

## 3. Modalities, File Naming Conventions, and Annotations

### 3.1 RGB Student Crops (`.png`)
- **Location**: Found inside class subdirectories (`Disengagement`, `Engagement`, `Normal`, `Very engagement`) within seat position folders (`X0_GocN`, `X1_GocN`, `X2_GocN`, `X3_GocN`).
- **Naming Pattern**: `Video{ID}_rgb_{student_id}_frame{frame_num}.png`  
  *Example*: `Video1113_rgb_3_frame107.png`
- **Characteristics**: Pre-cropped regional bounding boxes of individual students.
- **Sampling Variations**: Some sample folders contain full 15-frame sequences (e.g., frames 1, 9, 17, 25... 139), while others contain only 1 single sampled frame (e.g., `Video1110_rgb_3_frame131.png`).

### 3.2 Emotion Facial Crops (`.png`)
- **Location**: Found inside `Emotion/Video{ID}/id_{student_id}/` under each class subdirectory.
- **Naming Pattern**: `frame_{num}.png`  
  *Example*: `frame_005.png`
- **Characteristics**: Tight bounding box crops of facial expressions.

### 3.3 Context Global Views (`.jpg`)
- **Location**: Found inside `Context/X{N}_Goc{M}/Video{ID}/id_{student_id}/`.
- **Naming Pattern**: `id_{student_id}_frame{frame_num}.jpg`  
  *Example*: `id_3_frame71.jpg`
- **Characteristics**: Wide-angle classroom scene views. Typically contains 3 keyframes per video clip (e.g., frame 1, frame 71, frame 141).

### 3.4 Skeleton Keypoint Files (`skeletons.csv`)
- **Location**: Found inside `Context_Mask/X{N}_Goc{M}/Video{ID}/id_{student_id}/` or directly inside class folders (`Disengagement`, `Engagement`, etc.).
- **Format**: CSV with 26 keypoint columns:  
  `filename, kp_0, kp_1, kp_2, ..., kp_25` representing $(x, y)$ coordinates for 13 body joints.
- **Structure Sample**:
  ```csv
  filename,kp_0,kp_1,...,kp_25
  Video1_rgb_2_frame122.png,0.9148,0.1452,...,0.9462
  ```

---

## 4. Inconsistencies, Missing Files, & Anomaly Audit

### 4.1 Structural & Naming Inconsistencies
1. **Case Sensitivity Discrepancy**:
   - `Goc1\Table3\Very Engagement` uses a capital `'E'`.
   - All other 10 locations use lowercase `'e'` (`Very engagement`).
2. **Student Location Mapping Pattern**:
   - `X0_GocN` folders correspond to `Disengagement` (and Context).
   - `X1_GocN` folders correspond to `Normal` (and Context).
   - `X2_GocN` folders correspond to `Engagement` (and Context).
   - `X3_GocN` folders correspond to `Very engagement` (and Context).
3. **Duplicate Artifact Folder**:
   - `Goc1\Table3\Engagement\Emotion\Video1\id_2(1)` exists alongside `id_2`. Both are empty.

### 4.2 Empty Directories (14 Identified)
The following 14 leaf directories contain zero files:
1. `Goc1\Table3\Disengagement\Emotion\Video36\id_4`
2. `Goc1\Table3\Disengagement\Emotion\Video38\id_4`
3. `Goc1\Table3\Engagement\Emotion\Video1\id_2`
4. `Goc1\Table3\Engagement\Emotion\Video1\id_2(1)`
5. `Goc1\Table3\Engagement\Emotion\Video2\id_2`
6. `Goc2\Table1\Context\X1_Goc2\Video12\id_4`
7. `Goc3\Table1\Context_Mask\X1_Goc3\Video104\id_2`
8. `Goc3\Table1\Context_Mask\X1_Goc3\Video20\id_2`
9. `Goc3\Table1\Context_Mask\X1_Goc3\Video59\id_2`
10. `Goc3\Table1\Context_Mask\X1_Goc3\Video74\id_1`
11. `Goc3\Table1\Context_Mask\X2_Goc3\Video104\id_1`
12. `Goc3\Table1\Context_Mask\X2_Goc3\Video105\id_1`
13. `Goc3\Table1\Context_Mask\X2_Goc3\Video105\id_2`
14. `Goc3\Table1\Context_Mask\X2_Goc3\Video32\id_1`

### 4.3 Skeleton & Frame Sequence Irregularities
- **Duplicate CSV Rows**: Certain `skeletons.csv` files repeat keypoints for identical frame names multiple times (e.g., `id_3_frame71.jpg` duplicated 13 lines consecutively).
- **Incomplete Frame Sequences**: Frame counts per video range from 1 single PNG frame up to 15 PNG frames, requiring dynamic padding/sampling during batch loading.
- **Missing Modalities for Certain Videos**: Some videos have RGB frames but lack `skeletons.csv` or `Emotion` crops.

---

## 5. Comparative Matrix: COLER vs. OUC-CGE

| Structural Attribute | OUC-CGE Dataset | COLER Dataset |
|---|---|---|
| **Data Format** | Raw `.mp4` video clips (10s, 30fps) | Multi-modal (`.png` frames, `.jpg` context, `.csv` skeletons) |
| **Observation Granularity** | Group-level (entire classroom in frame) | Individual student level (cropped target regions) |
| **Class Taxonomy** | 3 Classes (`High`, `Mid`, `Low`) | 4 Classes (`Disengaged`, `Normal`, `Engaged`, `Very Engaged`) |
| **Directory Organization** | Flat root folders (`high/`, `mid/`, `low/`) | 5-level nested hierarchy (`Goc` → `Table` → Class → `X` → `Video` → `id`) |
| **Metadata File** | No official CSV (Custom `ouc_cge_5fold_splits.csv`) | No official CSV (Label inferred from folder path) |
| **Sample Count** | 7,439 clips (7,297 clean after removing 142 corrupted) | 3,924 annotated student clips / 746 unique video sessions |
| **Temporal Span** | Uniform 300 frames per video | Non-uniform (1 to 15 frames per clip) |

---

## 6. Standardization Framework for Cross-Dataset Experiments

To enable cross-dataset transfer learning, domain adaptation, and joint multi-task training between OUC-CGE and COLER without modifying raw files, a standardized dataset abstraction layer must be established.

```
+-----------------------------------------------------------------------------------+
|                            UNIFIED METADATA MANIFEST                              |
|                               (unified_dataset.csv)                               |
+-----------------------------------------------------------------------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
+-----------------------------+                         +-----------------------------+
|    OUC-CGE STANDARDIZER     |                         |      COLER STANDARDIZER     |
+-----------------------------+                         +-----------------------------+
| • Standardize paths         |                         | • Handle folder anomalies   |
| • Sample keyframes          |                         | • Frame sequence padding    |
| • Map labels (High/Mid/Low) |                         | • Collapse 4 classes -> 3   |
+-----------------------------+                         +-----------------------------+
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             UNIFIED PYTORCH DATALOADER                            |
|             Inputs: Tensor [Batch, Frames, Channels, Height, Width]               |
|             Target Labels: {0: Low, 1: Mid, 2: High}                              |
+-----------------------------------------------------------------------------------+
```

### 6.1 Unified Label Taxonomy Standard (3-Class Mapping)

To harmonize OUC-CGE (3 classes) and COLER (4 classes), map COLER to a unified 3-class ordinal scale:

| Unified Label Index | Unified Class Name | OUC-CGE Source Label | COLER Source Label | Mapping Rationale |
|---|---|---|---|---|
| `0` | **Low** | `Low` | `Disengaged` | Unengaged / off-task behavior |
| `1` | **Mid** | `Mid` | `Normal` | Passive attendance / neutral engagement |
| `2` | **High** | `High` | `Engaged` + `Very Engaged` | Active positive participation and interaction |

*Alternative Binary Mapping Option*:  
- `0 (Not Engaged)`: OUC-CGE (`Low`, `Mid`) | COLER (`Disengaged`, `Normal`)  
- `1 (Engaged)`: OUC-CGE (`High`) | COLER (`Engaged`, `Very Engaged`)

### 6.2 Modality Alignment Strategy

1. **Frame-Level Alignment (Recommended First Step)**:
   - Extract keyframes (e.g. 1 frame at midpoint or 3 frames at start/mid/end) from OUC-CGE `.mp4` clips.
   - Pair directly with COLER RGB frame crops (`.png`).
   - Enables standard 2D Convolutional (ResNet-50) or Vision Transformer (ViT-B/16) backbones to process both datasets identically.

2. **Video Clip Sequence Alignment**:
   - Uniformly sample $T=8$ or $T=16$ frames from OUC-CGE videos.
   - For COLER samples with $<T$ frames, apply temporal border padding or nearest-neighbor frame repetition.
   - Synthesize clip tensors `[C, T, H, W]` suitable for 3D CNNs (SlowFast / X3D).

### 6.3 Standardized Manifest Specification (`unified_dataset.csv`)

Create a unified CSV manifest that abstracts dataset-specific folder paths into a flat schema:

```csv
dataset_name,sample_id,relative_path,num_frames,raw_label,unified_label_3class,unified_label_binary,split_fold
OUC-CGE,video_0001,high/high-2636(2001-2190)/view2037.mp4,300,High,2,1,0
COLER,coler_Goc1_T1_V1196,Goc1/Table1/Engagement/X2_Goc1/Video1196/id_3,1,Engagement,2,1,0
```

---

*Report prepared for Cross-Dataset Video Engagement Analysis Research Pipeline.*
