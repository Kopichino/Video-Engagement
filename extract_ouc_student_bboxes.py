import os
import glob
import json
import time
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

def main():
    print("=" * 60)
    print("EXTRACTING OUC-CGE STUDENT BOUNDING BOXES FOR SPATIAL NORMALIZATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load SSDLite320 model with weights
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(device)

    base_dir = "data/extracted_frames/OUC-CGE"
    clip_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])
    total_clips = len(clip_dirs)
    print(f"Found {total_clips} OUC-CGE clip directories.")

    out_json = "data/manifests/ouc_student_bboxes.json"
    bbox_dict = {}

    # If partial exists, resume from it
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                bbox_dict = json.load(f)
            print(f"Resuming: found {len(bbox_dict)} existing precomputed bboxes.")
        except Exception:
            bbox_dict = {}

    batch_size = 16
    t0 = time.time()
    clips_to_process = [d for d in clip_dirs if os.path.basename(d) not in bbox_dict]
    print(f"Clips remaining to process: {len(clips_to_process)}")

    for i in range(0, len(clips_to_process), batch_size):
        batch_dirs = clips_to_process[i:i + batch_size]
        tensors = []
        valid_dirs = []
        img_dims = []

        for cdir in batch_dirs:
            frames = sorted(glob.glob(os.path.join(cdir, "*.png")) + glob.glob(os.path.join(cdir, "*.jpg")))
            if not frames:
                # Fallback to center crop
                bbox_dict[os.path.basename(cdir)] = [0.15, 0.1, 0.85, 0.9]
                continue
            mid_frame = frames[len(frames) // 2]
            try:
                with Image.open(mid_frame) as pil_img:
                    rgb = pil_img.convert("RGB")
                    w, h = rgb.size
                    img_dims.append((w, h))
                    tensors.append(F.to_tensor(rgb))
                    valid_dirs.append(os.path.basename(cdir))
            except Exception:
                bbox_dict[os.path.basename(cdir)] = [0.15, 0.1, 0.85, 0.9]

        if tensors:
            batch_tensors = [t.to(device) for t in tensors]
            with torch.no_grad():
                preds = model(batch_tensors)

            for cname, (w, h), pred in zip(valid_dirs, img_dims, preds):
                # Filter for person (class 1) with confidence > 0.35
                mask = (pred["labels"] == 1) & (pred["scores"] > 0.35)
                if mask.any():
                    boxes = pred["boxes"][mask].cpu().numpy()
                    scores = pred["scores"][mask].cpu().numpy()

                    # Find the most prominent / central person
                    # Calculate center distance and area
                    box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
                    box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
                    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                    # Score by combination of confidence, area, and centrality
                    center_dist = ((box_centers_x - w / 2.0) ** 2 + (box_centers_y - h / 2.0) ** 2) ** 0.5
                    rank_scores = scores * (box_areas ** 0.5) / (1.0 + center_dist / w)
                    best_idx = rank_scores.argmax()
                    best_box = boxes[best_idx]

                    # Add 15% margin around the box to include shoulders/head context
                    bw = best_box[2] - best_box[0]
                    bh = best_box[3] - best_box[1]
                    x1 = max(0.0, best_box[0] - bw * 0.15)
                    y1 = max(0.0, best_box[1] - bh * 0.10)
                    x2 = min(float(w), best_box[2] + bw * 0.15)
                    y2 = min(float(h), best_box[3] + bh * 0.15)

                    # Store normalized coordinates [0.0 - 1.0] as native Python floats
                    norm_box = [
                        round(float(x1 / w), 4),
                        round(float(y1 / h), 4),
                        round(float(x2 / w), 4),
                        round(float(y2 / h), 4),
                    ]
                    bbox_dict[cname] = norm_box
                else:
                    # Fallback to generous upper-center crop
                    bbox_dict[cname] = [0.15, 0.05, 0.85, 0.95]

        if (i // batch_size) % 25 == 0 or (i + batch_size >= len(clips_to_process)):
            elapsed = time.time() - t0
            done = len(bbox_dict)
            rate = done / (elapsed + 1e-5)
            print(f"Progress: {done}/{total_clips} clips ({done/total_clips*100:.1f}%) | Speed: {rate:.1f} clips/s | Elapsed: {elapsed:.1f}s")
            # Intermediate save
            with open(out_json, "w") as f:
                json.dump(bbox_dict, f)

    # Final save
    with open(out_json, "w") as f:
        json.dump(bbox_dict, f, indent=2)

    total_time = time.time() - t0
    print("=" * 60)
    print(f"COMPLETED BOUNDING BOX EXTRACTION FOR {len(bbox_dict)} CLIPS in {total_time:.1f}s!")
    print(f"Saved bounding box lookup to: {out_json}")
    print("=" * 60)

if __name__ == "__main__":
    main()
