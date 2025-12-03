import os
import json
import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np

DATASET_BASE = os.path.join('dataset', 'Sitam_Uniform_Detection1.v2-dataset.yolov8')
DATA_YAML = os.path.join(DATASET_BASE, 'data.yaml')
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), 'uniform_colors.json')


def read_class_names() -> List[str]:
    """Read class names from data.yaml. Falls back to default order if missing."""
    if not os.path.exists(DATA_YAML):
        # Fallback known classes
        return ['id card', 'pant', 'shirt', 'shoes']
    names: List[str] = []
    with open(DATA_YAML, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if l.startswith('names:'):
                # Expect a list like names: ['id card','pant','shirt','shoes']
                # Simple parse: extract between [ and ] then split by comma
                if '[' in l and ']' in l:
                    inner = l[l.find('[') + 1:l.find(']')]
                    parts = [p.strip().strip("'\"") for p in inner.split(',') if p.strip()]
                    names = parts
                break
    return names or ['id card', 'pant', 'shirt', 'shoes']


def norm_key(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


def collect_hsv_samples() -> Dict[str, List[Tuple[int, int, int]]]:
    """Iterate over train/valid/test, read labels and images, collect HSV medians per class."""
    splits = ['train', 'valid', 'test']
    class_names = read_class_names()
    samples: Dict[str, List[Tuple[int, int, int]]] = {norm_key(n): [] for n in class_names}

    for split in splits:
        img_dir = os.path.join(DATASET_BASE, split, 'images')
        lbl_dir = os.path.join(DATASET_BASE, split, 'labels')
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        for lbl_path in glob.glob(os.path.join(lbl_dir, '*.txt')):
            base = os.path.splitext(os.path.basename(lbl_path))[0]
            # Try .jpg first, then .png
            img_path_jpg = os.path.join(img_dir, base + '.jpg')
            img_path_png = os.path.join(img_dir, base + '.png')
            img_path = img_path_jpg if os.path.exists(img_path_jpg) else (img_path_png if os.path.exists(img_path_png) else None)
            if img_path is None:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            try:
                with open(lbl_path, 'r', encoding='utf-8') as lf:
                    lines = [ln.strip() for ln in lf if ln.strip()]
            except Exception:
                continue

            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    # Expect: cls xc yc w h (normalized)
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    xc = float(parts[1]); yc = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
                except Exception:
                    continue
                if cls_id < 0 or cls_id >= len(class_names):
                    continue

                # Convert normalized to pixel coords
                box_w = int(bw * w_img)
                box_h = int(bh * h_img)
                x_center = int(xc * w_img)
                y_center = int(yc * h_img)
                x1 = max(0, x_center - box_w // 2)
                y1 = max(0, y_center - box_h // 2)
                x2 = min(w_img, x_center + box_w // 2)
                y2 = min(h_img, y_center + box_h // 2)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Avoid borders: take inner region to reduce background leakage
                shrink_x = max(0, int(0.1 * roi.shape[1]))
                shrink_y = max(0, int(0.1 * roi.shape[0]))
                roi_inner = roi[shrink_y:roi.shape[0]-shrink_y, shrink_x:roi.shape[1]-shrink_x]
                if roi_inner.size == 0:
                    roi_inner = roi

                hsv = cv2.cvtColor(roi_inner, cv2.COLOR_BGR2HSV)
                h_med = int(np.median(hsv[:, :, 0]))
                s_med = int(np.median(hsv[:, :, 1]))
                v_med = int(np.median(hsv[:, :, 2]))

                key = norm_key(class_names[cls_id])
                samples[key].append((h_med, s_med, v_med))

    return samples


def compute_profile(samples: Dict[str, List[Tuple[int, int, int]]]) -> Dict[str, Dict[str, int]]:
    profile: Dict[str, Dict[str, int]] = {}
    for cls, vals in samples.items():
        if not vals:
            # Skip classes without samples
            continue
        hs = np.array([v[0] for v in vals], dtype=np.int32)
        ss = np.array([v[1] for v in vals], dtype=np.int32)
        vs = np.array([v[2] for v in vals], dtype=np.int32)
        h_med = int(np.median(hs))
        s_med = int(np.median(ss))
        v_med = int(np.median(vs))

        # Tolerances: derived from spread, with sensible minimums/maximums
        def tol(arr: np.ndarray, base: int, min_tol: int, max_tol: int) -> int:
            iqr = int(np.percentile(arr, 75) - np.percentile(arr, 25))
            t = max(min_tol, min(max_tol, iqr if iqr > 0 else base))
            return t

        tol_h = tol(hs, base=10, min_tol=8, max_tol=20)
        tol_s = tol(ss, base=50, min_tol=30, max_tol=80)
        tol_v = tol(vs, base=50, min_tol=30, max_tol=80)

        profile[cls] = {
            'h': h_med,
            's': s_med,
            'v': v_med,
            'tol_h': tol_h,
            'tol_s': tol_s,
            'tol_v': tol_v
        }
    return profile


def main():
    if not os.path.isdir(DATASET_BASE):
        print('Dataset not found at', DATASET_BASE)
        return

    print('Reading dataset classes...')
    names = read_class_names()
    print('Classes:', names)

    print('Collecting HSV samples from labels...')
    samples = collect_hsv_samples()
    counts = {k: len(v) for k, v in samples.items()}
    print('Sample counts per class:', counts)

    print('Computing color profile...')
    profile = compute_profile(samples)
    if not profile:
        print('No color samples found. Aborting.')
        return

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)

    print('Saved uniform color profile to', OUTPUT_JSON)


if __name__ == '__main__':
    main()
