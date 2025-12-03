import argparse
import os
import sys
import tempfile
import requests
import json

"""
Run Roboflow Serverless Workflow to detect:
- shirt (gray)
- pants (black)
- shoes (any)
- id-card (red/yellow/green/pink)
- inshirt (tucked)

Outputs: 1 if ALL present, else 0
"""

API_URL = "https://serverless.roboflow.com"
# Prefer environment variable; fall back to hardcoded if present
API_KEY = os.getenv("ROBFLOW_API_KEY", "5kfNvgujHDCz6aLZ6xKo")
WORKSPACE = "sridhar-sanapala-4tunp"
WORKFLOW_ID = "find-shirts-pants-shoes-inshirts-and-identification-cards"


def fetch_ip_webcam_frame(ip_url: str) -> str:
    """Fetch a single frame from IP Webcam and save to a temp file. Returns file path."""
    # Typical stream path is http://<ip>:8080/shot.jpg or /video
    candidates = [
        ip_url.rstrip('/') + "/shot.jpg",
        ip_url.rstrip('/') + "/photo.jpg",
        ip_url.rstrip('/') + "/video",
    ]
    for url in candidates:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and resp.content:
                # If it's a stream, this may not be a single image; prefer shot.jpg
                # Save bytes to temp file
                suffix = ".jpg"
                fd, path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, 'wb') as f:
                    f.write(resp.content)
                return path
        except Exception:
            continue
    raise RuntimeError(f"Unable to fetch image from IP Webcam at {ip_url}. Ensure IP Webcam app is running.")


def run_workflow_on_image(image_path: str) -> dict:
    url = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    
    # Read image and encode as base64
    with open(image_path, 'rb') as f:
        import base64
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "api_key": API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": img_b64}
        }
    }
    
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    return response.json()


def is_uniform_complete(result: dict) -> bool:
    """Apply rules: require shirt(gray), pants(black), shoes(any), id-card(color), and inshirt(tucked)."""
    # Result format depends on how the workflow is set up. We'll handle common patterns:
    # - result may contain keys per-model, e.g., {'shirt': [...], 'pants': [...], 'id-card': [...], 'inshirt': [...], 'shoes': [...]} where each is list of detections
    # - or a flat 'predictions' list with 'class' labels
    def collect_labels(res: dict):
        labels = []
        # common nested keys
        for key in ["shirt", "pants", "shoes", "id-card", "inshirt"]:
            if key in res and isinstance(res[key], list):
                for det in res[key]:
                    cls = det.get("class") or det.get("label") or key
                    labels.append(cls)
        # flat predictions
        preds = res.get("predictions") or res.get("output") or []
        if isinstance(preds, list):
            for det in preds:
                cls = det.get("class") or det.get("label")
                if cls:
                    labels.append(cls)
        return set(label.lower() for label in labels)

    labels = collect_labels(result)

    # Accept id-card variants possibly with color suffixes
    has_id = any(l.startswith("id-card") or l in {"idcard", "id_card"} for l in labels)
    has_shirt = any(l.startswith("shirt") or l == "shirt" for l in labels)
    # If workflow emits gray-shirt specifically, also accept
    has_gray_shirt = any(l in {"gray-shirt", "shirt-gray"} for l in labels)
    has_pants = any(l.startswith("pants") or l == "pants" for l in labels)
    has_black_pants = any(l in {"black-pants", "pants-black"} for l in labels)
    has_shoes = any(l.startswith("shoes") or l == "shoes" or l == "shoe" for l in labels)
    has_inshirt = any(l in {"inshirt", "tucked", "shirt-tucked", "tucked-shirt"} for l in labels)

    # Requirements: gray shirt AND tucked
    shirt_ok = (has_gray_shirt or has_shirt) and has_inshirt
    pants_ok = has_black_pants or has_pants
    shoes_ok = has_shoes
    id_ok = has_id

    return shirt_ok and pants_ok and shoes_ok and id_ok


def main():
    parser = argparse.ArgumentParser(description="Run Roboflow workflow and print 1 for complete uniform else 0")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--ip", type=str, help="IP Webcam base URL, e.g., http://10.252.44.248:8080")
    parser.add_argument("--debug", action="store_true", help="Print raw workflow JSON output")
    args = parser.parse_args()

    if not args.image and not args.ip:
        print("Usage: python run_roboflow_workflow.py --image <path> | --ip http://<ip>:8080")
        sys.exit(2)

    if args.ip:
        try:
            image_path = fetch_ip_webcam_frame(args.ip)
            cleanup_temp = True
        except Exception as e:
            print(f"0\nError fetching IP camera frame: {e}")
            sys.exit(1)
    else:
        image_path = args.image
        cleanup_temp = False
        if not os.path.exists(image_path):
            print("0\nImage not found: " + image_path)
            sys.exit(1)

    try:
        result = run_workflow_on_image(image_path)
        if args.debug:
            import json
            print(json.dumps(result, indent=2))
        uniform = is_uniform_complete(result)
        print("1" if uniform else "0")
    except Exception as e:
        print(f"0\nWorkflow error: {e}")
        sys.exit(1)
    finally:
        if cleanup_temp:
            try:
                os.remove(image_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
