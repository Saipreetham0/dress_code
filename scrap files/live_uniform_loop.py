import os
import time
import tempfile
import requests
import base64
import json

API_URL = "https://serverless.roboflow.com"
API_KEY = os.getenv("ROBFLOW_API_KEY", "5kfNvgujHDCz6aLZ6xKo")
WORKSPACE = "sridhar-sanapala-4tunp"
WORKFLOW_ID = "find-shirts-pants-shoes-inshirts-and-identification-cards"
IP_CAM = os.getenv("IP_CAM_URL", "http://10.252.44.248:8080")

REQUIRED = {
    'shirt': ['shirt', 'gray-shirt', 'shirt-gray'],
    'inshirt': ['inshirt', 'tucked', 'shirt-tucked', 'tucked-shirt'],
    'pants': ['pants', 'black-pants', 'pants-black'],
    'shoes': ['shoes', 'shoe'],
    'id-card': ['id-card', 'id_card', 'idcard']
}


def fetch_frame(ip_url: str) -> str:
    for path in ("/shot.jpg", "/photo.jpg"):
        url = ip_url.rstrip('/') + path
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and r.content:
                fd, tpath = tempfile.mkstemp(suffix=".jpg")
                with os.fdopen(fd, 'wb') as f:
                    f.write(r.content)
                return tpath
        except Exception:
            pass
    raise RuntimeError("Unable to fetch IP camera frame. Open IP Webcam and ensure /shot.jpg works in browser.")


def run_workflow(image_path: str) -> dict:
    url = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    with open(image_path, 'rb') as f:
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


def labels_from_result(res: dict):
    labels = []
    for key in ["shirt", "pants", "shoes", "id-card", "inshirt"]:
        if key in res and isinstance(res[key], list):
            for det in res[key]:
                cls = det.get("class") or det.get("label") or key
                labels.append(cls)
    preds = res.get("predictions") or res.get("output") or []
    if isinstance(preds, list):
        for det in preds:
            cls = det.get("class") or det.get("label")
            if cls:
                labels.append(cls)
    return set(l.lower() for l in labels)


def is_uniform(labels: set) -> bool:
    return (
        any(l in REQUIRED['shirt'] for l in labels)
        and any(l in REQUIRED['inshirt'] for l in labels)
        and any(l in REQUIRED['pants'] for l in labels)
        and any(l in REQUIRED['shoes'] for l in labels)
        and any(l in REQUIRED['id-card'] for l in labels)
    )


def main():
    print("Starting live uniform check loop... Ctrl+C to stop")
    while True:
        tfile = None
        try:
            tfile = fetch_frame(IP_CAM)
            res = run_workflow(tfile)
            labels = labels_from_result(res)
            print("1" if is_uniform(labels) else "0")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("0")
            print("Error:", e)
        finally:
            if tfile:
                try:
                    os.remove(tfile)
                except Exception:
                    pass
        time.sleep(1)


if __name__ == '__main__':
    main()
