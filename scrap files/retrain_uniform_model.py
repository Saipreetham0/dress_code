from ultralytics import YOLO
import argparse
import os
import torch

def parse_args():
    p = argparse.ArgumentParser(description='Retrain YOLOv8 on uniform dataset')
    p.add_argument('--data', default='Uniform_Detection.v1i.yolov8/data.yaml', help='path to data.yaml')
    p.add_argument('--weights', default='yolov8n.pt', help='initial weights (eg yolov8n.pt or path to checkpoint)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--project', default='runs/train', help='save results to project/name')
    p.add_argument('--name', default='uniform_retrain', help='run name')
    p.add_argument('--device', default=None, help='device: 0 or cpu (default auto)')
    return p.parse_args()


def main():
    args = parse_args()

    # Check device
    device = args.device
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'

    print(f"Starting training with device={device}, epochs={args.epochs}, batch={args.batch}")
    print(f"Data: {args.data}")

    # Ensure data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data YAML not found: {args.data}")

    model = YOLO(args.weights)

    # Start training
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        workers=8,
        optimizer='Adam'
    )

    print('Training finished. Check the runs directory for results (best.pt will be in the run folder).')


if __name__ == '__main__':
    main()
