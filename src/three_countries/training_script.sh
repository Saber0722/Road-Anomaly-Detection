#!/usr/bin/env bash
set -e  # stop if any command fails

echo "🚀 Starting YOLO training pipeline"

echo "▶️ Training YOLOv8 Nano"
python yolo_nano.py

echo "▶️ Training YOLOv8 Small"
python yolo_small.py

echo "▶️ Training YOLOv8 Medium"
python yolo_medium.py

echo "✅ All trainings completed successfully"
