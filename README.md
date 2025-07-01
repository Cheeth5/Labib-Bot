# 🤖 Labib-Bot by PlastiFind

**Labib-Bot** is an autonomous beach-cleaning robot powered by edge AI.  
It detects and collects plastic waste and identifies people littering on beaches using a real-time pose estimation model.

🏆 2nd place — International Robotics Exhibition  
🌍 Built for sustainable coastline protection

## 📦 Features

- Plastic bottle detection (YOLOv5/YOLOv11)
- Human littering detection via pose keypoints
- Audio awareness alerts
- Secure local Flask web dashboard for image reports
- Runs entirely offline on NVIDIA Jetson AGX Xavier

## 📁 Folder Structure
- `src/`: Main source code
- `models/`: Trained models and weights
- `web_dashboard/`: Flask-based local reporting interface
- `hardware/`: CAD files, electronics, and wiring diagrams
- `docs/`: Architecture diagrams, flowcharts

## 🧠 AI Details
- Fine-tuned YOLOv11-pose model for littering detection
- Trained on 10,000+ pose samples (~76% accuracy)
- Edge inference on Jetson Xavier




