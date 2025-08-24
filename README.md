# Helmet and Traffic Violation Detection System
📋 Overview
Motorcycle accidents have been rapidly growing throughout the years in many countries. Helmets are the primary safety equipment for motorcyclists, yet many riders neglect to wear them. This project proposes an automated system that monitors traffic in real-time, detects riders without helmets, identifies traffic signal violations, and provides comprehensive analytics for traffic safety enforcement.

Our system utilizes advanced computer vision and deep learning techniques to detect vehicles, identify riders without helmets, monitor traffic light compliance, and track violations across video streams. The application provides a user-friendly interface for traffic management authorities to analyze video footage and generate detailed violation reports.

🚀 Key Features
Real-time Helmet Detection: Identifies riders without helmets using a custom-trained YOLO model

Traffic Light Violation Detection: Monitors vehicles running red lights using color detection and object tracking

Multi-Object Tracking: Tracks vehicles across frames to accurately identify violations

Comprehensive Analytics: Provides detailed statistics and visualizations of detected violations

User-Friendly Interface: Streamlit-based web application for easy interaction and video processing

Export Functionality: Download processed videos with violation annotations and detailed reports

⚙️ Technical Implementation
This project implements a sophisticated pipeline that combines multiple computer vision techniques:

Object Detection: Uses YOLOv8 for detecting vehicles, persons, and traffic lights

Custom Helmet Detection: Employs a specially trained YOLO model for helmet detection

Color Recognition: Implements HSV color space analysis for traffic light status detection

Object Tracking: Utilizes a dedicated tracker to follow vehicles across frames

Violation Logic: Applies spatial and temporal reasoning to identify violations

Web Interface: Built with Streamlit for accessible interaction and visualization

📥 Installation
Prerequisites
Python 3.7+

pip package manager

Step-by-Step Installation
Clone the Repository

bash
git clone <repository-url>
cd Helmet-and-Traffic-Violation-Detection
Create a Virtual Environment (Recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

bash
pip install -r requirements.txt
If you don't have a requirements.txt file, install the following packages:

bash
pip install streamlit opencv-python numpy ultralytics cvzone scikit-learn pillow
Download Model Weights

Download YOLOv8n weights (will be automatically downloaded by ultralytics)

Place your custom helmet detection model (best_helmet.pt) in the appropriate directory

Ensure the model path in the code matches your file location

💻 Usage
Running the Application
Start the Streamlit Application

bash
streamlit run app.py
Access the Web Interface

Open your web browser and navigate to the local URL provided (typically http://localhost:8501)

Process a Video

Use the file uploader to select a traffic video (MP4, AVI, or MOV format)

Configure detection settings using the sidebar controls

Click "Start Processing" to begin analysis

View real-time detection results and violation metrics

Download the processed video with annotations after completion

Configuration Options
Detection Confidence: Adjust the sensitivity of object detection (0.2-1.0)

Show Bounding Boxes: Toggle visibility of detection bounding boxes

Enable Helmet Detection: Activate/deactivate helmet violation detection

Enable Traffic Light Detection: Activate/deactivate traffic light violation detection

🏗️ System Architecture
The application implements a multi-stage processing pipeline:

Video Input: Accepts uploaded video files

Frame Extraction: Processes video frame by frame

Object Detection:

YOLOv8 for vehicles, persons, and traffic lights

Custom helmet detection model for helmet classification

Traffic Light Analysis:

ROI extraction from detected traffic lights

HSV color space conversion and masking

Status determination (Red/Green)

Violation Detection:

Helmet violations: Persons associated with motorcycles without helmets

Traffic violations: Vehicles crossing stop line during red light

Tracking & Analytics:

Vehicle tracking across frames

Violation counting and logging

Output Generation:

Annotated video with bounding boxes and labels

Comprehensive violation report

🔧 Model Details
Object Detection Models
YOLOv8n: Pre-trained on COCO dataset for general object detection

Custom Helmet Detection: Fine-tuned YOLO model specifically for helmet detection

Tracking Algorithm
Implements a custom tracker that maintains vehicle identities across frames

Uses bounding box centroids and motion estimation for consistent tracking

Color Detection
Converts traffic light ROI to HSV color space

Applies thresholding for red and green color ranges

Determines traffic light status based on pixel counts

⚡ Performance Considerations
Processing speed depends on hardware capabilities

For real-time performance, consider using GPU acceleration

Adjust confidence thresholds to balance between precision and recall

Larger videos will require more processing time and memory

📊 Outputs
The system generates several outputs:

Processed Video: Original video with annotated violations

Violation Metrics: Real-time counters for helmet and traffic violations

Frame-by-Frame Alerts: Timestamped notifications for each violation detected

Summary Report: Comprehensive statistics after video processing completes

🔄 Customization
The system can be extended in several ways:

Adding new violation types (e.g., wrong-way driving, speeding estimation)

Integrating with license plate recognition systems

Connecting to databases for violation logging and management

Adding support for multiple camera inputs

Implementing real-time alerts and notifications

❗ Troubleshooting
Common Issues
Model not found errors

Check the path to your custom helmet detection model

Ensure the model file is in the correct format (.pt for YOLO models)

Performance issues

Reduce video resolution for faster processing

Adjust confidence thresholds to reduce detection computations

Memory errors

Process shorter video segments

Increase system RAM or use cloud processing

Getting Help
For issues with the application, please check:

The Streamlit documentation for interface-related problems

OpenCV and Ultralytics documentation for computer vision components

GitHub issues for known problems and solutions

📄 License
This project is intended for research and educational purposes. Please check the license terms of the dependent libraries (OpenCV, Ultralytics YOLO, Streamlit) for commercial use.

🙏 Acknowledgments
Thanks to the Ultralytics team for the YOLOv8 implementation

OpenCV community for computer vision utilities

Streamlit team for the web application framework

🔮 Future Enhancements
Planned improvements for the system include:

Integration with automatic license plate recognition

Speed estimation using tracking data

Support for multiple traffic lanes and directions

Cloud deployment for scalable processing

Historical data analysis and trend reporting

