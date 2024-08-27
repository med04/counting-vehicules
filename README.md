# Counting Vehicles
**Vehicle Counting Using Inference and Supervision**

This project implements a robust vehicle counting system using advanced computer vision techniques. Leveraging object detection, keypoint matching, and polygon zone filtering, the system can accurately count vehicles in video frames. The project integrates OpenCV, the Supervision library, and a YOLOv8 model from Roboflow, along with ByteTrack for object tracking.

## **Introduction**

In traffic management and urban planning, accurately counting vehicles can provide critical insights for decision-making. This project addresses this need by developing a reliable vehicle counting solution that can handle various challenges, such as camera movement and perspective changes. By combining object detection with keypoint-based polygon stabilization and object tracking, this project offers a flexible and scalable approach to vehicle counting that can be applied to different scenarios, including highways, city streets, and parking lots.

### **Features**

- **Real-Time Object Detection:** Utilizes the YOLOv8 model to detect vehicles in each video frame efficiently.
- **Polygon Zone Filtering:** Stabilizes a predefined polygon region across video frames using ORB keypoint matching and homography transformation to ensure consistent vehicle detection.
- **Object Tracking:** Tracks detected vehicles across frames using ByteTrack, ensuring that vehicles are counted accurately without duplication.
- **Dynamic Annotations:** Displays bounding boxes, labels, and a dynamically updating legend showing the total vehicle count and tracked object IDs in real-time.
- **Customization:** The system is designed to be easily customizable for different use cases, such as changing the detection zone or using different models.

## **Installation**

### **Prerequisites**

- Python 3.7 or higher
- OpenCV
- Supervision Library
- YOLOv8 Model from Roboflow
- ByteTrack

## **Usage**

### **Command-Line Arguments**

- `--source_video_path`: Path to the input video file. (Required)

## **Key Components**

- **ORB Keypoint Matching:** Stabilizes the region of interest by matching keypoints between frames, ensuring that the polygon zone remains consistent.
- **Supervision Library:** Handles the drawing tasks, including bounding boxes, labels, and polygons, making the annotations clear and informative.
- **ByteTrack:** A robust multi-object tracker that maintains vehicle identities across frames, critical for accurate counting.

## **Results and Outputs**

The system processes the input video frame-by-frame and generates an annotated output where:

- A stabilized polygon zone is displayed, adapting to minor camera movements.
- Bounding boxes and labels identify and track vehicles across frames.
- A legend dynamically updates with the total count of detected vehicles and the unique IDs of tracked objects.

## **Customization and Extensibility**

This project is designed with flexibility in mind. Users can easily:

- Adjust the polygon zone for different regions of interest.
- Swap in different object detection models depending on their specific needs.
- Modify tracking parameters to suit various video qualities and conditions.

## **Potential Use Cases**

- **Traffic Monitoring:** Use the system to monitor and count vehicles on highways, streets, or intersections.
- **Parking Management:** Implement the system to count vehicles in parking lots, assisting in space allocation and monitoring.
- **Urban Planning:** Provide data-driven insights for urban development by analyzing vehicle flow in different city areas.

## **Contribution**

Contributions to this project are welcome! Whether it's improving the existing code, adding new features, or proposing alternative approaches, feel free to submit pull requests or open issues on GitHub.
