# counting-vehicules
Vehicle Counting Using Inference and Supervision
This project implements a vehicle counting system using advanced computer vision techniques. It leverages object detection, keypoint matching, and polygon zone filtering to count vehicles in video frames accurately. The project uses OpenCV, the Supervision library, and a YOLOv8 model from Roboflow, with ByteTrack for object tracking.
Features

    - Real-Time Object Detection: Detects vehicles in each video frame using a YOLOv8 model.
    - Polygon Zone Filtering: Stabilizes a defined polygon region across video frames using keypoint matching and homography transformation.
    - Object Tracking: Tracks detected vehicles across frames using ByteTrack, ensuring accurate vehicle counting.
    - Custom Annotations: Displays bounding boxes, labels, and a dynamically updating legend showing the total vehicle count and tracked object IDs.
