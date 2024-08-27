import argparse
import cv2 as cv
import numpy as np
import supervision as sv
from inference.models.utils import get_roboflow_model
from supervision.draw.utils import Point

# ORB initialization
orb = cv.ORB_create()

# Store keypoints and descriptors from the first frame
first_keypoints = None
first_descriptors = None

# Define the polygon points
SOURCE = np.array([[522, 95], [824, 98], [1119, 473], [383, 458]], dtype=np.float32)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Vehicle Speed Estimation using Inference and Supervision'
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to video file",
        type=str,
    )
    return parser.parse_args()

def stabilize_polygon(frame, prev_frame, prev_kps, prev_des):
    # Detect keypoints and compute descriptors for the current frame
    kp, des = orb.detectAndCompute(frame, None)

    if prev_des is None or des is None:
        return SOURCE, kp, des

    # Match keypoints between the previous frame and the current frame
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(prev_des, des)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove weak matches
    NumGoodMatches = int(len(matches) * 0.1)
    matches = matches[:NumGoodMatches]

    if len(matches) < 4:
        return SOURCE, kp, des  # Not enough matches to compute homography

    # Extract matched points
    src_pts = np.float32([prev_kps[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])

    # Estimate the transformation matrix
    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    if M is None:
        return SOURCE, kp, des

    # Reshape SOURCE to the required format for perspectiveTransform
    SOURCE_reshaped = np.array([SOURCE], dtype=np.float32)

    # Transform the polygon points
    transformed_poly = cv.perspectiveTransform(SOURCE_reshaped, M)[0]

    return transformed_poly, kp, des


def create_legend(frame, legend_data):
    legend_height = 100
    legend_width = 300
    
    # Ensure the legend fits within the frame dimensions
    frame_height, frame_width = frame.shape[:2]
    if legend_width > frame_width:
        legend_width = frame_width
    if legend_height > frame_height:
        legend_height = frame_height

    # Create a copy of the frame to draw the legend on
    legend_img = frame[0:legend_height, 0:legend_width].copy()

    # Define font properties
    text_color = sv.Color.BLACK
    text_scale = 0.8
    text_thickness = 2
    text_padding = 15
    background_color = sv.Color.GREEN

    # Position for text
    y0, dy = 20, 65

    # Draw legend data onto the image
    for idx, (label, count) in enumerate(legend_data.items()):
        text = f'{label}: {count}'
        y = y0 + idx * dy
        legend_img = sv.draw_text(
            scene=legend_img,
            text=text,
            text_anchor=Point(x=text_padding*12, y=y),  # Adjust x and y as needed
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            text_font=cv.FONT_HERSHEY_SIMPLEX,
            background_color=background_color
        )

    return legend_img

def add_legend_to_frame(frame, legend_img):
    # Ensure the legend image fits within the frame
    frame_height, frame_width = frame.shape[:2]
    legend_height, legend_width = legend_img.shape[:2]

    if legend_width > frame_width:
        legend_width = frame_width
    if legend_height > frame_height:
        legend_height = frame_height

    # Create an output frame
    output_frame = frame.copy()

    # Place the legend image on the top-left corner of the frame
    output_frame[0:legend_height, 0:legend_width] = legend_img

    return output_frame


if __name__ == '__main__':
    args = parse_arguments()

    # Create a VideoInfo object by loading video metadata from the specified file path
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # Ensure resolution values are integers
    resolution = tuple(map(int, video_info.resolution_wh))

    frame_width = resolution[0]

    # Load a pretrained yolov8n-640 resolution
    model = get_roboflow_model("yolov8n-640")

    # Initialize the ByteTrack Tracker
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Calculate the optimal line thickness for annotations based on the video's resolution
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution)

    # Bounding box annotator
    bounding_box_annotator = sv.BoxAnnotator(
        thickness=thickness
    )

    # Label annotator
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale, 
        text_thickness=2
    )
    
    # Create an instance of frame generator to get frames from a video 
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    first_frame = next(frame_generator)
    first_keypoints, first_descriptors = orb.detectAndCompute(first_frame, None)

    # Polygon zone filtering feature
    polygone_zone = sv.PolygonZone(SOURCE.astype(int))
    
    # Create a named video with the ability to resize
    cv.namedWindow('Annotated frame', cv.WINDOW_NORMAL)

    # Set the window size to match the video's resolution
    cv.resizeWindow('Annotated frame', resolution[0], resolution[1])

    # Variables to count vehicles 
    total_objects_count = 0
    tracked_objects = set()  # To keep track of unique object IDs

    # Define the legend data
    legend_data = {
        'Total Count': str(total_objects_count),
        'Object IDs': 'Dynamic'
    }

    # Set the update interval (e.g., every 30 frames)
    update_interval = 30
    frame_count = 0

    for frame in frame_generator:
        frame_count += 1

        if frame_count % update_interval == 0:
            first_frame = frame.copy()
            first_keypoints, first_descriptors = orb.detectAndCompute(first_frame, None)
        
        transformed_polygon, first_keypoints, first_descriptors = stabilize_polygon(
            frame, first_frame, first_keypoints, first_descriptors
        )

        # Update the polygon zone with the stabilized polygon
        polygone_zone = sv.PolygonZone(transformed_polygon.astype(int))

        result = model.infer(frame)[0]

        # Convert the result into a supervision detections object
        detections = sv.Detections.from_inference(result)

        # Determines if the detections are within the polygon zone
        detections = detections[polygone_zone.trigger(detections)]
        
        # Update the Tracker with Detected Objects
        detections = byte_track.update_with_detections(detections=detections)

        # Count the number of detected and tracked objects
        # Count unique objects
        for tracker_id in detections.tracker_id:
            if tracker_id not in tracked_objects:
                tracked_objects.add(tracker_id)
                total_objects_count += 1

        labels = [
            f'#{tracker_id}'
            for tracker_id 
            in detections.tracker_id
        ]

        annotated_frame = frame.copy()

        # Convert transformed_polygon to integer and reshape
        transformed_polygon_int = transformed_polygon.astype(np.int32).reshape((-1, 1, 2))

        # Draw the polygon zone (debugging purposes)
        annotated_frame = sv.draw_polygon(annotated_frame, transformed_polygon_int, sv.Color.RED)

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        # Update legend data
        legend_data['Total Count'] = str(total_objects_count)
        legend_data['Object IDs'] = ', '.join(labels)
        legend_img = create_legend(annotated_frame, legend_data)

        # Add legend to the frame
        combined_frame = add_legend_to_frame(annotated_frame, legend_img)

        # Display the result (annotated frame) using imshow() method from opencv
        cv.imshow('Annotated frame', combined_frame)
        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cv.destroyAllWindows()
