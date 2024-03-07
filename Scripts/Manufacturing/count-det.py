import supervision as sv
import numpy as np
import cv2

INPUT_VIDEO = "cans.mp4"

roboflow.login()

model = project.version(1).model

CLASSES_TO_CHECK = ["can"]

CLASS_IDX = {class_name: i for i, class_name in enumerate(CLASSES_TO_CHECK)}

CLASS_ID_TO_FILTER = [CLASS_IDX[class_name] for class_name in CLASSES_TO_CHECK]

detections_buffer = []

for i, frame in enumerate(sv.get_video_frames_generator(source_path=INPUT_VIDEO)):
    if len(detections_buffer) > 240:
        detections_buffer.pop(0)

    inference_results = model.predict(frame)

    detections = sv.Detections.from_inference(inference_results.json(), class_list=CLASSES_TO_CHECK)

    detections = detections[np.isin(detections.class_id, CLASS_ID_TO_FILTER)]

    detections_buffer.append(detections)

    box_annotator = sv.BoxAnnotator()

    labels = [
        f"{CLASSES_TO_CHECK[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]
    
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        labels=labels
    )
    
    cv2.imshow("frame", annotated_frame)
    cv2.waitKey(1)


frames = cv2.VideoCapture(0)

while True:
	_, frame = frames.read()


# Record When No Object is Visible

detections = detections[np.isin(detections.class_id, CLASS_ID_TO_FILTER)]

# if no object in last 10 frames
if len(detections) == 0 and all(len(detections) == 0 for detections in detections_buffer):
	cv2.putText(annotated_frame, f"No objects found in the last 10 frames!", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    continue

for i, frame in enumerate(sv.get_video_frames_generator(source_path=INPUT_VIDEO)):
    if len(detections_buffer) > 240:
    	detections_buffer.pop(0)
        
    inference_results = model.predict(frame)
    detections = sv.Detections.from_inference(inference_results.json(), class_list=CLASSES_TO_CHECK)
    detections = detections[np.isin(detections.class_id, CLASS_ID_TO_FILTER)]
    detections_buffer.append(detections)
    
    # if no object in last 240 frames
    if len(detections_buffer) == 240 and all([len(detections) == 0 for detections in detections_buffer]):
        print("no object in last 240 frames")
        break

# Record When Too Many Objects Are Visible

if np.mean([len(detections) for detections in detections_buffer]) > 3:
	cv2.putText(annotated_frame, f"Too many objects detected!", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

for i, frame in enumerate(sv.get_video_frames_generator(source_path=INPUT_VIDEO)):
    if len(detections_buffer) > 240:
    	detections_buffer.pop(0)
        
    inference_results = model.predict(frame)
    detections = sv.Detections.from_inference(inference_results.json(), class_list=CLASSES_TO_CHECK)
    detections = detections[np.isin(detections.class_id, CLASS_ID_TO_FILTER)]
    detections_buffer.append(detections)
    
    if np.mean([len(d.xyxy) for d in detections_buffer if len(d.xyxy) > 0]) > 3 and len(detections_buffer) == 240:
    	print("Too many objects detected!")
