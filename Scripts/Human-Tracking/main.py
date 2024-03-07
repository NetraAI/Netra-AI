import cv2
import supervision as sv
from ultralytics import YOLO
from supervision import get_video_frames_generator

"""
for image 
model = YOLO("yolov8n.pt")
image = cv2.imread("test.mp4")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
"""

# for video

# Heat map


# model = YOLO('yolov8n.pt')
# heat_map_annotator = sv.HeatMapAnnotator()

# video_info = sv.VideoInfo.from_video_path(video_path='./Data/people-walking.mp4')
# frames_generator = get_video_frames_generator(source_path='./Data/people-walking.mp4')

# with sv.VideoSink(target_path='heatmap.mp4', video_info=video_info) as sink:
#     for frame in frames_generator:
#         result = model(frame)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         annotated_frame = heat_map_annotator.annotate(
#             scene=frame.copy(),
#             detections=detections)
#         sink.write_frame(frame=annotated_frame)


"""model = YOLO('yolov8n.pt')
blur_annotator = sv.BlurAnnotator()

video_info = sv.VideoInfo.from_video_path(video_path='./Data/test.mp4')
frames_generator = get_video_frames_generator(source_path='./Data/test.mp4')

with sv.VideoSink(target_path='blur.mp4', video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = blur_annotator.annotate(
            scene=frame.copy(),
            detections=detections)
        sink.write_frame(frame=annotated_frame)
"""
model = YOLO('yolov8n.pt')
mask_annotator = sv.BoundingBoxAnnotator()

video_info = sv.VideoInfo.from_video_path(video_path='./Data/people-walking.mp4')
frames_generator = get_video_frames_generator(source_path='./Data/people-walking.mp4')

with sv.VideoSink(target_path='mask.mp4', video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = mask_annotator.annotate(
            scene=frame.copy(),
            detections=detections)
        sink.write_frame(frame=annotated_frame)


# Tracking

# model = YOLO('yolov8n.pt')

# trace_annotator = sv.TraceAnnotator()

# video_info = sv.VideoInfo.from_video_path(video_path='./Data/test.mp4')
# frames_generator = get_video_frames_generator(source_path='./Data/test.mp4')
# tracker = sv.ByteTrack()

# with sv.VideoSink(target_path='people-trace.mp4', video_info=video_info) as sink:
#     for frame in frames_generator:
#         result = model(frame)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         detections = tracker.update_with_detections(detections)
#         annotated_frame = trace_annotator.annotate(
#             scene=frame.copy(),
#             detections=detections)
#         sink.write_frame(frame=annotated_frame)


"""
model = YOLO('yolov8n.pt')
mask_annotator = sv.BoundingBoxAnnotator()

video_info = sv.VideoInfo.from_video_path(video_path='test.mp4')
frames_generator = get_video_frames_generator(source_path='test.mp4')

with sv.VideoSink(target_path='mask-tested.mp4', video_info=video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = mask_annotator.annotate(
            scene=frame.copy(),
            detections=detections)
        sink.write_frame(frame=annotated_frame)
"""

# bounding_box_annotator = sv.BoundingBoxAnnotator()
# corner_annotator = sv.BoxCornerAnnotator()
# color_annotator = sv.ColorAnnotator()
# circle_annotator = sv.CircleAnnotator()
# dot_annotator = sv.DotAnnotator()
# triangle_annotator = sv.TriangleAnnotator()
# ellipse_annotator = sv.EllipseAnnotator()
# halo_annotator = sv.HaloAnnotator()
# mask_annotator = sv.MaskAnnotator()
# polygon_annotator = sv.PolygonAnnotator()
# label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
# blur_annotator = sv.BlurAnnotator()
# pixel_annotator = sv.PixelateAnnotator()

"""
  video only Annotators:
    Heatmap
    Trace
"""


# round_box_annotator = sv.RoundBoxAnnotator()
# percentage_annotator = sv.PercentBarAnnotator()


# label_annotator = sv.LabelAnnotator()

# labels = [
#   model.model.names[class_id]
#   for class_id
#   in detections.class_id
# ]

# annotated_image = bounding_box_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = corner_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = color_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = circle_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = dot_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = triangle_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = ellipse_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = halo_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = mask_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = polygon_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = blur_annotator.annotate(
#   scene = image, detections=detections)

# annotated_image = pixel_annotator.annotate(
#   scene = image, detections=detections)




# show Label
# annotated_image = label_annotator.annotate(
#   scene = annotated_image, detections=detections, labels=labels)

# sv.plot_image(annotated_image)