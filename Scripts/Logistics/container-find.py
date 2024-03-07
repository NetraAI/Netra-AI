from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yard-management-system/1",
    video_reference=0,
    on_prediction=render_boxes,
)
pipeline.start()
pipeline.join()

# To run a script
# python3 app.py --video=video.mp4 --model-id=model/1 --output=output.mp4