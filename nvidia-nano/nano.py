from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph
import tensorflow.contrib.tensorrt as trt

config_path, checkpoint_path = download_detection_model('ssd_mobilenet_v1_coco')
frozen_graph, input_names, output_names = build_detection_graph(
            config=config_path,
            checkpoint=checkpoint_path,
            score_threshold=0.3,
            batch_size=1
             )
trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 25,
            precision_mode='FP16',
            minimum_segment_size=50
            )

with open('./model/trt_graph.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())
    f.close()
