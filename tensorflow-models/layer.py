from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util


GRAPH_PB_PATH = r'D:\tensorflow\ssd_mobilenet_v1_coco_2018_01_28\frozen_inference_graph.pb' #path to your .pb file

#获取tensor张量
#reader = tf.train.NewCheckpointReader(r'D:\tensorflow\faster_rcnn_inception_v2_coco_2018_01_28\model.ckpt')
reader = tf.train.NewCheckpointReader(r'D:\tensorflow\mask_rcnn_inception_v2_coco_2018_01_28\model.ckpt')

all_variables = reader.get_variable_to_shape_map()
print(all_variables)
for k in all_variables.keys():
	print('key = {}'.format(k))
for v in all_variables.values():
    print('values = {}'.format(v))

#w0 = reader.get_tensor("FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean")


with tf.Session() as sess:
	print("load graph")
	with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
		for i,n in enumerate(graph_def.node):
			print("Name of the node - %s" % i,n.name)

image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
scores = sess.graph.get_tensor_by_name('detection_scores:0')
classes = sess.graph.get_tensor_by_name('detection_classes:0')
num_detections = sess.graph.get_tensor_by_name('num_detections:0')
Conv2D = sess.graph.get_tensor_by_name('BoxPredictor_0/ClassPredictor/Conv2D:0')

print(Conv2D)
print(image_tensor)
print(boxes)
print(scores)
print(classes)
print(num_detections)