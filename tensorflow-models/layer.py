from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util


GRAPH_PB_PATH = r'D:\tensorflow\ssd_mobilenet_v1_coco_2018_01_28\frozen_inference_graph.pb' #path to your .pb file
with tf.Session() as sess:
	print("load graph")
	with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
		# for i,n in enumerate(graph_def.node):
		# 	print("Name of the node - %s" % i,n.name)

image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
scores = sess.graph.get_tensor_by_name('detection_scores:0')
classes = sess.graph.get_tensor_by_name('detection_classes:0')
num_detections = sess.graph.get_tensor_by_name('num_detections:0')

print(boxes)
print(classes)