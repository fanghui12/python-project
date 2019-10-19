# 一定要保存为UTF8的格式哦
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import time

from tensorflow.models.research.object_detection.utils import label_map_util
from tensorflow.models.research.object_detection.utils import visualization_utils as vis_util

##################### Download Model，如果本地已下载也可修改成本地路径
# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download model if not already downloaded
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model... (This may take over 5 minutes)')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    print('Extracting...')
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
else:
    print('Model already downloaded.')

##################### Load a (frozen) Tensorflow model into memory.
print('Loading model...')
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

##################### Loading label map
print('Loading label map...')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


##################### Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


##################### Detection
# 测试图片的路径，可以根据自己的实际情况修改
TEST_IMAGE_PATH = 'test_images/image1.jpg'

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

cap = cv2.VideoCapture('test_video/ceshi.mp4')

print('Detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        print(TEST_IMAGE_PATH)
        #image = Image.open(TEST_IMAGE_PATH)
        # image_np = load_image_into_numpy_array(image)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while(1):

            ret, image_np = cap.read()
            height, width = image_np.shape[0:2]
            reSize2 = cv2.resize(image_np, (int(width / 3), int(height / 3)))
            reSize1 = cv2.resize(image_np, (int(width/ 2.5), int(height / 2.5)))
            print(reSize1.shape[0:2])
            image_np_expanded = np.expand_dims(image_np, axis=0)
            print(image_np_expanded.shape)
            # Actual detection.
            start_time = time.time()
            (oBoxes, oScores, oClasses, oNum_detections) = sess.run(
                 [boxes, scores, classes, num_detections],
                 feed_dict={image_tensor: image_np_expanded})
            moddle_time = time.time()
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(oBoxes),
                np.squeeze(oClasses).astype(np.int32),
                np.squeeze(oScores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)
            end_time = time.time()
            cv2.putText(image_np, "FPS: {:.2f}".format(1 / (moddle_time - start_time)), (10, 50), 6, 1, (0, 255, 0), 1)
            cv2.putText(image_np, "drow FPS: {:.2f}".format(1 / (end_time - moddle_time)), (10, 100), 6, 1, (0, 255, 0), 1)
            cv2.imshow("image",image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()