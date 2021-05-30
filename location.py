import argparse
from ast import parse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from GPS import draw_map, geocoding
from hfnet.settings import EXPER_PATH
from notebooks.utils import draw_matches

tf.contrib.resampler

def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    return 2 * (1 - desc1 @ desc2.T)

def match_with_ratio_test(desc1, desc2, thresh):
    dist = compute_distance(desc1, desc2)
    nearest = np.argpartition(dist, 2, axis=-1)[:, :2]
    dist_nearest = np.take_along_axis(dist, nearest, axis=-1)
    valid_mask = dist_nearest[:, 0] <= (thresh**2)*dist_nearest[:, 1]
    matches = np.stack([np.where(valid_mask)[0], nearest[valid_mask][:, 0]], 1)
    return matches

class HFNet:
    def __init__(self, model_path, outputs):
        self.session = tf.Session()
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n+':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')
        
    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)

def main(query):

    white = np.zeros([1024, 20, 3], dtype=np.uint8)
    white.fill(255)

    f = open('./db.json', 'r')
    data_json = json.load(f)
    f.close()

    database = './SJTU_Landmarks/db/'
    query_dir = './SJTU_Landmarks/query/'
    image_query = query_dir + query
    image_query = cv2.imread(image_query)
    image_query = cv2.resize(image_query, (1368, 1824))
    data = sorted(os.listdir(database))

    images_db = []
    for i, image in enumerate(data):
        img = database + image
        img = cv2.imread(img)
        images_db.append(img)

    images_db = [cv2.resize(i, (1368, 1824)) for i in images_db]

    model_path = Path(EXPER_PATH, 'saved_models/hfnet')
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
    hfnet = HFNet(model_path, outputs)

    db = [hfnet.inference(i) for i in images_db]
    global_index = np.stack([d['global_descriptor'] for d in db])
    query = hfnet.inference(image_query)
    nearest = np.argmin(compute_distance(query['global_descriptor'], global_index))

    matches = match_with_ratio_test(query['local_descriptors'], db[nearest]['local_descriptors'], 0.8)

    match_img = data[nearest]

    tile = draw_matches(image_query, query['keypoints'], images_db[nearest], db[nearest]['keypoints'], matches, color=(0, 255, 0))

    loc = data_json[match_img]
    lat, lon = loc['latitude'], loc['longitude']

    draw_map([lat, lon], './', 'map.html')
    location = geocoding(lat, lon)
    print(location)

    cv2.imwrite("./results/query.png", image_query)
    cv2.imwrite("./results/matched.png", images_db[nearest])
    cv2.imwrite("./results/results.png", tile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='location')
    parser.add_argument('--query', default=r'00001.jpg', type=str)
    args = parser.parse_args()

    query = args.query

    main(query)
