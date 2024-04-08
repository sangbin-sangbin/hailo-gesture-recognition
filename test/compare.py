import json
import json
import os
import subprocess
import sys
import time
from multiprocessing import Process

import cv2
import torch
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 

import utils
from cv2_utils import CV2Utils
from models.model import Model
from openvino_utils.hand_tracker import HandTracker
from openvino_utils.hand_tracker_hailo import HandTrackerHailo
from openvino_utils import mediapipe_utils as mpu

import numpy as np
import onnxruntime
import tensorflow as tf
import numpy as np
import cv2


with open("./data.json", "r") as f:
	data = json.load(f)["data"]

hand_tracker_hailo = HandTrackerHailo(
	pd_score_thresh=0.6,
	pd_nms_thresh=0.3,
	lm_score_threshold=0.6,
)

pd_scores = "Identity_1"
pd_bboxes = "Identity"
pd_score_thresh=0.6
pd_nms_thresh=0.3
lm_score_threshold=0.6
anchor_options = mpu.SSDAnchorOptions(
	num_layers=4,
	min_scale=0.1484375,
	max_scale=0.75,
	input_size_height=192,
	input_size_width=192,
	anchor_offset_x=0.5,
	anchor_offset_y=0.5,
	strides=[8, 16, 16, 16],
	aspect_ratios=[1.0],
	reduce_boxes_in_lowest_layer=False,
	interpolated_scale_aspect_ratio=1.0,
	fixed_anchor_size=True,
)

anchors = mpu.generate_anchors(anchor_options)

def pd_postprocess(inference, frame_size):
	scores = np.squeeze(inference[pd_scores])  # 2016
	bboxes = inference[pd_bboxes][0]  # 2016x18
	# Decode bboxes
	regions = mpu.decode_bboxes(
		pd_score_thresh, scores, bboxes, anchors
	)
	# Non maximum suppression
	regions = mpu.non_max_suppression(
		regions,
		pd_nms_thresh
	)
	regions = mpu.detections_to_rect(regions)
	regions = mpu.rect_transformation(
		regions,
		frame_size,
		frame_size
	)
	return regions

sum_dif_1 = 0
var_dif_1 = 0
sum_dif_2 = 0
var_dif_2 = 0

onnx_res = []
hailo_res = []

for i,d in enumerate(data):
    print(i)
    d = np.expand_dims(np.array([d[2], d[1], d[0]]), axis=0).astype(np.float32) / 255.0

    ort_session = onnxruntime.InferenceSession("./palm_detection.onnx")

    ort_inputs = {ort_session.get_inputs()[0].name: d.transpose(0, 2, 3, 1)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # print('onnx')
    # print('score shape', ort_outs[1].shape)
    # print('sum', np.sum(ort_outs[1]))
    # print('var', np.var(ort_outs[1]))
    # print()
    # print('bbox shape', ort_outs[0].shape)
    # print('sum', np.sum(ort_outs[0]))
    # print('var', np.var(ort_outs[0]))
    # print()
    regions_onnx = pd_postprocess({pd_scores: ort_outs[1], pd_bboxes: ort_outs[0]}, 192)
    for region in regions_onnx:
        # print(region.rect_points)
        onnx_res.append(region.rect_points)
    # print()



    hailo_output = hand_tracker_hailo.inference(d[0])

    # print('hailo')
    # print('score shape', hailo_output[pd_scores].shape)
    # print('sum', np.sum(hailo_output[pd_scores]))
    # print('var', np.var(hailo_output[pd_scores]))
    # print()
    # print('bbox hape', hailo_output[pd_bboxes].shape)
    # print('sum', np.sum(hailo_output[pd_bboxes]))
    # print('var', np.var(hailo_output[pd_bboxes]))
    # print()
    regions_hailo = pd_postprocess(hailo_output, 192)
    for region in regions_hailo:
        # print(region.rect_points)
        hailo_res.append(region.rect_points)
    # print()


    sum_dif_1 += abs(np.sum(ort_outs[1]) - np.sum(hailo_output[pd_scores])) / np.sum(ort_outs[1])
    var_dif_1 += abs(np.var(ort_outs[1]) - np.var(hailo_output[pd_scores])) / np.var(ort_outs[1])
    sum_dif_2 += abs(np.sum(ort_outs[0]) - np.sum(hailo_output[pd_bboxes])) / np.sum(ort_outs[0])
    var_dif_2 += abs(np.var(ort_outs[0]) - np.var(hailo_output[pd_bboxes])) / np.var(ort_outs[0])


    # print('---------------------------------------')

    # if i > 10:
    #     break

print()
print('average difference')
print()
print('score')
print('sum:', sum_dif_1 / i * 100, '%')
print('var:', var_dif_1 / i * 100, '%')
print()
print('bbox')
print('sum:', sum_dif_2 / i * 100, '%')
print('var:', var_dif_2 / i * 100, '%')

print()
print('region cnt')
print('onnx:', len(onnx_res))
print('hailo:', len(hailo_res))