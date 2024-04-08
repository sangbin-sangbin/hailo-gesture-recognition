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


with open("./data.json", "r") as f:
	data = json.load(f)["data"]


ht = HandTracker(
	pd_score_thresh=0.6,
	pd_nms_thresh=0.3,
	lm_score_threshold=0.6,
)

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



import numpy as np
import onnxruntime
import tensorflow as tf
import numpy as np
import cv2

for i,d in enumerate(data[:10]):
	# ri = ht.inference(np.array(d).astype(np.float32).transpose(1,2,0))
	# print('normal')
	# for region in ri:
	# 	print(region.rect_points)

	# r1_t = ht.test(np.array(d))
	# r1 = pd_postprocess(r1_t, 192)
	# print('sum', np.sum(r1_t[pd_scores]))
	# print('ov test')
	# for region in r1:
	# 	print(region.rect_points)

	d = np.expand_dims(np.array([d[2], d[1], d[0]]), axis=0).astype(np.float32) / 255.0


	# class TFLiteModel:
	# 	def __init__(self, model_path: str):
	# 		self.interpreter = tf.lite.Interpreter(model_path)
	# 		self.interpreter.allocate_tensors()

	# 		self.input_details = self.interpreter.get_input_details()
	# 		self.output_details = self.interpreter.get_output_details()

	# 	def predict(self, *data_args):
	# 		assert len(data_args) == len(self.input_details)
	# 		for data, details in zip(data_args, self.input_details):
	# 			self.interpreter.set_tensor(details["index"], data)
	# 		self.interpreter.invoke()
	# 		return [self.interpreter.get_tensor(self.output_details[0]["index"]), self.interpreter.get_tensor(self.output_details[1]["index"])]


	# model = TFLiteModel("./palm_detection_full.tflite")
	
	# res = model.predict(d.transpose(0,2,3,1))
	# ro = pd_postprocess({pd_scores: res[1], pd_bboxes: res[0]}, 192)

	# print('tflite')
	# print('sum', np.sum(res[1]))
	# for region in ro:
	# 	print(region.rect_points)



	# ort_session = onnxruntime.InferenceSession("./palm_detection.onnx")

	# ort_inputs = {ort_session.get_inputs()[0].name: d.transpose(0, 2, 3, 1)}
	# ort_outs = ort_session.run(None, ort_inputs)
	# print(ort_outs[0].shape)
	# print(ort_outs[0][0][0])
	# print(ort_outs[1].shape)
	# print(ort_outs[1][0][0])
	# print()

	# ro = pd_postprocess({pd_scores: ort_outs[1], pd_bboxes: ort_outs[0]}, 192)

	# print('onnx')
	# print('sum', np.sum(ort_outs[1]))
	# for region in ro:
	# 	print(region.rect_points)





	ort_session = onnxruntime.InferenceSession("./modified_palm_detection.onnx")

	# print(d.transpose(0, 2, 3, 1).shape)
	ort_inputs = {ort_session.get_inputs()[0].name: d.transpose(0, 2, 3, 1)}
	ort_outs = ort_session.run(None, ort_inputs)

	# print('modified_onnx')
	# print('sum 0', np.sum(ort_outs[0]))
	# print(ort_outs[0].shape)
	# print(ort_outs[0].transpose(0, 2, 3, 1).reshape(1, -1, 1).shape)
	# print(ort_outs[0].transpose(0, 2, 3, 1).reshape(1, -1, 1)[0][0])
	# print('sum 1', np.sum(ort_outs[1]))
	# print(ort_outs[1].shape)
	# print(ort_outs[1].transpose(0, 2, 3, 1).reshape(1, -1, 18).shape)
	# print(ort_outs[1].transpose(0, 2, 3, 1).reshape(1, -1, 18)[0][0])
	# print('sum 2', np.sum(ort_outs[2]))
	# print(ort_outs[2].shape)
	# print(ort_outs[2].transpose(0, 2, 3, 1).reshape(1, -1, 1).shape)
	# print(ort_outs[2].transpose(0, 2, 3, 1).reshape(1, -1, 1)[0][0])
	# print('sum 3', np.sum(ort_outs[3]))
	# print(ort_outs[3].shape)
	# print(ort_outs[3].transpose(0, 2, 3, 1).reshape(1, -1, 18).shape)
	# print(ort_outs[3].transpose(0, 2, 3, 1).reshape(1, -1, 18)[0][0])
	# print()




	# r2_t = hand_tracker_hailo.test(d)

	# r2 = pd_postprocess(r2_t, 192)
	# print('hailo')
	# print('sum', np.sum(r2_t[pd_scores]))
	# for region in r2:
	# 	print(region.rect_points)

	# print(d.transpose(0, 2, 3, 1).shape)
	# r2_t = hand_tracker_hailo.test(d.transpose(0, 2, 3, 1))
	# print('sum 0', np.sum(r2_t[0]))
	# print(r2_t[0].shape)
	# # print(r2_t[0].reshape(1, -1, 18).shape)
	# # print(r2_t[0].reshape(1, -1, 18)[0][0])
	# print('sum 1', np.sum(r2_t[1]))
	# print(r2_t[1].shape)
	# # print(r2_t[1].reshape(1, -1, 18).shape)
	# # print(r2_t[1].reshape(1, -1, 18)[0][0])
	# print('sum 2', np.sum(r2_t[2]))
	# print(r2_t[2].shape)
	# # print(r2_t[2].reshape(1, -1, 1).shape)
	# # print(r2_t[2].reshape(1, -1, 1)[0][0])
	# print('sum 3', np.sum(r2_t[3]))
	# print(r2_t[3].shape)
	# # print(r2_t[3].reshape(1, -1, 1).shape)
	# # print(r2_t[3].reshape(1, -1, 1)[0][0])
	# print()
	

	# print("onnx palm")
	dic = {pd_scores: np.concatenate((ort_outs[0].transpose(0, 2, 3, 1).reshape(1, -1, 1), ort_outs[2].transpose(0, 2, 3, 1).reshape(1, -1, 1)), axis=1), pd_bboxes: np.concatenate((ort_outs[1].transpose(0, 2, 3, 1).reshape(1, -1, 18), ort_outs[3].transpose(0, 2, 3, 1).reshape(1, -1, 18)), axis=1)}
	regions = pd_postprocess(dic, 192)
	for region in regions:
		print(region.rect_points)
	print()

	# print("hef palm")
	# dic = {pd_scores: np.concatenate((r2_t[3].reshape(1, -1, 1), r2_t[2].reshape(1, -1, 1)), axis=1), pd_bboxes: np.concatenate((r2_t[0].reshape(1, -1, 18), r2_t[1].reshape(1, -1, 18)), axis=1)}
	# regions = pd_postprocess(dic, 192)
	# for region in regions:
	# 	print(region.rect_points)
	# print()


	print('---------------------------------------')
	if i > 10:
		break