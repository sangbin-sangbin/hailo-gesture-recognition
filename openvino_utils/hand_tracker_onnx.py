import os
import re
import sys
import time

import cv2
import numpy as np
import openvino.runtime as ov
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from openvino_utils import mediapipe_utils as mpu


import numpy as np
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import cv2
import glob
import time

import onnxruntime


with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class HandTrackerONNX:
    def __init__(
        self,
        pd_score_thresh=0.1,
        pd_nms_thresh=0.3,
        lm_score_threshold=0.1,
    ):
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_score_threshold = lm_score_threshold

        self.dataset = []
        self.state = "break"
        self.start_time = time.time()
        self.gesture_num = 0
        self.gestures = config["gestures"]

        # Create SSD anchors
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
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
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Load Openvino models
        self.load_models()

    # Getter method
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    # Setter method
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def load_models(self):
        self.target = VDevice()

        # Loading compiled HEFs to device:
        palm_hef_path = './palm_detection.hef'
        self.palm_hef = HEF(palm_hef_path)

        hand_hef_path = './hand_landmark.hef'
        self.hand_hef = HEF(hand_hef_path)

        self.pd_scores = "Identity_1"
        self.pd_bboxes = "Identity"

        self.lm_score = "Identity_1"
        self.lm_handedness = "Identity_2"
        self.lm_landmarks = "Identity_dense/BiasAdd/Add"
        

        # Configure network groups
        self.palm_configure_params = ConfigureParams.create_from_hef(hef=self.palm_hef, interface=HailoStreamInterface.PCIe)
        self.palm_network_groups = self.target.configure(self.palm_hef, self.palm_configure_params)
        self.palm_network_group = self.palm_network_groups[0]
        self.palm_network_group_params = self.palm_network_group.create_params()

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False .
        self.palm_input_vstreams_params = InputVStreamParams.make(self.palm_network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.palm_output_vstreams_params = OutputVStreamParams.make(self.palm_network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.palm_input_vstream_info = self.palm_hef.get_input_vstream_infos()[0]
        self.palm_output_vstream_info_1 = self.palm_hef.get_output_vstream_infos()[0]
        self.palm_output_vstream_info_2 = self.palm_hef.get_output_vstream_infos()[1]
        self.palm_output_vstream_info_3 = self.palm_hef.get_output_vstream_infos()[2]
        self.palm_output_vstream_info_4 = self.palm_hef.get_output_vstream_infos()[3]
        
        palm_input_vstream_info = self.palm_hef.get_input_vstream_infos()[0]
        self.pd_h, self.pd_w, _ = palm_input_vstream_info.shape


        # Configure network groups
        self.hand_configure_params = ConfigureParams.create_from_hef(hef=self.hand_hef, interface=HailoStreamInterface.PCIe)
        self.hand_network_groups = self.target.configure(self.hand_hef, self.hand_configure_params)
        self.hand_network_group = self.hand_network_groups[0]
        self.hand_network_group_params = self.hand_network_group.create_params()

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False .
        self.hand_input_vstreams_params = InputVStreamParams.make(self.hand_network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.hand_output_vstreams_params = OutputVStreamParams.make(self.hand_network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.hand_input_vstream_info = self.hand_hef.get_input_vstream_infos()[0]
        self.hand_output_vstream_info_1 = self.hand_hef.get_output_vstream_infos()[0]
        self.hand_output_vstream_info_2 = self.hand_hef.get_output_vstream_infos()[1]
        self.hand_output_vstream_info_3 = self.hand_hef.get_output_vstream_infos()[2]
        self.hand_output_vstream_info_4 = self.hand_hef.get_output_vstream_infos()[3]

        hand_input_vstream_info = self.hand_hef.get_input_vstream_infos()[0]
        self.lm_h, self.lm_w, _ = hand_input_vstream_info.shape

        # Get palm detection
        self.pd_session = onnxruntime.InferenceSession("./palm_detection.onnx")
        self.lm_session = onnxruntime.InferenceSession("./hand_landmark.onnx")


    def pd_postprocess(self, inference, frame_size):
        scores = np.squeeze(inference[self.pd_scores])  # 2016
        bboxes = inference[self.pd_bboxes][0]  # 2016x18
        # Decode bboxes
        regions = mpu.decode_bboxes(
            self.pd_score_thresh, scores, bboxes, self.anchors
        )
        # Non maximum suppression
        regions = mpu.non_max_suppression(
            regions,
            self.pd_nms_thresh
        )
        regions = mpu.detections_to_rect(regions)
        regions = mpu.rect_transformation(
            regions,
            frame_size,
            frame_size
        )
        return regions

    def lm_postprocess(self, region, inference):
        region.lm_score = np.squeeze(inference[self.lm_score])
        region.handedness = np.squeeze(inference[self.lm_handedness])
        lm_raw = np.squeeze(inference[self.lm_landmarks])

        lm = []
        for i in range(int(len(lm_raw) / 3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3 * i: 3 * (i + 1)] / self.lm_w)
        region.landmarks = lm
        return region

    def lm_xy_coordinates(self, region):
        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        dst = np.array(
            [(x, y) for x, y in region.rect_points[1:]], dtype=np.float32
        )  # region.rect_points[0] is left bottom point !
        mat = cv2.getAffineTransform(src, dst)
        lm_xy = np.expand_dims(
            np.array([(lm[0], lm[1]) for lm in region.landmarks]), axis=0
        )
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)
        return lm_xy

    def inference(self, frame):
        h, w = frame.shape[:2]

        # Padding on the small side to get a square shape
        frame_size = max(h, w)
        pad_h = int((frame_size - h) / 2)
        pad_w = int((frame_size - w) / 2)
        frame = cv2.copyMakeBorder(
            frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT
        )

        # Resize image to NN square input shape
        frame_nn = cv2.resize(
            frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA
        )

        # Transpose hxwx3 -> 1x3xhxw
        frame_nn = np.transpose(frame_nn, (2, 0, 1))
        frame_nn = np.expand_dims(np.array([frame_nn[2], frame_nn[1], frame_nn[0]]), axis=0).astype(np.float32) / 255.0
        frame_nn = np.transpose(frame_nn, (0, 2, 3, 1))

        

        ort_inputs = {self.pd_session.get_inputs()[0].name: frame_nn}
        pd_outs = self.pd_session.run(None, ort_inputs)

        regions = self.pd_postprocess({self.pd_scores: pd_outs[1], self.pd_bboxes: pd_outs[0]}, frame_size)

        # Hand landmarks
        results = []
        for region in regions:
            frame_nn = mpu.warp_rect_img(
                region.rect_points, frame, self.lm_w, self.lm_h
            )
            # Transpose hxwx3 -> 1x3xhxw
            frame_nn = np.transpose(frame_nn, (2, 0, 1))
            frame_nn = np.expand_dims(np.array([frame_nn[2], frame_nn[1], frame_nn[0]]), axis=0).astype(np.float32) / 255.0
            frame_nn = np.transpose(frame_nn, (0, 2, 3, 1))

            ort_inputs = {self.lm_session.get_inputs()[0].name: frame_nn}
            lm_outs = self.lm_session.run(None, ort_inputs)
            lm_result = {self.lm_score: lm_outs[1], self.lm_handedness: lm_outs[2], self.lm_landmarks: lm_outs[0]}

            region = self.lm_postprocess(region, lm_result)

            results.append({"handedness": region.handedness, "landmark": self.lm_xy_coordinates(region)})

        return regions, results

    def test(self, frame_nn):
        with InferVStreams(self.palm_network_group, self.palm_input_vstreams_params, self.palm_output_vstreams_params) as palm_infer_pipeline:
            input_data = {self.palm_input_vstream_info.name: np.array(frame_nn)}
            with self.palm_network_group.activate(self.palm_network_group_params):
                pd_infer_results = palm_infer_pipeline.infer(input_data)
                return [pd_infer_results[self.palm_output_vstream_info_1.name], pd_infer_results[self.palm_output_vstream_info_2.name], pd_infer_results[self.palm_output_vstream_info_3.name], pd_infer_results[self.palm_output_vstream_info_4.name]]
                pd_result_1 = pd_infer_results[self.palm_output_vstream_info_1.name].reshape([1, -1, 18])
                pd_result_2 = pd_infer_results[self.palm_output_vstream_info_2.name].reshape([1, -1, 18])
                pd_result_3 = pd_infer_results[self.palm_output_vstream_info_3.name].reshape([1, -1, 1])
                pd_result_4 = pd_infer_results[self.palm_output_vstream_info_4.name].reshape([1, -1, 1])
                pd_result = {self.pd_scores: np.concatenate((pd_result_4, pd_result_3), axis=1), self.pd_bboxes: np.concatenate((pd_result_1, pd_result_2), axis=1)}
                # regions = self.pd_postprocess(pd_result, frame_size)
        return pd_result
                # print("hand tracker hailo")
                # print(pd_result[self.pd_scores].shape)
                # print(pd_result[self.pd_bboxes].shape)
                # print(np.sum(pd_result[self.pd_scores].flatten()))
                # print(np.sum(pd_result[self.pd_bboxes].flatten()))
                # print(np.max(pd_result[self.pd_scores].flatten()))
                # print(np.min(pd_result[self.pd_scores].flatten()))
                # print()
