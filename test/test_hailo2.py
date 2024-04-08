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
from openvino_utils.hand_tracker_hailo import HandTrackerHailo

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class Drawer():
    def draw(frame, cv2_utils, prob_text, recognized_hands, recognized_hand, prev_action, time_threshold):
        annotated_frame = cv2_utils.annotated_frame(frame)

        for rh in recognized_hands:
            annotated_frame = cv2_utils.print_landmark(annotated_frame, rh)
        if len(recognized_hand) > 0:
            annotated_frame = cv2_utils.print_landmark(annotated_frame, recognized_hand, (255, 0, 0))

        # Print Current Hand's Gesture
        cv2.putText(
            annotated_frame,
            prob_text,
            (annotated_frame.shape[1] // 2 + 230, annotated_frame.shape[0] // 2 - 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )

        # print recognized gesture
        if time.time() - prev_action[1] < time_threshold * 2:
            cv2.putText(
                annotated_frame,
                prev_action[0],
                (
                    annotated_frame.shape[1] // 2 + 250,
                    annotated_frame.shape[0] // 2 - 100,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )

        annotated_frame = cv2_utils.unpad(annotated_frame)

        cv2_utils.fps.display(annotated_frame, orig=(50, 50), color=(240, 180, 100))

        return annotated_frame


class Test():
    def __init__(self, hand_tracker, model):
        self.gestures = config["gestures"]

        self.model = model
        self.hand_tracker = hand_tracker

        self.state = {
            "prev_gesture": len(self.gestures) - 1,
            "multi_action_start_time": -1,
            "multi_action_cnt": 0,
            "prev_action": ["", 0],
        }
        self.prev_gestures = []

        self.frame_num = 0
        self.landmark_skip_frame = 1

        subprocess.run(
            "adb connect 192.168.1.103:5555; adb root; adb connect 192.168.1.103:5555",
            shell=True,
            check=False,
        )

        self.recognized_hands = []
        self.recognized_hand = []
        self.prob_text = ""

        self.recognizing = False
        self.recognized_hand_prev_pos = [-999, -999]

        self.last_hand_time = time.time()

        self.wake_up_state = []


    def recognize_gesture(self, prev_gestures, time_threshold, infinite=False):
        CNT_THRES = 0.8
        FPS = 25
        LEN_THRES = 0.9

        if len(prev_gestures) == 0:
            return -1

        cnt_per_gesture = [0 for _ in enumerate(config["gestures"])]
        cnt = 0
        start_time = -1
        for i, prev_gesture in enumerate(prev_gestures):
            if time.time() - prev_gesture['time'] < time_threshold:
                cnt_per_gesture[prev_gesture['gesture']] += 1
                cnt += 1
                if start_time == -1:
                    if i > 0:
                        cnt_per_gesture[prev_gestures[i - 1]['gesture']] += 1
                        cnt += 1
                        start_time = prev_gestures[i - 1]['time']
                    else:
                        start_time = prev_gesture['time']
        max_cnt = max(cnt_per_gesture)
        if cnt > 0 and max_cnt / cnt > CNT_THRES and ((time.time() - start_time > time_threshold and cnt > FPS * time_threshold * LEN_THRES) or infinite):
            return cnt_per_gesture.index(max(cnt_per_gesture))

        return -1


    def update_prev_gestures(self, prev_gestures):
        STORE_TIME_THRES = 3
        while time.time() - prev_gestures[0]['time'] > STORE_TIME_THRES:
            prev_gestures.pop(0)
        return prev_gestures


    def stop_recognizing(self):
        print("stop recognizing")
        utils.play_audio_file("Stop")

        self.recognized_hands = []
        self.recognized_hand = []
        self.prob_text = ""

        self.recognizing = False
        self.state = {
            "prev_gesture": len(self.gestures) - 1,
            "multi_action_start_time": -1,
            "multi_action_cnt": 0,
            "prev_action": ["", 0],
        }


    def start_recognizing(self, r_hand):
        print("start recognizing")
        self.recognized_hand_prev_pos = utils.get_center(r_hand)
        utils.play_audio_file("Start")
        self.recognizing = True
        self.wake_up_state = []


    def update_parameter(self):
        self.time_threshold = (
            cv2.getTrackbarPos("time", "gesture recognition") / 100
        )
        # distance between this frame's hand and last frame's recognized hand should be smaller than same_hand_threshold to regard them as same hand
        self.same_hand_threshold = (
            cv2.getTrackbarPos("same_hand", "gesture recognition") * 100
        )
        self.landmark_skip_frame = max(
            cv2.getTrackbarPos("skip_frame", "gesture recognition"), 1
        )
        self.start_recognizing_time_threshold = cv2.getTrackbarPos(
            "start_time", "gesture recognition"
        )
        self.stop_recognizing_time_threshold = cv2.getTrackbarPos(
            "stop_time", "gesture recognition"
        )
        self.multi_action_time_threshold = cv2.getTrackbarPos(
            "multi_time", "gesture recognition"
        )
        self.multi_action_cooltime = (
            cv2.getTrackbarPos("multi_cooltime", "gesture recognition") / 10
        )
        

    def recognizing_with_target_hand(self, hand_idx):
        self.last_hand_time = time.time()

        self.recognized_hand = self.recognized_hands[hand_idx]
        self.recognized_hand_prev_pos = utils.get_center(self.recognized_hand)

        lst, _ = utils.normalize_points(self.recognized_hand)

        res = list(
            self.model.result_with_softmax(
                torch.tensor(
                    [element for row in lst for element in row],
                    dtype=torch.float,
                )
            )
        )

        probability = max(res)
        gesture_idx = (
            res.index(probability) if probability >= config["gesture_prob_threshold"] else len(self.gestures) - 1
        )
        self.prev_gestures.append({"gesture": gesture_idx, "time": time.time()})
        self.prev_gestures = self.update_prev_gestures(self.prev_gestures)

        self.prob_text = f"{self.gestures[gesture_idx]} {int(probability * 100)}%"

        if self.recognize_gesture(self.prev_gestures, self.multi_action_time_threshold, infinite=True) != -1:
            if self.state["multi_action_start_time"] == -1:
                self.state["multi_action_start_time"] = time.time()
            if self.prev_gestures[-1]["time"] >= self.state["multi_action_start_time"] + self.multi_action_cooltime * self.state["multi_action_cnt"]:
                self.state["prev_action"] = utils.perform_action(self.state["prev_action"][0], infinite=True)
                self.state["multi_action_cnt"] += 1
        else:
            gesture = self.recognize_gesture(self.prev_gestures, self.time_threshold)
            if gesture != -1:
                if self.gestures[self.state["prev_gesture"]] == "default":
                    self.state["prev_action"] = utils.perform_action(self.gestures[gesture])
                self.state["prev_gesture"] = gesture
                self.state["multi_action_start_time"] = -1
                self.state["multi_action_cnt"] = 0
            else:
                self.state = {
                    "prev_gesture": self.state["prev_gesture"],
                    "multi_action_start_time": -1,
                    "multi_action_cnt": 0,
                    "prev_action": ["", 0],
                }


    def not_recognizing(self):
        # when not recognizing, get hands with 'default' gesture and measure elapsed time
        delete_list = []
        checked = [0 for _ in range(len(self.recognized_hands))]
        for i, [prev_pos, self.prev_gestures] in enumerate(self.wake_up_state):
            hand_idx, prev_pos = utils.same_hand_tracking(
                self.recognized_hands, prev_pos, self.same_hand_threshold
            )
            if hand_idx == -1:
                delete_list = [i] + delete_list
            elif self.recognize_gesture(self.prev_gestures, self.start_recognizing_time_threshold) == 0:
                # when there are default gestured hand for enough time, start recognizing and track the hand
                self.start_recognizing(self.recognized_hands[hand_idx])
                break
            else:
                lst, _ = utils.normalize_points(self.recognized_hands[hand_idx])
                res = list(
                    self.model.result_with_softmax(
                        torch.tensor(
                            [element for row in lst for element in row],
                            dtype=torch.float,
                        )
                    )
                )
                probability = max(res)
                gesture_idx = (
                    res.index(probability) if probability >= config["gesture_prob_threshold"] else len(self.gestures) - 1
                )
                checked[hand_idx] = 1
                self.prev_gestures.append({"gesture": gesture_idx, "time": time.time()})

        # wake_up_state refreshing
        if not self.recognizing:
            for i in delete_list:
                self.wake_up_state.pop(i)

            for idx, _ in enumerate(checked):
                if checked[idx] == 0:
                    lst, _ = utils.normalize_points(self.recognized_hands[idx])
                    res = list(
                        self.model.result_with_softmax(
                            torch.tensor(
                                [element for row in lst for element in row],
                                dtype=torch.float,
                            )
                        )
                    )
                    probability = max(res)
                    gesture_idx = (
                        res.index(probability) if probability >= config["gesture_prob_threshold"] else len(self.gestures) - 1
                    )
                    self.wake_up_state.append(
                        [utils.get_center(self.recognized_hands[idx]), [{"gesture": gesture_idx, "time": time.time()}]]
                    )

    def recognize_action(self, frame):
        # Process the frame with MediaPipe Hands
        regions, results = self.hand_tracker.inference(frame)

        self.recognized_hands = []
        if results:
            for result in results:
                if result["handedness"] > 0.5:  # Right Hand
                    # Convert right hand coordinations for rendering
                    self.recognized_hands.append(result["landmark"])

            if self.recognizing:
                # find closest hand
                hand_idx, self.recognized_hand_prev_pos = utils.same_hand_tracking(
                    self.recognized_hands, self.recognized_hand_prev_pos, self.same_hand_threshold
                )

                if hand_idx != -1:
                    self.recognizing_with_target_hand(hand_idx)
                else:
                    # stop recognizing
                    if (self.recognizing and time.time() - self.last_hand_time > self.stop_recognizing_time_threshold):
                        self.stop_recognizing()
            else:
                self.not_recognizing()
        else:
            # stop recognizing
            if (self.recognizing and time.time() - self.last_hand_time > self.stop_recognizing_time_threshold):
                self.stop_recognizing()

        for region in regions:
            for x, y in region.rect_points:
                cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)


    def run(self, cv2_utils):
        while True:
            cv2_utils.fps.update()
            # require more time than time_threshold to recognize it as an gesture
            self.update_parameter()

            ok, frame = cv2_utils.read()
            if not ok:
                break

            self.frame_num += 1
            if self.frame_num % self.landmark_skip_frame == 0:
                self.recognize_action(frame)


            
            annotated_frame = Drawer.draw(frame, cv2_utils, self.prob_text, self.recognized_hands, self.recognized_hand, self.state["prev_action"], self.time_threshold)
            cv2.imshow("gesture recognition", annotated_frame)

            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)


def initialize_model():
    model = Model()
    if os.path.exists("../models/model.pt"):
        model.load_state_dict(torch.load("../models/model.pt"))
    else:
        model.load_state_dict(torch.load("../models/base_model.pt"))
    return model


def load_parameters(parameters_dir):
    res = "y"  # input("want to use saved parameters? [ y / n ]\n>>> ")
    if res == "y":
        parameter = json.load(open(parameters_dir))
    else:
        parameter = {
            "time": 10,
            "same_hand": 10,
            "skip_frame": 1,
            "start_time": 1,
            "stop_time": 1,
            "multi_time": 1,
            "multi_cooltime": 2,
        }
    return parameter


def create_trackbars(parameter):
    cv2.namedWindow("gesture recognition")
    cv2.createTrackbar(
        "time", "gesture recognition", parameter["time"], 100, utils.nothing
    )
    cv2.createTrackbar(
        "same_hand", "gesture recognition", parameter["same_hand"], 100, utils.nothing
    )
    cv2.createTrackbar(
        "skip_frame", "gesture recognition", parameter["skip_frame"], 50, utils.nothing
    )
    cv2.createTrackbar(
        "start_time", "gesture recognition", parameter["start_time"], 10, utils.nothing
    )
    cv2.createTrackbar(
        "stop_time", "gesture recognition", parameter["stop_time"], 10, utils.nothing
    )
    cv2.createTrackbar(
        "multi_time", "gesture recognition", parameter["multi_time"], 10, utils.nothing
    )
    cv2.createTrackbar(
        "multi_cooltime",
        "gesture recognition",
        parameter["multi_cooltime"],
        10,
        utils.nothing,
    )


def save_current_parameters(parameters_dir, parameter):
    res = input("want to save current parameters? [ y / n ]\n>>> ")
    if res == "y":
        with open(parameters_dir, "w") as f:
            json.dump(parameter, f)


if __name__ == "__main__":    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # This could cause error
    pid = os.getpid() 
    cpu_process = Process(target=utils.monitor, args=(config["device"], pid))
    cpu_process.start()

    # Get the directory of test.py
    current_dir = os.path.dirname(os.path.relpath(__file__))

    # Construct the path to palm_detection.xml
    pd_model_path = os.path.join(
        current_dir,
        "..",
        "openvino_utils",
        "mediapipe_models",
        "palm_detection_FP16.xml"
    )
    lm_model_path = os.path.join(
        current_dir,
        "..",
        "openvino_utils",
        "mediapipe_models",
        "hand_landmark_FP16.xml"
    )

    ht = HandTrackerHailo(
        pd_score_thresh=0.6,
        pd_nms_thresh=0.3,
        lm_score_threshold=0.6,
    )

    model_obj = initialize_model()

    PARAMETERS_DIR = "./parameters.json"
    my_parameters = load_parameters(PARAMETERS_DIR)
    create_trackbars(my_parameters)

    cv2_utils = CV2Utils()

    test = Test(ht, model_obj)
    test.run(cv2_utils)

    # Release the webcam and close all windows
    cv2_utils.cap.release()
    cv2.destroyAllWindows()

    # save_current_parameters(PARAMETERS_DIR, my_parameters)