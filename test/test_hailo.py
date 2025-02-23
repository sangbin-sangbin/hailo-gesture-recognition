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

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def recognize_gesture(prev_gestures, time_threshold, infinite=False):
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


def update_prev_gestures(prev_gestures):
    STORE_TIME_THRES = 3
    while time.time() - prev_gestures[0]['time'] > STORE_TIME_THRES:
        prev_gestures.pop(0)
    return prev_gestures


def run(hand_tracker, model, cv2_util):
    gestures = config["gestures"]

    state = {
        "prev_gesture": len(gestures) - 1,
        "multi_action_start_time": -1,
        "multi_action_cnt": 0,
        "prev_action": ["", 0],
    }
    prev_gestures = []

    frame_num = 0

    subprocess.run(
        "adb connect 192.168.1.103:5555; adb root; adb connect 192.168.1.103:5555",
        shell=True,
        check=False,
    )

    recognized_hands = []
    recognized_hand = []
    prob_text = ""

    recognizing = False
    recognized_hand_prev_pos = [-999, -999]

    last_hand_time = time.time()

    wake_up_state = []

    while True:
        cv2_util.fps.update()
        # require more time than time_threshold to recognize it as an gesture
        time_threshold = (
            cv2.getTrackbarPos("time", "gesture recognition") / 100
        )
        # distance between this frame's hand and last frame's recognized hand should be smaller than same_hand_threshold to regard them as same hand
        same_hand_threshold = (
            cv2.getTrackbarPos("same_hand", "gesture recognition") * 100
        )
        landmark_skip_frame = max(
            cv2.getTrackbarPos("skip_frame", "gesture recognition"), 1
        )
        start_recognizing_time_threshold = cv2.getTrackbarPos(
            "start_time", "gesture recognition"
        )
        stop_recognizing_time_threshold = cv2.getTrackbarPos(
            "stop_time", "gesture recognition"
        )
        multi_action_time_threshold = cv2.getTrackbarPos(
            "multi_time", "gesture recognition"
        )
        multi_action_cooltime = (
            cv2.getTrackbarPos("multi_cooltime", "gesture recognition") / 10
        )

        ok, frame = cv2_util.read()
        if not ok:
            print("camera error")
            break

        frame_num += 1
        if frame_num % landmark_skip_frame == 0:
            # Process the frame with MediaPipe Hands
            regions, results = hand_tracker.inference(frame)


            right_hands = []
            recognized_hands = []

            if results:
                for result in results:
                    if result["handedness"] > 0.5:  # Right Hand
                        # Convert right hand coordinations for rendering
                        right_hands.append(result["landmark"])
                        recognized_hands.append(result["landmark"])

                if recognizing:
                    # find closest hand
                    hand_idx, recognized_hand_prev_pos = utils.same_hand_tracking(
                        right_hands, recognized_hand_prev_pos, same_hand_threshold
                    )

                    if hand_idx != -1:
                        last_hand_time = time.time()

                        recognized_hand = recognized_hands[hand_idx]
                        recognized_hand_prev_pos = utils.get_center(recognized_hand)

                        lst, _ = utils.normalize_points(recognized_hand)

                        res = list(
                            model.result_with_softmax(
                                torch.tensor(
                                    [element for row in lst for element in row],
                                    dtype=torch.float,
                                )
                            )
                        )

                        probability = max(res)
                        gesture_idx = (
                            res.index(probability) if probability >= config["gesture_prob_threshold"] else len(gestures) - 1
                        )
                        prev_gestures.append({"gesture": gesture_idx, "time": time.time()})
                        prev_gestures = update_prev_gestures(prev_gestures)

                        prob_text = f"{gestures[gesture_idx]} {int(probability * 100)}%"

                        if recognize_gesture(prev_gestures, multi_action_time_threshold, infinite=True) != -1:
                            if state["multi_action_start_time"] == -1:
                                state["multi_action_start_time"] = time.time()
                            if prev_gestures[-1]["time"] >= state["multi_action_start_time"] + multi_action_cooltime * state["multi_action_cnt"]:
                                state["prev_action"] = utils.perform_action(state["prev_action"][0], infinite=True)
                                state["multi_action_cnt"] += 1
                        else:
                            gesture = recognize_gesture(prev_gestures, time_threshold)
                            if gesture != -1:
                                if gestures[state["prev_gesture"]] == "default":
                                    state["prev_action"] = utils.perform_action(gestures[gesture])
                                state["prev_gesture"] = gesture
                                state["multi_action_start_time"] = -1
                                state["multi_action_cnt"] = 0
                            else:
                                state = {
                                    "prev_gesture": state["prev_gesture"],
                                    "multi_action_start_time": -1,
                                    "multi_action_cnt": 0,
                                    "prev_action": ["", 0],
                                }
                    else:
                        # stop recognizing
                        recognized_hand = []
                        prob_text = ""
                        if (
                            recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold
                        ):
                            print("stop recognizing")
                            utils.play_audio_file("Stop")
                            recognizing = False
                            state = {
                                "prev_gesture": len(gestures) - 1,
                                "multi_action_start_time": -1,
                                "multi_action_cnt": 0,
                                "prev_action": ["", 0],
                            }
                else:
                    # when not recognizing, get hands with 'default' gesture and measure elapsed time
                    delete_list = []
                    checked = [0 for _ in range(len(right_hands))]
                    for i, [prev_pos, prev_gestures] in enumerate(wake_up_state):
                        hand_idx, prev_pos = utils.same_hand_tracking(
                            right_hands, prev_pos, same_hand_threshold
                        )
                        if hand_idx == -1:
                            delete_list = [i] + delete_list
                        elif recognize_gesture(prev_gestures, start_recognizing_time_threshold) == 0:
                            # when there are default gestured hand for enough time, start recognizing and track the hand
                            print("start recognizing")
                            recognized_hand_prev_pos = utils.get_center(right_hands[hand_idx])
                            utils.play_audio_file("Start")
                            recognizing = True
                            wake_up_state = []
                            break
                        else:
                            lst, _ = utils.normalize_points(right_hands[hand_idx])
                            res = list(
                                model.result_with_softmax(
                                    torch.tensor(
                                        [element for row in lst for element in row],
                                        dtype=torch.float,
                                    )
                                )
                            )
                            probability = max(res)
                            gesture_idx = (
                                res.index(probability) if probability >= config["gesture_prob_threshold"] else len(gestures) - 1
                            )
                            checked[hand_idx] = 1
                            prev_gestures.append({"gesture": gesture_idx, "time": time.time()})

                    # wake_up_state refreshing
                    if not recognizing:
                        for i in delete_list:
                            wake_up_state.pop(i)

                        for idx, _ in enumerate(checked):
                            if checked[idx] == 0:
                                lst, _ = utils.normalize_points(right_hands[idx])
                                res = list(
                                    model.result_with_softmax(
                                        torch.tensor(
                                            [element for row in lst for element in row],
                                            dtype=torch.float,
                                        )
                                    )
                                )
                                probability = max(res)
                                gesture_idx = (
                                    res.index(probability) if probability >= config["gesture_prob_threshold"] else len(gestures) - 1
                                )
                                wake_up_state.append(
                                    [utils.get_center(right_hands[idx]), [{"gesture": gesture_idx, "time": time.time()}]]
                                )
            else:
                # stop recognizing
                recognized_hands = []
                recognized_hand = []
                prob_text = ""
                if (
                    recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold
                ):
                    print("stop recognizing")
                    utils.play_audio_file("Stop")
                    recognizing = False
                    state = {
                        "prev_gesture": len(gestures) - 1,
                        "multi_action_start_time": -1,
                        "multi_action_cnt": 0,
                        "prev_action": ["", 0],
                    }

                

        annotated_frame = cv2_util.annotated_frame(frame)

        for rh in recognized_hands:
            annotated_frame = cv2_util.print_landmark(annotated_frame, rh)
        if len(recognized_hand) > 0:
            annotated_frame = cv2_util.print_landmark(annotated_frame, recognized_hand, (255, 0, 0))

        for region in regions:
            for x, y in region.rect_points:
                cv2.circle(annotated_frame, (x, y), 6, (255, 0, 0), -1)

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
        if time.time() - state["prev_action"][1] < time_threshold * 2:
            cv2.putText(
                annotated_frame,
                state["prev_action"][0],
                (
                    annotated_frame.shape[1] // 2 + 250,
                    annotated_frame.shape[0] // 2 - 100,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                3,
            )

        annotated_frame = cv2_util.unpad(annotated_frame)

        cv2_util.fps.display(annotated_frame, orig=(50, 50), color=(240, 180, 100))

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
    ht = HandTrackerHailo(
        pd_score_thresh=0.6,
        pd_nms_thresh=0.3,
        lm_score_threshold=0.6,
    )

    model_obj = initialize_model()

    PARAMETERS_DIR = "./parameters.json"
    my_parameters = load_parameters(PARAMETERS_DIR)
    create_trackbars(my_parameters)

    cv2_util_obj = CV2Utils()

    run(ht, model_obj, cv2_util_obj)

    # Release the webcam and close all windows
    cv2_util_obj.cap.release()
    cv2.destroyAllWindows()

    # save_current_parameters(PARAMETERS_DIR, my_parameters)