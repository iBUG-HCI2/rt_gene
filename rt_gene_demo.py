import os
import cv2
import math
import time
import numpy as np
from argparse import ArgumentParser

import rt_gene
from rt_gene.estimate_gaze_base import GazeEstimatorBase

from saic_vision.face_alignment import FaceAlignment
from saic_vision.object_detector import S3FMobileV2Detector
from saic_vision.head_pose_estimation import HeadPoseEstimator

import torch


def plot_landmarks(frame, landmarks, scores, threshold):
    for idx in range(len(landmarks) - 1):
        if (idx != 16 and idx != 21 and idx != 26 and idx != 30 and
                idx != 35 and idx != 41 and idx != 47 and idx != 59):
            if scores[idx] >= threshold and scores[idx + 1] >= threshold:
                cv2.line(frame, tuple(landmarks[idx].astype(int).tolist()),
                         tuple(landmarks[idx + 1].astype(int).tolist()),
                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        if idx == 30:
            if scores[30] >= threshold and scores[33] >= threshold:
                cv2.line(frame, tuple(landmarks[30].astype(int).tolist()),
                         tuple(landmarks[33].astype(int).tolist()),
                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        elif idx == 36:
            if scores[36] >= threshold and scores[41] >= threshold:
                cv2.line(frame, tuple(landmarks[36].astype(int).tolist()),
                         tuple(landmarks[41].astype(int).tolist()),
                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        elif idx == 42:
            if scores[42] >= threshold and scores[47] >= threshold:
                cv2.line(frame, tuple(landmarks[42].astype(int).tolist()),
                         tuple(landmarks[47].astype(int).tolist()),
                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        elif idx == 48:
            if scores[48] >= threshold and scores[59] >= threshold:
                cv2.line(frame, tuple(landmarks[48].astype(int).tolist()),
                         tuple(landmarks[59].astype(int).tolist()),
                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        elif idx == 60:
            if scores[60] >= threshold and scores[67] >= threshold:
                cv2.line(frame, tuple(landmarks[60].astype(int).tolist()),
                         tuple(landmarks[67].astype(int).tolist()),
                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    for landmark, score in zip(landmarks, scores):
        if score >= threshold:
            cv2.circle(frame, tuple(landmark.astype(int).tolist()), 1, (0, 0, 255), -1)


def main():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', default=None,
                        help='video source, either a path to a video or the camera source (as an int)')
    args = parser.parse_args()

    # Make the models run a bit faster
    torch.backends.cudnn.benchmark = True

    # Load face models
    detector = S3FMobileV2Detector(th=0.25, device='cuda:0')
    landmarker = FaceAlignment(device='cuda:0')
    head_pose_estimator = HeadPoseEstimator()
    print('Face detector and landmark detectors loaded.')

    # Hack the head pose estimator
    rt_gene_model_dir = os.path.realpath(os.path.join(os.path.dirname(rt_gene.__file__), '..', '..', 'model_nets'))
    with open(os.path.join(rt_gene_model_dir, 'face_model_68.txt'), 'r') as mean_face_file:
        mean_face = np.array([float(x) for x in mean_face_file.read().splitlines()]).reshape(3, -1).transpose()
        head_pose_estimator._mean_shape = np.vstack((mean_face[17:60, :2], mean_face[61:64, :2], mean_face[65:, :2]))
    head_pose_estimator._coefficients = np.eye(3)

    # Load gaze model
    model_path = os.path.join(rt_gene_model_dir, 'Model_allsubjects1.h5')
    gaze_estimator = GazeEstimatorBase(device_id_gaze="gpu:0", model_files=model_path)

    if os.path.exists(args.video):
        vid = cv2.VideoCapture(args.video)
        print('Video file opened: %s.' % args.video)
    else:
        vid = cv2.VideoCapture(int(args.video))
        print('Webcam #%d opened.' % int(args.video))

    # Detect faces in the frames
    try:
        frame_number = 0
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        print('Face alignment started, press \'Q\' to quit.')
        while True:
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect faces
                bboxes, labels, probs = detector.detect_from_image(frame)
                face_boxes = [bbox for bbox, label in zip(bboxes, labels) if label == 1]

                # Localise landmarks
                landmarks_list, scores_list = landmarker.points_from_image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face_boxes)

                # Prepare for gaze detection
                valid_faces = []
                left_eye_images = []
                right_eye_images = []
                head_poses = []
                for idx, (landmarks, scores) in enumerate(zip(landmarks_list, scores_list)):
                    if scores.min() >= 0.2:
                        # Get left eye box
                        left_eye_corners = np.vstack((landmarks[36], landmarks[39]))
                        left_eye_width = left_eye_corners[1, 0] - left_eye_corners[0, 0]
                        left_eye_center = left_eye_corners.mean(axis=0)
                        left_eye_box = np.vstack((left_eye_center - [left_eye_width, 0],
                                                  left_eye_center + [left_eye_width, 0]))
                        left_eye_box[0, 1] -= (left_eye_box[1, 0] - left_eye_box[0, 0]) * 0.3
                        left_eye_box[1, 1] += (left_eye_box[1, 0] - left_eye_box[0, 0]) * 0.3
                        left_eye_box = left_eye_box.round().astype(int)

                        # Get right eye box
                        right_eye_corners = np.vstack((landmarks[42], landmarks[45]))
                        right_eye_width = right_eye_corners[1, 0] - right_eye_corners[0, 0]
                        right_eye_center = right_eye_corners.mean(axis=0)
                        right_eye_box = np.vstack((right_eye_center - [right_eye_width, 0],
                                                   right_eye_center + [right_eye_width, 0]))
                        right_eye_box[0, 1] -= (right_eye_box[1, 0] - right_eye_box[0, 0]) * 0.3
                        right_eye_box[1, 1] += (right_eye_box[1, 0] - right_eye_box[0, 0]) * 0.3
                        right_eye_box = right_eye_box.round().astype(int)

                        # Extract eye patches
                        if (0 <= left_eye_box[0, 0] < left_eye_box[1, 0] <= frame.shape[1] and
                                0 <= left_eye_box[0, 1] < left_eye_box[1, 1] <= frame.shape[0] and
                                0 <= right_eye_box[0, 0] < right_eye_box[1, 0] <= frame.shape[1] and
                                0 <= right_eye_box[0, 1] < right_eye_box[1, 1] <= frame.shape[0]):
                            left_eye_images.append(gaze_estimator.input_from_image(
                                cv2.resize(frame[left_eye_box[0, 1]: left_eye_box[1, 1],
                                           left_eye_box[0, 0]: left_eye_box[1, 0], :],
                                           (60, 36), interpolation=cv2.INTER_CUBIC)))
                            right_eye_images.append(gaze_estimator.input_from_image(
                                cv2.resize(frame[right_eye_box[0, 1]: right_eye_box[1, 1],
                                           right_eye_box[0, 0]: right_eye_box[1, 0], :],
                                           (60, 36), interpolation=cv2.INTER_CUBIC)))

                            # Get head pose
                            pitch, yaw, _ = head_pose_estimator.estimate_head_pose(landmarks)
                            head_poses.append([-pitch / 180.0 * math.pi, -yaw / 180.0 * math.pi])

                            valid_faces.append(idx)

                # Predict eye gaze
                start_time = time.time()
                eye_gazes = np.zeros_like(head_poses)
                if len(valid_faces) > 0:
                    eye_gazes = gaze_estimator.estimate_gaze_twoeyes(left_eye_images, right_eye_images, head_poses)
                elapsed_time = time.time() - start_time

                # Plot landmarks
                for landmarks, scores in zip(landmarks_list, scores_list):
                    plot_landmarks(frame, landmarks, scores, 0.2)

                # Plot head pose and eye gazes
                for idx, head_pose, eye_gaze in zip(valid_faces, head_poses, eye_gazes):
                    nose_tip = landmarks_list[idx][30]
                    end_point = nose_tip - 80.0 * np.array([math.cos(head_pose[0]) * math.sin(head_pose[1]),
                                                            math.sin(head_pose[0])])
                    cv2.line(frame, tuple(nose_tip.astype(int).tolist()), tuple(end_point.astype(int).tolist()),
                             (0, 0, 255), 2, cv2.LINE_AA)
                    gaze_displacement = 80.0 * np.array([math.cos(eye_gaze[0]) * math.sin(eye_gaze[1]),
                                                         math.sin(eye_gaze[0])])
                    left_eye_center = (landmarks_list[idx][36] + landmarks_list[idx][39]) / 2.0
                    end_point = left_eye_center - gaze_displacement
                    cv2.line(frame, tuple(left_eye_center.astype(int).tolist()), tuple(end_point.astype(int).tolist()),
                             (0, 255, 0), 2, cv2.LINE_AA)
                    right_eye_center = (landmarks_list[idx][42] + landmarks_list[idx][45]) / 2.0
                    end_point = right_eye_center - gaze_displacement
                    cv2.line(frame, tuple(right_eye_center.astype(int).tolist()), tuple(end_point.astype(int).tolist()),
                             (0, 255, 0), 2, cv2.LINE_AA)

                # Show the result
                print('Frame #%d: Eye gaze estimated from %d faces in %.04f ms.' %
                      (frame_number, len(valid_faces), elapsed_time * 1000.0))
                cv2.imshow(script_name, frame)
                key = cv2.waitKey(1) % 2 ** 16
                if key == ord('q') or key == ord('Q'):
                    print("\'Q\' pressed, we are done here.")
                    break
                else:
                    frame_number += 1
    finally:
        cv2.destroyAllWindows()
        vid.release()
        print('We are done here.')


if __name__ == '__main__':
    main()
