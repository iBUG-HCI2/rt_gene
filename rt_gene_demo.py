import os
import cv2
import math
import time
import torch
import numpy as np
from typing import Tuple
from argparse import ArgumentParser

from rt_gene.src import __file__ as rt_gene_loc
from rt_gene.src.rt_gene.estimate_gaze_base import GazeEstimatorBase

from ibug.face_alignment import FANPredictor
from ibug.face_detection import S3FDPredictor
from ibug.face_alignment.utils import plot_landmarks
from ibug.face_pose_augmentation import TDDFAPredictor


def crop_eye_patches(frame: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if landmarks.shape[0] > 0:
        left_eye_widths = abs(landmarks[:, 39, 0] - landmarks[:, 36, 0])
        left_eye_centres = (landmarks[:, 39] + landmarks[:, 36]) / 2.0
        left_eye_boxes = np.stack((left_eye_centres, left_eye_centres), axis=1)
        left_eye_boxes[:, 0, 0] -= left_eye_widths
        left_eye_boxes[:, 1, 0] += left_eye_widths
        left_eye_boxes[:, 0, 1] -= left_eye_widths * 0.6
        left_eye_boxes[:, 1, 1] += left_eye_widths * 0.6
        left_eye_boxes = left_eye_boxes.round().astype(int)
        left_eye_boxes[:, 1, :] += 1

        right_eye_widths = abs(landmarks[:, 45, 0] - landmarks[:, 42, 0])
        right_eye_centres = (landmarks[:, 45] + landmarks[:, 42]) / 2.0
        right_eye_boxes = np.stack((right_eye_centres, right_eye_centres), axis=1)
        right_eye_boxes[:, 0, 0] -= right_eye_widths
        right_eye_boxes[:, 1, 0] += right_eye_widths
        right_eye_boxes[:, 0, 1] -= right_eye_widths * 0.6
        right_eye_boxes[:, 1, 1] += right_eye_widths * 0.6
        right_eye_boxes = right_eye_boxes.round().astype(int)
        right_eye_boxes[:, 1, :] += 1

        left_eye_patches = []
        right_eye_patches = []
        left_border = min(left_eye_boxes[..., 0].min(), right_eye_boxes[..., 0].min())
        top_border = min(left_eye_boxes[..., 1].min(), right_eye_boxes[..., 1].min())
        right_border = max(left_eye_boxes[..., 0].max(), right_eye_boxes[..., 0].max())
        bottom_border = max(left_eye_boxes[..., 1].max(), right_eye_boxes[..., 1].max())
        paddings = [max(0, -left_border), max(0, -top_border),
                    max(0, right_border - frame.shape[1]),
                    max(0, bottom_border - frame.shape[0])]
        frame = cv2.copyMakeBorder(frame, paddings[1], paddings[3], paddings[0], paddings[2],
                                   cv2.BORDER_CONSTANT, value=0)
        left_eye_boxes[..., 0] += paddings[0]
        left_eye_boxes[..., 1] += paddings[1]
        right_eye_boxes[..., 0] += paddings[0]
        right_eye_boxes[..., 1] += paddings[1]
        for left_box, right_box in zip(left_eye_boxes, right_eye_boxes):
            left_eye_patches.append(GazeEstimatorBase.input_from_image(cv2.resize(
                frame[left_box[0, 1]: left_box[1, 1], left_box[0, 0]: left_box[1, 0]],
                (60, 36), interpolation=cv2.INTER_CUBIC)))
            right_eye_patches.append(GazeEstimatorBase.input_from_image(cv2.resize(
                frame[right_box[0, 1]: right_box[1, 1], right_box[0, 0]: right_box[1, 0]],
                (60, 36), interpolation=cv2.INTER_CUBIC)))

        return np.array(left_eye_patches), np.array(right_eye_patches)
    else:
        return np.empty(shape=(0, 36, 60, 3), dtype=np.float32), np.empty(shape=(0, 36, 60, 3), dtype=np.float32)


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video path or webcam index (default=0)', default=0)
    parser.add_argument('--output', '-o', help='Output file path', default=None)
    parser.add_argument('--fourcc', '-f', help='FourCC of the output video (default=mp4v)',
                        type=str, default='mp4v')
    parser.add_argument('--no-display', '-n', help='No display if processing a video file',
                        action='store_true', default=False)

    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--device', '-d', default='cuda:0',
                        help='Device to be used by all models (default=cuda:0')

    parser.add_argument('--detection-threshold', '-dt', type=float, default=0.8,
                        help='Confidence threshold for face detection (default=0.8)')
    parser.add_argument('--detection-frame-size', '-ds', type=int, default=640,
                        help='Target frame size for face detection (default=640)')

    parser.add_argument('--alignment-threshold', '-at', type=float, default=0.2,
                        help='Score threshold used when visualising detected landmarks (default=0.2)')
    parser.add_argument('--alignment-weights', '-aw', default='2dfan2_alt',
                        help='Weights to be loaded for face alignment, can be either 2DFAN2, 2DFAN4, ' +
                             'or 2DFAN2_ALT (default=2DFAN2_ALT)')
    parser.add_argument('--alignment-alternative-pth', '-ap', default=None,
                        help='Alternative pth file to be loaded for face alignment')
    parser.add_argument('--alignment-alternative-landmarks', '-al', default=None,
                        help='Alternative number of landmarks to detect')

    rt_gene_model_dir = os.path.realpath(os.path.join(os.path.dirname(rt_gene_loc), '..', 'model_nets'))
    rt_gene_default_model_path = os.path.join(rt_gene_model_dir, 'Model_allsubjects1.h5')
    parser.add_argument('--gaze-detection-model', '-gm', default=rt_gene_default_model_path,
                        help=f'Model file to be loaded for gaze detection (default="{rt_gene_default_model_path}")')
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark

    vid = None
    out_vid = None
    has_window = False
    try:
        # Create the face detector
        face_detector = S3FDPredictor(device=args.device, threshold=args.detection_threshold)
        print('Face detector created.')

        # Create the landmark detector
        if args.alignment_weights is None:
            fa_model = FANPredictor.get_model()
        else:
            fa_model = FANPredictor.get_model(args.alignment_weights)
        if args.alignment_alternative_pth is not None:
            fa_model.weights = args.alignment_alternative_pth
        if args.alignment_alternative_landmarks is not None:
            fa_model.config.num_landmarks = int(args.alignment_alternative_landmarks)
        landmark_detector = FANPredictor(device=args.device, model=fa_model)
        print('Landmark detector created.')

        # Instantiate 3DDFA
        tddfa = TDDFAPredictor(device=args.device)
        print('3DDFA initialised.')

        # Create the gaze estimator
        gaze_estimator = GazeEstimatorBase(device_id_gaze=args.device.lower().replace('cuda', 'gpu'),
                                           model_files=args.gaze_detection_model)
        print('Gaze estimator created.')

        # Open the input video
        using_webcam = not os.path.exists(args.input)
        vid = cv2.VideoCapture(int(args.input) if using_webcam else args.input)
        assert vid.isOpened()
        if using_webcam:
            print(f'Webcam #{int(args.input)} opened.')
        else:
            print(f'Input video "{args.input}" opened.')

        # Open the output video (if a path is given)
        if args.output is not None:
            out_vid = cv2.VideoWriter(args.output, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc(*args.fourcc))
            assert out_vid.isOpened()

        # Process the frames
        frame_number = 0
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        print('Processing started, press \'Q\' to quit.')
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect faces
                start_time = time.time()
                frame_short_size = min(frame.shape[:2])
                if args.detection_frame_size < frame_short_size:
                    fd_scale = args.detection_frame_size / frame_short_size
                    fd_frame = cv2.resize(frame, (0, 0), fx=fd_scale, fy=fd_scale)
                else:
                    fd_scale = 1.0
                    fd_frame = frame
                faces = face_detector(fd_frame, rgb=False)
                if faces.shape[0] > 0:
                    faces[:, :4] /= fd_scale
                    faces[:, 5:] /= fd_scale
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Face alignment
                start_time = current_time
                landmarks, scores = landmark_detector(frame, faces, rgb=False)
                current_time = time.time()
                elapsed_time2 = current_time - start_time

                # Filter valid faces using landmark scores
                validities = scores.min(axis=1) >= args.alignment_threshold

                # Run 3DDFA on the valid faces
                start_time = current_time
                tddfa_results = TDDFAPredictor.decode(tddfa(frame, landmarks[validities], rgb=False))
                current_time = time.time()
                elapsed_time3 = current_time - start_time

                # Gaze detection
                start_time = current_time
                left_eye_patches, right_eye_patches = crop_eye_patches(frame, landmarks[validities])
                if left_eye_patches.shape[0] > 0 and (
                        left_eye_patches.shape[0] == right_eye_patches.shape[0] == len(tddfa_results)):
                    head_poses = np.array([[tr['face_pose']['pitch'], tr['face_pose']['yaw']]
                                           for tr in tddfa_results])
                    eye_gazes = gaze_estimator.estimate_gaze_twoeyes(left_eye_patches, right_eye_patches, head_poses)
                else:
                    head_poses = np.empty(shape=(0, 2), dtype=np.float32)
                    eye_gazes = np.empty(shape=(0, 2), dtype=np.float32)
                current_time = time.time()
                elapsed_time4 = current_time - start_time

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time * 1000.0:.04f} + ' +
                      f'{elapsed_time2 * 1000.0:.04f} + {elapsed_time3 * 1000.0:.04f} + ' +
                      f'{elapsed_time4 * 1000.0:.04f} ms: {len(faces)} faces analysed.')

                # Rendering
                for face, lm, sc, vf in zip(faces, landmarks, scores, validities):
                    bbox = face[:4].astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  color=(0, 0, 255) if vf else (128, 128, 128), thickness=2)
                    plot_landmarks(frame, lm, sc, threshold=args.alignment_threshold,
                                   pts_colour=(0, 0, 255) if vf else (128, 128, 128),
                                   line_colour=(0, 255, 0) if vf else (192, 192, 192))

                # Plot eye gazes
                for lmks, head_pose, eye_gaze in zip(landmarks[validities], head_poses, eye_gazes):
                    nose_tip = lmks[30]
                    end_point = nose_tip + 80.0 * np.array([math.cos(head_pose[0]) * math.sin(head_pose[1]),
                                                            math.sin(head_pose[0])])
                    cv2.line(frame, tuple(nose_tip.astype(int).tolist()), tuple(end_point.astype(int).tolist()),
                             (0, 0, 255), 2, cv2.LINE_AA)
                    gaze_displacement = 80.0 * np.array([math.cos(eye_gaze[0]) * math.sin(eye_gaze[1]),
                                                         math.sin(eye_gaze[0])])
                    left_eye_center = (lmks[36] + lmks[39]) / 2.0
                    end_point = left_eye_center - gaze_displacement
                    cv2.line(frame, tuple(left_eye_center.astype(int).tolist()),
                             tuple(end_point.astype(int).tolist()),
                             (0, 255, 0), 2, cv2.LINE_AA)
                    right_eye_center = (lmks[42] + lmks[45]) / 2.0
                    end_point = right_eye_center - gaze_displacement
                    cv2.line(frame, tuple(right_eye_center.astype(int).tolist()),
                             tuple(end_point.astype(int).tolist()),
                             (0, 255, 0), 2, cv2.LINE_AA)

                # Write the frame to output video (if recording)
                if out_vid is not None:
                    out_vid.write(frame)

                # Display the frame
                if using_webcam or not args.no_display:
                    has_window = True
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) % 2 ** 16
                    if key == ord('q') or key == ord('Q'):
                        print('\'Q\' pressed, we are done here.')
                        break
                frame_number += 1
    finally:
        if has_window:
            cv2.destroyAllWindows()
        if out_vid is not None:
            out_vid.release()
        if vid is not None:
            vid.release()
        print('All done.')


if __name__ == '__main__':
    main()
