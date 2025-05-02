import time
import cv2
import numpy as np
import os


def downsample_video(video_path):
    """
    Downsample the video to 1 fps while retaining the first and last frames, and return a list of downsampled frames.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Fail to read video: ", video_path)
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    downsampled_frames = []
    interval = int(fps)
    frame_indices_to_keep = list(range(0, frame_count, interval))
    if (frame_count - 1) not in list(range(0, frame_count, interval)):
        frame_indices_to_keep.append(frame_count - 1)
    success, frame = cap.read()
    index = 0
    while success:
        if index in frame_indices_to_keep:
            downsampled_frames.append(frame)
        success, frame = cap.read()
        index += 1
    cap.release()
    return downsampled_frames


def extract_keyframes_from_video(video_path, output_dir, max_frame=32, min_frame=4):
    """
    Extract keyframes from the given video file and save them to the specified output directory.

    Parameters：
    - video_path: Path of video file
    - max_frame: Max number of frames
    - min_frame: Min number of frames
    - output_dir: save path
    """

    # Check if the output directory exists, create if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Downsampling the video to 1fps
    frames = downsample_video(video_path)
    if not frames or len(frames) == 0:
        print(f"Video {video_path} downsampling failed")
        return
    frame_count = len(frames)

    # Obtain the idx of keyframes
    keyframe_indices = extract_keyframes(frames, 0.5)
    print("关键帧索引：", keyframe_indices)
    print("关键帧数量：", len(keyframe_indices))

    # Ensure the keyframe number is in [min_frame, max_frame]
    if len(keyframe_indices) < min_frame:
        keyframe_indices = [i for i in range(frame_count)]
        keyframe_indices = constrained_sampling(keyframe_indices, min_frame)
    elif len(keyframe_indices) > max_frame:
        keyframe_indices = constrained_sampling(keyframe_indices, max_frame)

    # Save the keyframes
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)
    frame_height, frame_width, _ = frames[0].shape
    fps = 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for idx in keyframe_indices:
        frame = frames[idx]
        out.write(frame)
    out.release()
    print(f"Keyframes save to {output_path}")


def extract_keyframes(frames, x):
    """
    Extract keyframes from the frame list based on the coincidence ratio.

    Parameters：
    - frames: A list containing video frames, with each frame being a numpy array
    - x: Overlap ratio threshold, between 0 and 1

    Return：
    - keyframe_indices: Index list of keyframes
    """
    keyframe_indices = [0]  # Keep the first frame
    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize feature detectors and matchers
    orb = cv2.ORB_create()
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for t in range(1, len(frames)):
        curr_frame = frames[t]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

        # Check if the descriptor is empty
        if prev_des is None or curr_des is None:
            # Unable to match, keep the current frame
            keyframe_indices.append(t)
            prev_frame = curr_frame
            prev_gray = curr_gray
            prev_kp = curr_kp
            prev_des = curr_des
            continue

        # feature matching
        matches = bf.match(prev_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # When there are enough matching points, estimate the transformation
        if len(matches) > 10:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate transformation matrix (e.g. perspective transformation)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Calculate the proportion of overlapping areas
                h, w = prev_gray.shape
                corners_prev = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                corners_curr_in_prev = cv2.perspectiveTransform(corners_prev, M)

                # Calculate the overlapping area of two polygons
                overlap_ratio = calculate_overlap_ratio(corners_prev, corners_curr_in_prev, w, h)

                # Determine whether to retain the current frame based on the overlap ratio
                if overlap_ratio < x:
                    keyframe_indices.append(t)
                    # Update the information of the former frame
                    prev_frame = curr_frame
                    prev_gray = curr_gray
                    prev_kp = curr_kp
                    prev_des = curr_des
            else:
                # Unable to calculate transformation, keep current frame
                keyframe_indices.append(t)

                # Update the information of the former frame
                prev_frame = curr_frame
                prev_gray = curr_gray
                prev_kp = curr_kp
                prev_des = curr_des
        else:
            # Too few matching points, keep the current frame
            keyframe_indices.append(t)

            # Update the information of the previous frame
            prev_frame = curr_frame
            prev_gray = curr_gray
            prev_kp = curr_kp
            prev_des = curr_des

    if keyframe_indices[-1] != len(frames) - 1:
        keyframe_indices.append(len(frames) - 1)
    return keyframe_indices

def calculate_overlap_ratio(corners1, corners2, width, height):
    """
    Calculate the overlap ratio of two polygons (image regions)

    Parameters：
    - corners1: The corner coordinates of the first polygon
    - corners2: The corner coordinates of the second polygon
    - width, height: Width and height of the image

    Return：
    - overlap_ratio: The ratio of overlapping area to total image area
    """
    # Convert coordinates into a format suitable for calculation
    poly1 = np.array([c[0] for c in corners1], dtype=np.float32)
    poly2 = np.array([c[0] for c in corners2], dtype=np.float32)

    # Convert polygon coordinates to a format suitable for cv2.fillPoly
    poly1_int = np.int32([poly1])
    poly2_int = np.int32([poly2])

    # Create black and white images
    img1 = np.zeros((height, width), dtype=np.uint8)
    img2 = np.zeros((height, width), dtype=np.uint8)

    # draw a polygon
    cv2.fillPoly(img1, poly1_int, 1)
    cv2.fillPoly(img2, poly2_int, 1)

    # Calculate overlapping areas
    intersection = cv2.bitwise_and(img1, img2)
    overlap_area = np.sum(intersection)
    total_area = width * height
    overlap_ratio = overlap_area / total_area
    return overlap_ratio


def constrained_sampling(lst, m):
    """
    Sample the list based on length constraints.
    """
    n = len(lst)
    if n <= m:
        # If the length of the input list is less than or equal to the constraint length, return the original list directly
        return lst
    else:
        # Keep the first and last elements
        x = lst[0]
        y = lst[-1]

        # Uniformly sample m-2 elements in the middle section
        num_samples = m - 2
        step = (n - 2) / num_samples
        sampled_indices = [int(round(i * step)) + 1 for i in range(num_samples)]
        sampled_elements = [lst[idx] for idx in sampled_indices]

        if sampled_elements[-1] == y:
            print('Error')

        # Return to new list, keep order
        return [x] + sampled_elements + [y]