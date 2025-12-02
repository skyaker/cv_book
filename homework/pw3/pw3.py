import cv2
import numpy as np
import sys
import torch

IMG1_NAME = 'object_img_1.png'
IMG2_NAME = 'object_img_2.png'
VID1_NAME = 'object_vid_1.avi'
VID2_NAME = 'object_vid_2.avi'

VIDEO_SCALE = 0.5
VIDEO_SKIP = 3

repo_path = 'SuperPointPretrainedNetwork' 
if repo_path not in sys.path:
    sys.path.append(repo_path)

try:
    from demo_superpoint import SuperPointFrontend
except ImportError:
    print(f"no demo_superpoint.py")
    sys.exit(1)

weights_path = 'superpoint_v1.pth'
fe = SuperPointFrontend(weights_path=weights_path,
                        nms_dist=4,
                        conf_thresh=0.015,
                        nn_thresh=0.7,
                        cuda=torch.cuda.is_available())


def run_superpoint(image, frontend, scale=1.0):
    if scale != 1.0:
        h, w = image.shape[:2]
        new_dim = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_float = img_gray.astype(np.float32) / 255.0

    pts, desc, _ = frontend.run(img_float)

    pts = pts.T 
    desc = desc.T
    
    kps = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts]
    
    return image, kps, desc

def match_and_draw(img1, kp1, desc1, img2, kp2, desc2):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return np.hstack((img1, img2))

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(img1, kp1, img2, kp2, 
                             matches[:50], None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result

img1_orig = cv2.imread(IMG1_NAME)
img2_orig = cv2.imread(IMG2_NAME)

if img1_orig is None or img2_orig is None:
    print("err: no images")
else:
    res_img1, kp1, desc1 = run_superpoint(img1_orig, fe, scale=1.0)
    res_img2, kp2, desc2 = run_superpoint(img2_orig, fe, scale=1.0)

    final_matches_img = match_and_draw(res_img1, kp1, desc1, res_img2, kp2, desc2)

    cv2.imwrite("Test_Matches.png", final_matches_img)


cap1 = cv2.VideoCapture(VID1_NAME)
cap2 = cv2.VideoCapture(VID2_NAME)
out = None
last_frame_result = None

if not cap1.isOpened() or not cap2.isOpened():
    print("err: no video")
else:
    frame_count = 0
    processed_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        if frame_count % VIDEO_SKIP != 0:
            frame_count += 1
            continue

        img1_small, kp1, desc1 = run_superpoint(frame1, fe, scale=VIDEO_SCALE)
        img2_small, kp2, desc2 = run_superpoint(frame2, fe, scale=VIDEO_SCALE)

        result_frame = match_and_draw(img1_small, kp1, desc1, img2_small, kp2, desc2)
        last_frame_result = result_frame

        if out is None:
            h, w = result_frame.shape[:2]
            out = cv2.VideoWriter('matches_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))

        out.write(result_frame)
        
        cv2.imshow("Video Matches", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        processed_count += 1

    cap1.release()
    cap2.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if last_frame_result is not None:
        cv2.imwrite("video_final_frame.png", last_frame_result)
