"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    save_frames = {
        28: os.path.join(output_dir, 'ps5-5-a-1.png'),
        55: os.path.join(output_dir, 'ps5-5-a-2.png'),
        70: os.path.join(output_dir, 'ps5-5-a-3.png')
    }

    filter_class = ps5.MDParticleFilter   # or ParticleFilter
    imgs_dir =  os.path.join(input_dir, "TUD-Campus")

     # --- HOG detector ---
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # --- Load images ---
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    trackers = []
    frame_num = 0

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))
        frame_out = frame.copy()

        # --- Detect people ---
        (rects, weights) = hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )

        detections = []
        for (x, y, w, h) in rects:
            cx = x + w // 2
            cy = y + h // 2
            detections.append((cx, cy, w, h))

        used = set()

        # --- MATCH + UPDATE existing trackers ---
        for tracker in trackers:

            best_dist = float('inf')
            best_det = None
            best_idx = -1

            for i, (cx, cy, w, h) in enumerate(detections):
                if i in used:
                    continue

                # particle filter mean position
                particles = tracker["pf"].particles
                weights_pf = tracker["pf"].weights

                x_mean = np.sum(particles[:, 0] * weights_pf)
                y_mean = np.sum(particles[:, 1] * weights_pf)

                dist = np.linalg.norm(np.array([x_mean, y_mean]) - np.array([cx, cy]))

                if dist < best_dist:
                    best_dist = dist
                    best_det = (cx, cy, w, h)
                    best_idx = i

            if best_det is not None and best_dist < 60:
                tracker["pf"].process(frame)
                tracker["bbox"] = (best_det[2], best_det[3])
                used.add(best_idx)
            else:
                tracker["pf"].process(frame)

        # --- CREATE new trackers ---
        for i, (cx, cy, w, h) in enumerate(detections):
            if i not in used:

                x1 = int(cx - w // 2)
                y1 = int(cy - h // 2)
                x2 = x1 + w
                y2 = y1 + h

                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                template = frame[y1:y2, x1:x2]

                if template.size == 0:
                    continue

                pf = filter_class(
                    frame,
                    template,
                    num_particles=200,
                    sigma_exp=10,
                    sigma_dyn=10,
                    alpha=0.05,
                    template_coords={'x': x1, 'y': y1, 'w': w, 'h': h}
                )

                trackers.append({
                    "pf": pf,
                    "bbox": (w, h)
                })

        # --- RENDER ---
        for tracker in trackers:
            pf = tracker["pf"]

            particles = pf.particles
            weights_pf = pf.weights

            x_mean = np.sum(particles[:, 0] * weights_pf)
            y_mean = np.sum(particles[:, 1] * weights_pf)

            w, h = tracker["bbox"]

            x1 = int(x_mean - w // 2)
            y1 = int(y_mean - h // 2)
            x2 = int(x_mean + w // 2)
            y2 = int(y_mean + h // 2)

            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- SAVE ---
        if frame_num in save_frames:
            cv2.imwrite(save_frames[frame_num], frame_out)

        frame_num += 1

        if frame_num % 10 == 0:
            print(f'Processing frame {frame_num}')

    return 0

def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    save_frames = {
        59: os.path.join(output_dir, 'ps5-6-a-1.png'),
        159: os.path.join(output_dir, 'ps5-6-a-2.png'),
        185: os.path.join(output_dir, 'ps5-6-a-3.png')
    }

    filter_class = ps5.MDParticleFilter   # or ParticleFilter
    imgs_dir =  os.path.join(input_dir, "follow")

     # --- HOG detector ---
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # --- Load images ---
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    pf = None
    bbox = None   # (w, h)

    for frame_num, img in enumerate(imgs_list):

        frame = cv2.imread(os.path.join(imgs_dir, img))
        frame_out = frame.copy()

        # --- Initialize tracker on first frame ---
        if pf is None:
            (rects, weights) = hog.detectMultiScale(
                frame,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05
            )

            best_rect = None
            best_score = -1

            # Heuristic: choose detection in expected area
            # You may need to tune these bounds after checking 001.jpg
            for (x, y, w, h), wt in zip(rects, weights):
                cx = x + w // 2
                cy = y + h // 2

                if 100 < cx < 350 and 50 < cy < 300:
                    score = float(wt) + 0.001 * (w * h)
                    if score > best_score:
                        best_score = score
                        best_rect = (x, y, w, h)

            # fallback: strongest detection
            if best_rect is None and len(rects) > 0:
                best_rect = rects[np.argmax(weights)]

            if best_rect is None:
                print("Could not initialize tracker.")
                return 0

            x, y, w, h = best_rect

            # ensure valid crop
            img_h, img_w = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))

            template = frame[y:y+h, x:x+w]

            if template.size == 0:
                print("Invalid initialization template.")
                return 0

            pf = filter_class(
                frame,
                template,
                num_particles=300,
                sigma_exp=10,
                sigma_dyn=12,
                alpha=0.05,
                template_coords={'x': x, 'y': y, 'w': w, 'h': h}
            )

            bbox = (w, h)

        else:
            # --- Update tracker ---
            pf.process(frame)

            # --- Optional detector-based bbox refresh ---
            particles = pf.particles
            weights_pf = pf.weights

            x_mean = np.sum(particles[:, 0] * weights_pf)
            y_mean = np.sum(particles[:, 1] * weights_pf)

            (rects, weights) = hog.detectMultiScale(
                frame,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05
            )

            best_rect = None
            best_dist = float('inf')

            for (x, y, w, h) in rects:
                cx = x + w // 2
                cy = y + h // 2

                dist = np.linalg.norm(
                    np.array([x_mean, y_mean]) - np.array([cx, cy])
                )

                if dist < best_dist and dist < 80:
                    best_dist = dist
                    best_rect = (x, y, w, h)

            if best_rect is not None:
                bbox = (best_rect, best_rect)

        # --- Render particles ---
        particles = pf.particles
        weights_pf = pf.weights

        for p in particles:
            px = int(p[0])  # x-coordinate
            py = int(p[1])  # y-coordinate
            cv2.circle(frame_out, (px, py), 1, (0, 255, 0), -1)

        # --- Render estimated bounding box ---
        x_mean = np.sum(particles[:, 0] * weights_pf)
        y_mean = np.sum(particles[:, 1] * weights_pf)

        bbox_flat = np.array(bbox).ravel()  # ravel works even if it's nested
        if len(bbox_flat) >= 4:
            x, y, w, h = map(int, bbox_flat[:4])
        else:
            # fallback: guess first 2 are x,y, estimate w,h
            x, y = map(int, bbox_flat[:2])
            w, h = 50, 50  # default width/height if missing
      
        x1 = int(x_mean - w // 2)
        y1 = int(y_mean - h // 2)
        x2 = int(x_mean + w // 2)
        y2 = int(y_mean + h // 2)

        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- Save ---
        if frame_num in save_frames:
            cv2.imwrite(save_frames[frame_num], frame_out)

        if (frame_num + 1) % 10 == 0:
            print(f'Processing frame {frame_num + 1}')

    return 0


if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    # part_4()
    part_5()
    part_6()
