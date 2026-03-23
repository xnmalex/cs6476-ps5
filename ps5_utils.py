import cv2
import os
import numpy as np

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0


def run_kalman_filter(filter_class,
                      imgs_dir,
                      noise,
                      sensor,
                      save_frames={},
                      template_loc=None,
                      Q=0.1 * np.eye(4),
                      R=0.1 * np.eye(2)):
    kf = filter_class(template_loc['x'], template_loc['y'], Q, R)

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0

def run_multi_particle_filter(filter_class, imgs_dir, save_frames):
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

def run_single_particle_filter_follow(filter_class, imgs_dir, save_frames):
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