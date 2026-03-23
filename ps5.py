"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import numpy as np

from ps5_utils import run_kalman_filter, run_particle_filter, run_multi_particle_filter, run_single_particle_filter_follow

# I/O directories
input_dir = "input"
output_dir = "output"



# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # State: position and velocity
        self.state = np.array([init_x, init_y, 0., 0.], dtype=float)

        # Time step
        dt = 1.0

        # State transition matrix
        self.F = np.array([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])


        # Measurement matrix: only x and y are observed
        self.H = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])

        # Covariance matrix
        self.P = np.eye(4)

        # Noise matrices
        self.Q = Q
        self.R = R

    def predict(self):
        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, meas_x, meas_y):
        z = np.array([meas_x, meas_y], dtype=float)

        # Innovation / residual
        y = z - (self.H @ self.state)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame

        h, w = frame.shape[:2]
        
        self.particles = np.zeros((self.num_particles, 2))  # Initialize your particles array. Read the docstring.
        self.particles[:, 0] = np.random.uniform(0, w, self.num_particles)  # x
        self.particles[:, 1] = np.random.uniform(0, h, self.num_particles)  # y

        self.weights = np.ones(self.num_particles) / self.num_particles  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.


    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        if template.shape != frame_cutout.shape:
            return 0.0

        # Mean squared error
        mse = np.mean((template.astype(float) - frame_cutout.astype(float)) ** 2)

        # Convert to likelihood
        return np.exp(-mse / (2 * (self.sigma_exp ** 2)))

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        weights = np.nan_to_num(self.weights)
        weights_sum = np.sum(weights)

        if weights_sum == 0:
            weights = np.ones(self.num_particles) / self.num_particles
        else:
            weights = weights / weights_sum

        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=weights
        )

        return self.particles[indices]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.frame = frame
        h, w = frame.shape[:2]
        th, tw = self.template.shape[:2]

        # --- 1. Predict (add motion noise) ---
        noise = np.random.normal(0, self.sigma_dyn, self.particles.shape)
        self.particles += noise

        # Clip to image bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)

        # --- 2. Measurement (compute weights) ---
        new_weights = np.zeros(self.num_particles)

        for i, (x, y) in enumerate(self.particles):
            x, y = int(x), int(y)

            x1 = int(x - tw // 2)
            y1 = int(y - th // 2)
            x2 = x1 + tw
            y2 = y1 + th

            # Check bounds
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                new_weights[i] = 0
                continue

            patch = frame[y1:y2, x1:x2]
            new_weights[i] = self.get_error_metric(self.template, patch)

            # Normalize weights
            new_weights = np.nan_to_num(new_weights)

            total = np.sum(new_weights)

            if total == 0:
                self.weights = np.ones(self.num_particles) / self.num_particles
            else:
                self.weights = new_weights / total

            # FINAL safety normalization
            self.weights = self.weights / np.sum(self.weights)

            # --- 3. Resample ---
            self.particles = self.resample_particles()

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        th, tw = self.template.shape[:2]

        # Weighted mean
        x_mean = np.sum(self.particles[:, 0] * self.weights)
        y_mean = np.sum(self.particles[:, 1] * self.weights)

        # Draw particles
        for p in self.particles:
            x, y = p[0], p[1]
            cv2.circle(frame_in, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Draw bounding box
        x1 = int(x_mean - tw // 2)
        y1 = int(y_mean - th // 2)
        x2 = x1 + tw
        y2 = y1 + th

        cv2.rectangle(frame_in, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw spread (uncertainty)
        dist = np.sqrt((self.particles[:, 0] - x_mean) ** 2 +
                    (self.particles[:, 1] - y_mean) ** 2)

        radius = int(np.sum(dist * self.weights))
        


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha', 0.1)  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        self.frame = frame
        h, w = frame.shape[:2]
        th, tw = self.template.shape[:2]

        # --- 1. Predict (same as base PF) ---
        noise = np.random.normal(0, self.sigma_dyn, self.particles.shape)
        self.particles += noise

        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)

        # --- 2. Measurement ---
        new_weights = np.zeros(self.num_particles)

        best_idx = 0
        best_weight = -1

        for i, (x, y) in enumerate(self.particles):
            x, y = int(x), int(y)

            x1 = int(x - tw // 2)
            y1 = int(y - th // 2)
            x2 = x1 + tw
            y2 = y1 + th

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                new_weights[i] = 0
                continue

            patch = frame[y1:y2, x1:x2]
            weight = self.get_error_metric(self.template, patch)
            new_weights[i] = weight

            # Track best particle
            if weight > best_weight:
                best_weight = weight
                best_idx = i

        # Normalize weights
        if np.sum(new_weights) == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = new_weights / np.sum(new_weights)

        # --- 3. Resample ---
        self.particles = self.resample_particles()

        # --- 4. Appearance update (THIS IS THE KEY PART) ---
        best_particle = self.particles[best_idx]
        x, y = int(best_particle[0]), int(best_particle[1])

        x1 = int(x - tw // 2)
        y1 = int(y - th // 2)
        x2 = x1 + tw
        y2 = y1 + th

        # Make sure it's valid
        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
            best_patch = frame[y1:y2, x1:x2]

            if best_patch.shape == self.template.shape:
                # Exponential moving average update
                self.template = (
                    (1 - self.alpha) * self.template.astype(float) +
                    self.alpha * best_patch.astype(float)
                ).astype(np.uint8)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.particles = np.zeros((self.num_particles, 4))

        h, w = frame.shape[:2]

        # դիր
        self.particles[:, 0] = np.random.uniform(0, w, self.num_particles)  # x
        self.particles[:, 1] = np.random.uniform(0, h, self.num_particles)  # y

        # velocity
        self.particles[:, 2] = np.random.normal(0, 1, self.num_particles)  # vx
        self.particles[:, 3] = np.random.normal(0, 1, self.num_particles)  # vy

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.frame = frame
        h, w = frame.shape[:2]
        th, tw = self.template.shape[:2]

        # --- 1. Predict (motion model with velocity) ---
        noise_pos = np.random.normal(0, self.sigma_dyn, (self.num_particles, 2))
        noise_vel = np.random.normal(0, 1, (self.num_particles, 2))

        # Update position using velocity
        self.particles[:, 0:2] += self.particles[:, 2:4] + noise_pos

        # Update velocity
        self.particles[:, 2:4] += noise_vel

        # Clip position
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)

        # --- 2. Measurement ---
        new_weights = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])

            x1 = int(x - tw // 2)
            y1 = int(y - th // 2)
            x2 = x1 + tw
            y2 = y1 + th

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                new_weights[i] = 0
                continue

            patch = frame[y1:y2, x1:x2]
            new_weights[i] = self.get_error_metric(self.template, patch)

        # Normalize weights (robust)
        new_weights = np.nan_to_num(new_weights)
        total = np.sum(new_weights)

        if total == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = new_weights / total

        self.weights = self.weights / np.sum(self.weights)

        # --- 3. Resample ---
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]

        # --- 4. Appearance update (same as AppearanceModelPF) ---
        best_idx = np.argmax(self.weights)
        x, y = int(self.particles[best_idx, 0]), int(self.particles[best_idx, 1])

        x1 = int(x - tw // 2)
        y1 = int(y - th // 2)
        x2 = x1 + tw
        y2 = y1 + th

        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
            best_patch = frame[y1:y2, x1:x2]

            if best_patch.shape == self.template.shape:
                self.template = (
                    (1 - self.alpha) * self.template.astype(float) +
                    self.alpha * best_patch.astype(float)
                ).astype(np.uint8)


def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 0  # Define the number of particles
    sigma_mse = 0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 0  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 0  # Define the number of particles
    sigma_mse = 0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 0  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.1  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_md = 100  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 100  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out

def part_5(obj_class, save_frames, input_folder):

    out = run_multi_particle_filter(
        obj_class,        # model class (PF / MD PF)
        input_folder,
        save_frames
    )

    return out

def part_6(obj_class, save_frames, input_folder):

    out = run_single_particle_filter_follow(
        obj_class,        # model class (PF / MD PF)
        input_folder,
        save_frames
    )

    return out
