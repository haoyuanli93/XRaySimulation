"""
Anyway, in the end, I think it is better for me to just
create a module to mimic the motion of the motors I used
for the SD table

To be compatible with all other modules, we use unit
1. um
2. fs
3. rad

"""

import numpy as np
from XRaySimulation import util


class LinearMotor:
    def __init__(self,
                 upperLim=25000,
                 lowerLim=-25000,
                 res=5,
                 backClash=100,
                 feedback_noise_level=1,
                 speed=1 * 1000 / 1e12,
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backClash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed: the speed in um / ps
        """
        self.motor_type = "Linear"

        # The actual state of the motor
        self.location = np.zeros(3, dtype=np.float64)
        self.positive_direction = np.zeros(3, dtype=np.float64)
        self.positive_direction[0] = 1.0
        self.limits = np.zeros(2)
        self.limits[0] = lowerLim
        self.limits[1] = upperLim
        self.resolution = res
        self.backClash = backClash
        self.speed = speed

        # TODO: Define the position direction variable

        # Define all the parameters for noises
        self.feedback_noise_level = feedback_noise_level
        self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level

        # The linear position of the motor assuming that there is no error.
        self.location_feedback = 0.0

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 4cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.boundary = np.array([np.array([0, -2e4, -5e4]),
                                  np.array([0, -2e4, 5e4]),
                                  np.array([0, 2e4, 5e4]),
                                  np.array([0, 2e0, -5e4]),
                                  np.array([0, -2e40, -5e4]),
                                  ])

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.location += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.location = np.ascontiguousarray(rot_mat.dot(self.location))
        self.positive_direction = np.ascontiguousarray(rot_mat.dot(self.positive_direction))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.location_feedback
            if delta * self.backClash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 2: if it is with in the limit, then change the physical position and change the motor status
                self.location = (self.location
                                 + self.positive_direction * (
                                         delta + self.resolution * (np.random.rand() - 0.5)))

                self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                self.location_feedback = self.location + self.feedback_noise

                print("Motor moved to {:.2f} um".format(self.location_feedback))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.location_feedback + self.backClash + delta):
                    self.location = (self.location
                                     + self.positive_direction * (self.backClash + delta +
                                                                  self.resolution * (np.random.rand() - 0.5)))
                    self.location = (self.location
                                     + self.positive_direction * (- self.backClash +
                                                                  self.resolution * (np.random.rand() - 0.5)))

                    self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                    self.location_feedback = self.location + self.feedback_noise
                    print("Motor moved to {:.2f} um".format(self.location_feedback))
                else:
                    print("The target location {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

        if getMotionTime:
            return 0

    def user_getPosition(self):
        return self.location_feedback

    def __check_limit(self, val):
        if (val >= self.limits[0]) and (val <= self.limits[1]):
            return True
        else:
            return False


class VerticalMotor:
    def __init__(self,
                 upperLim=25000,
                 lowerLim=-25000,
                 res=5,
                 backClash=100,
                 feedback_noise_level=1,
                 speed=1 * 1000 / 1e12,
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backClash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed: the speed in um / ps
        """
        self.motor_type = "Linear"

        # The actual state of the motor
        self.location = np.zeros(3, dtype=np.float64)
        self.positive_direction = np.zeros(3, dtype=np.float64)
        self.positive_direction[0] = 1.0
        self.limits = np.zeros(2)
        self.limits[0] = lowerLim
        self.limits[1] = upperLim
        self.resolution = res
        self.backClash = backClash
        self.speed = speed

        # TODO: Define the position direction variable

        # Define all the parameters for noises
        self.feedback_noise_level = feedback_noise_level
        self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level

        # The linear position of the motor assuming that there is no error.
        self.location_feedback = 0.0

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 4cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.boundary = np.array([np.array([0, -2e4, -5e4]),
                                  np.array([0, -2e4, 5e4]),
                                  np.array([0, 2e4, 5e4]),
                                  np.array([0, 2e0, -5e4]),
                                  np.array([0, -2e40, -5e4]),
                                  ])

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.location += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.location = np.ascontiguousarray(rot_mat.dot(self.location))
        self.positive_direction = np.ascontiguousarray(rot_mat.dot(self.positive_direction))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.location_feedback
            if delta * self.backClash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 2: if it is with in the limit, then change the physical position and change the motor status
                self.location = (self.location
                                 + self.positive_direction * (
                                         delta + self.resolution * (np.random.rand() - 0.5)))

                self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                self.location_feedback = self.location + self.feedback_noise

                print("Motor moved to {:.2f} um".format(self.location_feedback))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.location_feedback + self.backClash + delta):
                    self.location = (self.location
                                     + self.positive_direction * (self.backClash + delta +
                                                                  self.resolution * (np.random.rand() - 0.5)))
                    self.location = (self.location
                                     + self.positive_direction * (- self.backClash +
                                                                  self.resolution * (np.random.rand() - 0.5)))

                    self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                    self.location_feedback = self.location + self.feedback_noise
                    print("Motor moved to {:.2f} um".format(self.location_feedback))
                else:
                    print("The target location {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

        if getMotionTime:
            return 0

    def user_getPosition(self):
        return self.location_feedback

    def __check_limit(self, val):
        if (val >= self.limits[0]) and (val <= self.limits[1]):
            return True
        else:
            return False


class RotationMotor:
    def __init__(self,
                 upperLim=np.deg2rad(180),
                 lowerLim=-np.deg2rad(-180),
                 res=1,
                 backClash=100,
                 feedback_noise_level=1,
                 speed_mm_per_s=1 * 1000 / 1e12,
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backClash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed_mm_per_s: the speed in um / ps
        """
        self.motor_type = "Rotation"

        # The actual state of the motor
        self.deg0direction = np.zeros(3, dtype=np.float64)
        self.deg0direction[1] = 1.0
        self.rotation_axis = np.zeros(3, dtype=np.float64)
        self.rotation_axis[0] = 1.0
        self.rotation_center = np.zeros(3, dtype=np.float64)

        self.limits = np.zeros(2)
        self.limits[0] = lowerLim
        self.limits[1] = upperLim
        self.resolution = res
        self.backClash = backClash
        self.speed = speed_mm_per_s

        # Define all the parameters for noises
        self.feedback_noise_level = feedback_noise_level
        self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level

        # The angle position of the motor assuming that there is no error.
        self.location_feedback = 0.0

        # The internal track of the angle information
        self.location = 0.0

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 10cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.boundary = self.boundary = np.array([np.array([0, -5e4, -5e4]),
                                                  np.array([0, -5e4, 5e4]),
                                                  np.array([0, 5e4, 5e4]),
                                                  np.array([0, 5e0, -5e4]),
                                                  np.array([0, -5e40, -5e4]),
                                                  ])

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.rotation_center += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))
        self.rotation_axis = np.ascontiguousarray(rot_mat.dot(self.rotation_axis))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.location_feedback

            if delta * self.backClash <= 0:  # Move to the opposite direction as the back-clash direction

                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + delta + self.resolution * (np.random.rand() - 0.5),
                    axis=self.rotation_axis)

                # Update the zero deg direction
                self.deg0direction = np.dot(rotMat, self.deg0direction)

                # Update the internal record of the current status
                self.location += delta

                # Step 2: if it is with in the limit, then change the physical position and change the motor status
                self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                self.location_feedback = self.location + self.feedback_noise

                print("Motor moved to {:.2f} um".format(self.location_feedback))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.location_feedback + self.backClash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.backClash + delta + self.resolution * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)

                    self.deg0direction = np.dot(rotMat1, self.deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.backClash + self.resolution * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)
                    self.deg0direction = np.dot(rotMat2, self.deg0direction)

                    self.location += delta
                    self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                    self.location_feedback = self.location + self.feedback_noise
                    print("Motor moved to {:.2f} um".format(self.location_feedback))
                else:
                    print("The target location {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

        if getMotionTime:
            return 0

    def user_getPosition(self):
        return self.location_feedback

    def __check_limit(self, val):
        if (val >= self.limits[0]) and (val <= self.limits[1]):
            return True
        else:
            return False


class SwivelMotor:
    def __init__(self,
                 upperLim=np.deg2rad(10),
                 lowerLim=-np.deg2rad(10),
                 res=1,
                 backClash=100,
                 feedback_noise_level=1,
                 speed=1 * 1000 / 1e12,
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backClash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed: the speed in deg / s
        """
        self.motor_type = "Swivel"

        # The actual state of the motor
        self.deg0direction = np.zeros(3, dtype=np.float64)
        self.deg0direction[0] = 1.0
        self.rotation_axis = np.zeros(3, dtype=np.float64)
        self.rotation_axis[1] = 1.0
        self.rotation_center = np.zeros(3, dtype=np.float64)

        self.limits = np.zeros(2)
        self.limits[0] = lowerLim
        self.limits[1] = upperLim
        self.resolution = res
        self.backClash = backClash
        self.speed = speed

        # Define all the parameters for noises
        self.location = 0.0
        self.feedback_noise_level = feedback_noise_level
        self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level

        # The linear position of the motor assuming that there is no error.
        self.location_feedback = 0.0

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 10cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.boundary = self.boundary = np.array([np.array([0, -5e4, -5e4]),
                                                  np.array([0, -5e4, 5e4]),
                                                  np.array([0, 5e4, 5e4]),
                                                  np.array([0, 5e0, -5e4]),
                                                  np.array([0, -5e40, -5e4]),
                                                  ])

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.rotation_center += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

    def user_getPosition(self):
        return self.location_feedback

    def __check_limit(self, val):
        if (val >= self.limits[0]) and (val <= self.limits[1]):
            return True
        else:
            return False

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.location_feedback

            if delta * self.backClash <= 0:  # Move to the opposite direction as the back-clash direction

                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + delta + self.resolution * (np.random.rand() - 0.5),
                    axis=self.rotation_axis)

                # Update the zero deg direction
                self.deg0direction = np.dot(rotMat, self.deg0direction)

                # Update the internal record of the current status
                self.location += delta

                # Step 2: if it is with in the limit, then change the physical position and change the motor status
                self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                self.location_feedback = self.location + self.feedback_noise

                print("Motor moved to {:.2f} um".format(self.location_feedback))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.location_feedback + self.backClash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.backClash + delta + self.resolution * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)

                    self.deg0direction = np.dot(rotMat1, self.deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.backClash + self.resolution * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)
                    self.deg0direction = np.dot(rotMat2, self.deg0direction)

                    self.location += delta
                    self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level
                    self.location_feedback = self.location + self.feedback_noise
                    print("Motor moved to {:.2f} um".format(self.location_feedback))
                else:
                    print("The target location {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")

        if getMotionTime:
            return 0


class CrystalTower_1236:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 channelCut):
        self.x = LinearMotor()
        self.y = VerticalMotor()
        self.th = RotationMotor()
        self.chi = SwivelMotor()
        self.optics = channelCut

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.positive_direction * (target - self.x.location_feedback)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.chi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

        self.colors = ['k', 'brown', 'yellow', 'purple', 'red']

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.positive_direction * (target - self.y.location_feedback)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.chi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def th_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.th.positive_direction * (target - self.th.location_feedback)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.th.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.th.user_move_abs(target=target, getMotionTime=True)
        self.chi.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center, include_boundary=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center, include_boundary=True)

    def chi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.chi.positive_direction * (target - self.chi.location_feedback)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.chi.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.chi.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.chi.rotation_center, include_boundary=True)

    def plot_motors(self, ax):
        # Plot motors and crystals one by one
        ax.plot(self.x.boundary[:, 2], self.x.boundary[:, 1],
                linestyle='--', linewidth=1, label="x", color=self.colors[0])
        ax.plot(self.y.boundary[:, 2], self.y.boundary[:, 1],
                linestyle='--', linewidth=1, label="y", color=self.colors[1])
        ax.plot(self.th.boundary[:, 2], self.th.boundary[:, 1],
                linestyle='--', linewidth=1, label="th", color=self.colors[2])
        ax.plot(self.chi.boundary[:, 2], self.chi.boundary[:, 1],
                linestyle='--', linewidth=1, label="chi", color=self.colors[3])
        ax.plot(self.optics.boundary[:, 2], self.optics.boundary[:, 1],
                linestyle='-', linewidth=3, label="crystal", color=self.colors[4])
