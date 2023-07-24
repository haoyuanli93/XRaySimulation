"""
Anyway, in the end, I think it is better for me to just
create a module to mimic the motion of the motors I used
for the SD table

To be compatible with all other modules, we use unit
1. um
2. fs
3. rad

In my definition, the relation between the coordinate I used in this code and the
coordinate used by XPP is the following:

 my definition                  XPP                         physical
 x axis                         y axis                       vertical
 y axis                         x axis                       horizontal
 z axis                         z axis                       x-ray proportion direction
"""

import numpy as np
from XRaySimulation import util


class LinearMotor:
    def __init__(self,
                 upperLim=25000.0,
                 lowerLim=-25000.0,
                 res=5.0,
                 backlash=100.0,
                 feedback_noise_level=1.0,
                 speed_um_per_ps=1 * 1000 / 1e12,
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed_um_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        # With respect to the default positive direction, whether change the motor motion direction
        self.control_motor_type = "Linear"

        self.control_location = 0.0
        self.control_positive = 1.
        self.control_speed = speed_um_per_ps

        self.control_backlash = backlash

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim

        # ---------------------------------------------------
        # Define quantities of the physical system
        # ---------------------------------------------------
        self.physical_location = np.zeros(3, dtype=np.float64)
        self.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.physical_positive_direction[0] = 1.0

        # ---------------------------------------------------
        # Define quantities describing the error between the control system and the physical system
        # I call them device property (dp_...)
        # ---------------------------------------------------
        self.dp_resolution = res
        self.dp_feedback_noise = feedback_noise_level
        self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 4cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.dp_boundary = np.array([np.array([0, -2e4, -5e4]),
                                     np.array([0, -2e4, 5e4]),
                                     np.array([0, 2e4, 5e4]),
                                     np.array([0, 2e4, -5e4]),
                                     np.array([0, -2e4, -5e4]),
                                     ])

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.physical_location += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.dp_boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.physical_location = np.ascontiguousarray(rot_mat.dot(self.physical_location))
        self.physical_positive_direction = np.ascontiguousarray(rot_mat.dot(self.physical_positive_direction))

        if include_boundary:
            self.dp_boundary = np.asanyarray(np.dot(self.dp_boundary, rot_mat.T))

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
            delta = target - self.control_location
            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 3: change the physical location

                # Get the physical displacement of the table
                physical_motion = delta + self.dp_resolution * (np.random.rand() - 0.5)
                physical_motion = physical_motion * self.control_positive * self.physical_positive_direction

                # Move the stage table
                self.physical_location = self.physical_location + physical_motion

                # Step 4: Change the status in the control system

                self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                self.control_location = target + self.dp_feedback_noise_instance

                print("Motor moved to {:.2f} um".format(self.control_location))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):
                    # Get the physical displacement of the table
                    physical_motion = self.control_backlash + delta + self.dp_resolution * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.physical_positive_direction

                    # Move the stage table
                    self.physical_location = self.physical_location + physical_motion

                    # Get the physical displacement of the table
                    physical_motion = -self.control_backlash + self.dp_resolution * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.physical_positive_direction

                    # Move the stage table
                    self.physical_location = self.physical_location + physical_motion

                    # Step 4: Change the status in the control system
                    self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                    self.control_location = target + self.dp_feedback_noise_instance
                    print("Motor moved to {:.2f} um".format(self.control_location))

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
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False


class RotationMotor:
    def __init__(self,
                 upperLim=np.deg2rad(180),
                 lowerLim=-np.deg2rad(-180),
                 res=1.0,
                 backlash=0.05,
                 feedback_noise_level=1.0,
                 speed_rad_per_ps=1 * 1000 / 1e12,
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed_rad_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        self.control_motor_type = "Rotation"

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim

        self.control_location = 0.0  # rad

        self.control_backlash = backlash
        self.control_speed = speed_rad_per_ps

        # ---------------------------------------------------
        # Define quantities of the physical system
        # ---------------------------------------------------
        self.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.physical_deg0direction[1] = 1.0
        self.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.physical_rotation_axis[0] = 1.0
        self.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # ---------------------------------------------------
        # Define quantities describing the error between the control system and the physical system
        # I call them device property (dp_...)
        # ---------------------------------------------------
        self.dp_resolution = res
        self.dp_feedback_noise = feedback_noise_level
        self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 10cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.dp_boundary = np.array([np.array([0, -5e4, -5e4]),
                                     np.array([0, -5e4, 5e4]),
                                     np.array([0, 5e4, 5e4]),
                                     np.array([0, 5e4, -5e4]),
                                     np.array([0, -5e4, -5e4]),
                                     ])

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.physical_rotation_center += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.dp_boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.physical_deg0direction = np.ascontiguousarray(rot_mat.dot(self.physical_deg0direction))
        self.physical_rotation_center = np.ascontiguousarray(rot_mat.dot(self.physical_rotation_center))
        self.physical_rotation_axis = np.ascontiguousarray(rot_mat.dot(self.physical_rotation_axis))

        if include_boundary:
            self.dp_boundary = np.asanyarray(np.dot(self.dp_boundary, rot_mat.T))

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
            delta = target - self.control_location

            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction

                # Step 3: change the physical status of the motor
                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + self.dp_resolution * (np.random.rand() - 0.5),
                    axis=self.physical_rotation_axis)

                # Update the zero deg direction
                self.physical_deg0direction = np.dot(rotMat, self.physical_deg0direction)

                # Step 4 : change the control system information
                self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                self.control_location = target + self.dp_feedback_noise_instance

                print("Motor moved to {:.2f} um".format(self.control_location))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.control_backlash + delta + self.dp_resolution * (np.random.rand() - 0.5),
                        axis=self.physical_rotation_axis)

                    self.physical_deg0direction = np.dot(rotMat1, self.physical_deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.control_backlash + self.dp_resolution * (np.random.rand() - 0.5),
                        axis=self.physical_rotation_axis)
                    self.physical_deg0direction = np.dot(rotMat2, self.physical_deg0direction)

                    self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                    self.control_location = target + self.dp_feedback_noise_instance
                    print("Motor moved to {:.2f} rad".format(self.control_location))
                else:
                    print("The target location {:.2f} rad plus backlash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

        if getMotionTime:
            return 0

    def user_getPosition(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False


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
        # Create the instance of each motors

        self.x = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = LinearMotor(upperLim=25000,
                             lowerLim=-25000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.th = RotationMotor(upperLim=np.deg2rad(45),
                                lowerLim=-np.deg2rad(45),
                                res=2e-6,
                                backlash=0.05,
                                feedback_noise_level=1,
                                speed_rad_per_ps=0.01 / 1e12, )

        self.chi = RotationMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=1e-6,
                                 backlash=0.05,
                                 feedback_noise_level=1,
                                 speed_rad_per_ps=0.1 / 1e12, )

        self.optics = channelCut

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x.physical_positive_direction[1] = 1.0  #
        # Define the installation location of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.y.physical_positive_direction[0] = 1.0  #
        # Define the installation location of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.th.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.th.physical_deg0direction[1] = 1.0  #
        self.th.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.th.physical_rotation_axis[0] = 1.0
        self.th.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation location of the x stage
        th_stage_center = np.zeros(3, dtype=np.float64)
        th_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.th.shift(displacement=th_stage_center)

        self.chi.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.chi.physical_deg0direction[0] = 1.0  #
        self.chi.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.chi.physical_rotation_axis[1] = 1.0
        self.chi.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.chi.physical_rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation location of the x stage
        chi_stage_center = np.zeros(3, dtype=np.float64)
        chi_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.chi.shift(displacement=chi_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        self.optics.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.physical_positive_direction * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.chi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.physical_positive_direction * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.chi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def th_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.th.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.th.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.th.user_move_abs(target=target, getMotionTime=True)
        self.chi.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.physical_rotation_center, include_boundary=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.physical_rotation_center, include_boundary=True)

    def chi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.chi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.chi.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.chi.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.chi.physical_rotation_center, include_boundary=True)

    def plot_motors(self, ax):
        # Plot motors and crystals one by one
        ax.plot(self.x.dp_boundary[:, 2], self.x.dp_boundary[:, 1],
                linestyle='--', linewidth=1, label="x", color=self.color_list[0])
        ax.plot(self.y.dp_boundary[:, 2], self.y.dp_boundary[:, 1],
                linestyle='--', linewidth=1, label="y", color=self.color_list[1])
        ax.plot(self.th.dp_boundary[:, 2], self.th.dp_boundary[:, 1],
                linestyle='--', linewidth=1, label="th", color=self.color_list[2])
        ax.plot(self.chi.dp_boundary[:, 2], self.chi.dp_boundary[:, 1],
                linestyle='--', linewidth=1, label="chi", color=self.color_list[3])
        for crystal in self.optics.crystal_list:
            ax.plot(crystal.boundary[:, 2], crystal.boundary[:, 1],
                    linestyle='-', linewidth=3, label="crystal", color=self.color_list[4])
