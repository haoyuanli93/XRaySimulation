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


class LinearMotor:
    def __init__(self,
                 motor_type="Linear",
                 upperLim=25000,
                 lowerLim=-25000,
                 res=5,
                 backClash=100,
                 feedback_noise_level=1,
                 speed=1 * 1000 / 1e12,
                 ):
        """

        :param motor_type:
        :param upperLim:
        :param lowerLim:
        :param res:
        :param backClash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed: the speed in um / ps
        """
        self.motor_type = motor_type

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

        # Define all the parameters for noises
        self.feedback_noise_level = feedback_noise_level
        self.feedback_noise = (np.random.rand() - 0.5) * self.feedback_noise_level

        # The linear position of the motor assuming that there is no error.
        self.location_feedback = 0.0

    def shift(self):
        pass

    def rotate(self):
        pass

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
