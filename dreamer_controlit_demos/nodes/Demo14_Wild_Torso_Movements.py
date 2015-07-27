#!/usr/bin/env python

'''
A demonstration of wild torso movements when the posture task
is not provided a trajectory that is "compatible" with the
Cartesian space trajectories.

-----------------
Dependency notes:

If you're using Python 2.7, you need to install Python's
enum package. Download it from here: https://pypi.python.org/pypi/enum34#downloads
Then run:
  $ sudo python setup.py install

Visualizing the FSM requires smach_viewer:
  $ sudo apt-get install ros-indigo-smach-viewer
You will need to modify /opt/ros/indigo/lib/python2.7/dist-packages/xdot/xdot.py
lines 487, 488, 593, and 594 to contain self.read_float() instead of self.read_number().

-----------------
Usage Notes:

  $ roslaunch dreamer_controlit simulate_cartesian_both_hands.launch
  $ rosrun dreamer_controlit_demos Demo14_Wild_Torso_Movements.py

To visualize FSM:
  $ rosrun smach_viewer smach_viewer.py
'''

# import sys, getopt     # for getting and parsing command line arguments
# import time
# import math
# import threading
import rospy
import smach
import smach_ros
from enum import IntEnum
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, Bool, Int32

# import numpy as np
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import DreamerInterface
import Trajectory
# import TrajectoryGeneratorCubicSpline

# import roslib; roslib.load_manifest('controlit_dreamer_integration')

ENABLE_USER_PROMPTS = False

# Shoulder abductors and elbows at about 10 degrees
DEFAULT_POSTURE = [0.0, 0.0,                                    # torso
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0,  # left arm
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0]  # right arm

# The time each trajectory should take
TIME_GO_TO_READY = 10.0
TIME_GO_TO_IDLE = 10.0

class TrajectoryState(smach.State):
    """
    A SMACH state that makes the robot follow a trajectory.
    """

    def __init__(self, dreamerInterface, traj):
        """
        The constructor.

        Keyword Parameters:
          - dreamerInterface: The object to which to provide the trajectory.
          - traj: The trajectory to follow.
        """

        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface
        self.traj = traj

    def execute(self, userdata):
        """
        Executes a trajectory then prompts the user whether to exit or continue.
        """

        rospy.loginfo('TrajectoryState: Executing...')

        if self.dreamerInterface.followTrajectory(self.traj):
            if rospy.is_shutdown():
                return "exit"
            else:
                index = raw_input("Exit? y/N\n")
                if index == "Y" or index == "y":
                    return "exit"
                else:
                    return "done"
        else:
            return "exit"

class Demo14_Wild_Torso_Movements:
    """
    The primary class that implement's the demo's FSM.
    """

    def __init__(self):
        self.dreamerInterface = DreamerInterface.DreamerInterface(ENABLE_USER_PROMPTS)

    def createTrajectories(self, providePostureTraj):

        # ==============================================================================================
        # Define the GoToReady trajectory
        self.trajGoToReady = Trajectory.Trajectory("GoToReady", TIME_GO_TO_READY)

        # These are the initial values as specified in the YAML ControlIt! configuration file
        self.trajGoToReady.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        self.trajGoToReady.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoToReady.setInitRHOrientWP([1.0, 0.0, 0.0])
        self.trajGoToReady.setInitLHOrientWP([1.0, 0.0, 0.0])
        self.trajGoToReady.setInitPostureWP(DEFAULT_POSTURE)

        self.trajGoToReady.addRHCartWP([0.019903910090688474, -0.28423307267223147, 0.9179288590591458])
        self.trajGoToReady.addRHCartWP([-0.055152798770261954, -0.2907526623508046, 1.009663652974324])
        self.trajGoToReady.addRHCartWP([-0.03366873622218044, -0.40992725074781894, 1.1144948070701866])
        self.trajGoToReady.addRHCartWP([0.11866831717348489, -0.4101100845056917, 1.209699047600146])
        self.trajGoToReady.addRHCartWP([0.21649227857092893, -0.3006839904787592, 1.1140502834793191])
        self.trajGoToReady.addRHCartWP([0.25822435038901964, -0.1895604971725577, 1.0461857180093073])

        self.trajGoToReady.addRHOrientWP([0.8950968852599132, 0.26432788250814326, 0.3590714922223199])
        self.trajGoToReady.addRHOrientWP([0.8944226954968388, 0.33098423072776184, 0.3007615015086225])
        self.trajGoToReady.addRHOrientWP([0.8994250702615956, 0.22626156457297464, 0.3739521993275524])
        self.trajGoToReady.addRHOrientWP([0.19818667912613866, -0.8161433027447201, 0.5428002851895832])
        self.trajGoToReady.addRHOrientWP([0.260956993686226, -0.8736061290033836, 0.4107478287392042])
        self.trajGoToReady.addRHOrientWP([0.5409881394605172, -0.8191390472602035, 0.19063854336595773])

        self.trajGoToReady.addLHCartWP([0.019903910090688474, 0.28423307267223147, 0.9179288590591458])
        self.trajGoToReady.addLHCartWP([-0.055152798770261954, 0.2907526623508046, 1.009663652974324])
        self.trajGoToReady.addLHCartWP([-0.03366873622218044, 0.40992725074781894, 1.1144948070701866])
        self.trajGoToReady.addLHCartWP([0.11866831717348489, 0.4101100845056917, 1.209699047600146])
        self.trajGoToReady.addLHCartWP([0.21649227857092893, 0.3006839904787592, 1.1140502834793191])
        self.trajGoToReady.addLHCartWP([0.25822435038901964, 0.25,               1.0461857180093073])

        self.trajGoToReady.addLHOrientWP([0.8950968852599132, -0.26432788250814326, 0.3590714922223199])
        self.trajGoToReady.addLHOrientWP([0.8944226954968388, -0.33098423072776184, 0.3007615015086225])
        self.trajGoToReady.addLHOrientWP([0.8994250702615956, -0.22626156457297464, 0.3739521993275524])
        self.trajGoToReady.addLHOrientWP([0.19818667912613866, 0.8161433027447201, 0.5428002851895832])
        self.trajGoToReady.addLHOrientWP([0.260956993686226, 0.8736061290033836, 0.4107478287392042])
        self.trajGoToReady.addLHOrientWP([0.5409881394605172, 0.8191390472602035, 0.19063854336595773])

        if providePostureTraj:
            self.trajGoToReady.addPostureWP([0.06826499288341317, 0.06826499288341317,
                           -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836,
                           -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836])
            self.trajGoToReady.addPostureWP([0.0686363596318602,  0.0686363596318602,
                           -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179,
                           -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179])
            self.trajGoToReady.addPostureWP([0.06804075180539401, 0.06804075180539401,
                           -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467,
                           -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467])
            self.trajGoToReady.addPostureWP([0.06818415549992426, 0.06818415549992426,
                           -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628,
                           -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628])
            self.trajGoToReady.addPostureWP([0.06794500584573498, 0.06794500584573498,
                           -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457,
                           -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457])
            self.trajGoToReady.addPostureWP([0.06796522908004803, 0.06796522908004803,                                                                   # torso
                           -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
                           -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])   # right arm
        else:
            self.trajGoToReady.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToReady.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToReady.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToReady.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToReady.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToReady.addPostureWP(DEFAULT_POSTURE)

        # ==============================================================================================
        # Define the GoToIdle trajectory
        self.trajGoToIdle = Trajectory.Trajectory("GoToIdle", TIME_GO_TO_IDLE)
        self.trajGoToIdle.setPrevTraj(self.trajGoToReady)                        # This trajectory always starts where the GoToReady trajectory ends

        # 2015.01.06 Trajectory
        self.trajGoToIdle.addRHCartWP([0.25822435038901964, -0.1895604971725577, 1.0461857180093073])
        self.trajGoToIdle.addRHCartWP([0.21649227857092893, -0.3006839904787592, 1.1140502834793191])
        self.trajGoToIdle.addRHCartWP([0.11866831717348489, -0.4101100845056917, 1.209699047600146])
        self.trajGoToIdle.addRHCartWP([-0.03366873622218044, -0.40992725074781894, 1.1144948070701866])
        self.trajGoToIdle.addRHCartWP([-0.055152798770261954, -0.2907526623508046, 1.009663652974324])
        self.trajGoToIdle.addRHCartWP([0.019903910090688474, -0.28423307267223147, 0.9179288590591458])
        self.trajGoToIdle.addRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addRHOrientWP([0.5409881394605172, -0.8191390472602035, 0.19063854336595773])
        self.trajGoToIdle.addRHOrientWP([0.260956993686226, -0.8736061290033836, 0.4107478287392042])
        self.trajGoToIdle.addRHOrientWP([0.19818667912613866, -0.8161433027447201, 0.5428002851895832])
        self.trajGoToIdle.addRHOrientWP([0.8994250702615956, 0.22626156457297464, 0.3739521993275524])
        self.trajGoToIdle.addRHOrientWP([0.8944226954968388, 0.33098423072776184, 0.3007615015086225])
        self.trajGoToIdle.addRHOrientWP([0.8950968852599132, 0.26432788250814326, 0.3590714922223199])
        self.trajGoToIdle.addRHOrientWP([1.0, 0.0, 0.0]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addLHCartWP([0.25822435038901964, 0.1895604971725577, 1.0461857180093073])
        self.trajGoToIdle.addLHCartWP([0.21649227857092893, 0.3006839904787592, 1.1140502834793191])
        self.trajGoToIdle.addLHCartWP([0.11866831717348489, 0.4101100845056917, 1.209699047600146])
        self.trajGoToIdle.addLHCartWP([-0.03366873622218044, 0.40992725074781894, 1.1144948070701866])
        self.trajGoToIdle.addLHCartWP([-0.055152798770261954, 0.2907526623508046, 1.009663652974324])
        self.trajGoToIdle.addLHCartWP([0.019903910090688474, 0.28423307267223147, 0.9179288590591458])
        self.trajGoToIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addLHOrientWP([0.5409881394605172, 0.8191390472602035, 0.19063854336595773])
        self.trajGoToIdle.addLHOrientWP([0.260956993686226, 0.8736061290033836, 0.4107478287392042])
        self.trajGoToIdle.addLHOrientWP([0.19818667912613866, 0.8161433027447201, 0.5428002851895832])
        self.trajGoToIdle.addLHOrientWP([0.8994250702615956, -0.22626156457297464, 0.3739521993275524])
        self.trajGoToIdle.addLHOrientWP([0.8944226954968388, -0.33098423072776184, 0.3007615015086225])
        self.trajGoToIdle.addLHOrientWP([0.8950968852599132, -0.26432788250814326, 0.3590714922223199])
        self.trajGoToIdle.addLHOrientWP([1.0, 0.0, 0.0]) # Matches the start of trajectory GoToReady

        if providePostureTraj:
            self.trajGoToIdle.addPostureWP([0.06796522908004803, 0.06796522908004803,                                                  # torso
                           -0.08569654146540764, 0.07021124925432169,                    0, 1.7194162945362514, 1.51, -0.07, -0.18,    # left arm
                           -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])   # right arm
            self.trajGoToIdle.addPostureWP([0.06794500584573498, 0.06794500584573498, -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457, -0.24608246913199228, 0.13441397755549533, 0.2542869735593113,   2.0227000417984633, 1.3670468713459782,  -0.45978204939890815, 0.030219082955597457])
            self.trajGoToIdle.addPostureWP([0.06818415549992426, 0.06818415549992426, -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628, -0.8497599545494692,  0.47079074342878563, 0.8355038507753617,   2.2318590905389852, 1.8475059506175733,  -0.405570582208143,   -0.0277359315904628])
            self.trajGoToIdle.addPostureWP([0.06804075180539401, 0.06804075180539401, -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467, -1.3637873691001094,  0.3926057912988488,  0.575755053425441,    1.9732992187122156, 0.29999797251313004, -0.20309827518257023, 0.05586603055643467])
            self.trajGoToIdle.addPostureWP([0.0686363596318602,  0.0686363596318602,  -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179, -1.0914342991625676,  0.39040871074764566, -0.03720209764435387, 1.7583823306095314, 0.05438773164693069, -0.20257591921666193, 0.06386553930484179])
            self.trajGoToIdle.addPostureWP([0.06826499288341317, 0.06826499288341317, -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836, -0.6249282444166423,  0.3079607416653748,  -0.1220981510225299,  1.3675006234559883, 0.06394316468492173, -0.20422693251592328, 0.06223224746326836])
            self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE) # Matches the start of trajectory GoToReady
        else:
            self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE)
            self.trajGoToIdle.addPostureWP(DEFAULT_POSTURE)

    def createFSM(self):
        # define the states
        goToReadyState = TrajectoryState(self.dreamerInterface, self.trajGoToReady)
        goToIdleState = TrajectoryState(self.dreamerInterface, self.trajGoToIdle)

        # wire the states into a FSM
        self.fsm = smach.StateMachine(outcomes=['exit'])
        # self.fsm.userdata.endEffectorSide = "right"
        self.fsm.userdata.demoName = "none"

        with self.fsm:
            smach.StateMachine.add("GoToReadyState", goToReadyState,
                transitions={'done':'GoToIdleState',
                             'exit':'exit'})
            smach.StateMachine.add("GoToIdleState", goToIdleState,
                transitions={'done':'GoToReadyState',
                             'exit':'exit'})

    def run(self):
        """
        Runs Demo 14.
        """

        if not self.dreamerInterface.connectToControlIt(DEFAULT_POSTURE):
            return

        providePostureTraj = False
        index = raw_input("Provide \"compatible\" posture trajectory? y/N\n")
        if index == "Y" or index == "y":
            providePostureTraj = True

        self.createTrajectories(providePostureTraj)
        self.createFSM()

        # Create and start the introspection server
        sis = smach_ros.IntrospectionServer('server_name', self.fsm, '/SM_ROOT')
        sis.start()

        if ENABLE_USER_PROMPTS:
            index = raw_input("Start demo? Y/n\n")
            if index == "N" or index == "n":
                return

        outcome = self.fsm.execute()

        print "Demo 14 done, waiting until ctrl+c is hit..."
        rospy.spin()  # just to prevent this node from exiting
        sis.stop()


# Main method
if __name__ == "__main__":
    rospy.init_node('Demo14_Wild_Torso_Movements', anonymous=True)
    demo = Demo14_Wild_Torso_Movements()
    demo.run()