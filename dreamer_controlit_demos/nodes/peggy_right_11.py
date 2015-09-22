#!/usr/bin/env python

'''
Real-Time Experiments:
This uses both posture and orientation control.
'''

import sys, getopt     # for getting and parsing command line arguments
import time
import math
import threading
import rospy
import smach
import smach_ros

from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, Bool, Int32

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import DreamerInterface
import Trajectory
import TrajectoryGeneratorCubicSpline

ENABLE_USER_PROMPTS = False

# Shoulder abductors about 10 degrees away from body and elbows bent 90 degrees
# DEFAULT_POSTURE = [0.0, 0.0,                                    # torso
#                    0.0, 0.174532925, 0.0, 1.57, 0.0, 0.0, 0.0,  # left arm
#                    0.0, 0.174532925, 0.0, 1.57, 0.0, 0.0, 0.0]  # right arm

# Shoulder abductors and elbows at about 10 degrees
DEFAULT_POSTURE = [0.0, 0.0,                                    # torso
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0,  # left arm
                   0.0, 0.174532925, 0.0, 0.174532925, 0.0, 0.0, 0.0]  # right arm

# The time each trajectory should take
TIME_GO_TO_GRASP = 5.0
TIME_TEST_TRAJECTORY = 2.0
TIME_GO_TO_LOCATION = 3.0
TIME_DROP = 2.0
TIME_GO_TO_IDLE = 4.0


class TrajectoryState(smach.State):
    def __init__(self, dreamerInterface, traj):
        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface
        self.traj = traj

    def execute(self, userdata):
        rospy.loginfo('Executing TrajectoryState')

        print  "FIX ME..."

        if self.dreamerInterface.followTrajectory(self.traj):
            return "done"
        else:
            return "exit"

class EnablePowerGraspState(smach.State):
    def __init__(self, dreamerInterface):
        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface

    def execute(self, userdata):
        rospy.loginfo('Executing EnablePowerGraspState')

        self.dreamerInterface.rightHandCmdMsg.data = True
        self.dreamerInterface.rightHandCmdPublisher.publish(self.dreamerInterface.rightHandCmdMsg)

        rospy.sleep(5) # allow fingers to move

        if rospy.is_shutdown():
            return "exit"
        else:
            return "done"

class DisablePowerGraspState(smach.State):
    def __init__(self, dreamerInterface):
        smach.State.__init__(self, outcomes=["done", "exit"])
        self.dreamerInterface = dreamerInterface

    def execute(self, userdata):
        rospy.loginfo('Executing DisablePowerGraspState')

        self.dreamerInterface.rightHandCmdMsg.data = False
        self.dreamerInterface.rightHandCmdPublisher.publish(self.dreamerInterface.rightHandCmdMsg)

        rospy.sleep(5) # allow fingers to move

        if rospy.is_shutdown():
            return "exit"
        else:
            return "done"

class SleepState(smach.State):
    def __init__(self, goodResult, period):
        smach.State.__init__(self, outcomes=[goodResult, "exit"])
        self.goodResult = goodResult
        self.period = period

    def execute(self, userdata):
        rospy.loginfo("Executing SleepState")
        rospy.sleep(self.period)

        if rospy.is_shutdown():
            return "exit"
        else:
            return self.goodResult

class TestTrajectory:
    def __init__(self):
        self.dreamerInterface = DreamerInterface.DreamerInterface(enableUserPrompts = ENABLE_USER_PROMPTS, useQuaternionControl = True)

    def createTrajectories(self):

	# get object position and orientation from sys.argv
	#if len(sys.argv) < 8:
    #        print 'Input argument is not enough! Abort!'
    #        sys.exit(1)
        #obj_pos = map(float, sys.argv[1:4])
        #obj_ori = map(float, sys.argv[4:])

        # ==============================================================================================
        #Define the GoToGrasp trajectory
        self.trajGoGrasp = Trajectory.Trajectory("GoToReady", TIME_GO_TO_GRASP)

        # These are the initial values as specified in the YAML ControlIt! configuration file
        self.trajGoGrasp.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        self.trajGoGrasp.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitPostureWP(DEFAULT_POSTURE)

        # right arm Position	
        self.trajGoGrasp.addRHCartWP([-0.08166930689863366, -0.19338011371236366, 0.776141554715435])
        self.trajGoGrasp.addRHCartWP([-0.08301425558470789, -0.34291346810074025, 0.79905655482559])
        self.trajGoGrasp.addRHCartWP([-0.015591230408061601, -0.48707671047333917, 0.868886407457948])
        self.trajGoGrasp.addRHCartWP([-0.0021217884938035093, -0.5983356653821588, 0.967200718001097])
        #self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        self.trajGoGrasp.addRHOrientWP([0.006531650461268684, 0.9992457356574519, -0.01913538927611968, 0.0331531927557672])
        self.trajGoGrasp.addRHOrientWP([0.13962307060600251, 0.9894866445809412, -0.014967359972568694, 0.0346057290914828])
        self.trajGoGrasp.addRHOrientWP([0.2733581341536706, 0.9534184814456771, -0.04511860159943651, 0.1193014732497229])
        self.trajGoGrasp.addRHOrientWP([0.39285807383231386, 0.9046878815730998, -0.04036977970332632, 0.15991451355723])
        #self.trajGoGrasp.addRHOrientWP(obj_ori)


        # left arm Position does not move
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])



        # trajectory
        self.trajGoGrasp.addPostureWP([-0.126714531471,-0.126714531471, # torso down
             -0.325610311638,0.0871395087978,-0.19980062622,0.484956899483,0.235398555636,-0.434080358261,-0.151741298631, # left arm
             -0.24325678349,-0.0407318589499,0.0332946809616,-0.0581534511906,0.00228373080246,0.116103856766,0.0528217657134]) # right arm
        self.trajGoGrasp.addPostureWP([-0.126841202673,-0.126841202673, # torso down
             -0.325754880423,0.0870588817173,-0.199831964863,0.485240541512,0.235530570682,-0.434242609683,-0.151839421871, # left arm
             -0.245053246522,0.231979342402,0.0328861757014,-0.0588514891828,0.0015740611617,0.116907176355,0.0472601716991]) # right arm
        self.trajGoGrasp.addPostureWP([-0.126277037919,-0.126277037919, # torso down
             -0.32616613053,0.0869972588084,-0.199706841463,0.485078586074,0.235497838653,-0.434154953736,-0.15180568869, # left arm
             -0.131165055468,0.516896402431,0.032814224251,-0.0044100687725,0.051498391,0.110343180062,0.0380047060515]) # right arm
        self.trajGoGrasp.addPostureWP([-0.12625051481,-0.12625051481, # torso down
             -0.326339641069,0.0868852714092,-0.199609357014,0.485052025306,0.235479880191,-0.434016537038,-0.15179315738, # left arm
             -0.109575468681,0.779616030724,0.0347690501114,0.0577783103503,0.057508181294,0.110070437604,0.0314356350873]) # right arm

 


    def createFSM(self):
        # define the states
        goToGraspState = TrajectoryState(self.dreamerInterface, self.trajGoGrasp)
        #goToLocationState = TrajectoryState(self.dreamerInterface, self.trajGoLocation)
        #goToIdleState = TrajectoryState(self.dreamerInterface, self.trajGoIdle)

        enablePowerGraspState = EnablePowerGraspState(self.dreamerInterface)
        #disablePowerGraspState = DisablePowerGraspState(self.dreamerInterface)


        # wire the states into a FSM
        self.fsm = smach.StateMachine(outcomes=['exit'])
        with self.fsm:
            smach.StateMachine.add("GoToGrasp", goToGraspState,
                transitions={'done':'EnablePowerGraspState',
                             'exit':'exit'})
            smach.StateMachine.add("EnablePowerGraspState", enablePowerGraspState,
                transitions={'done':'exit',
                             'exit':'exit'})
 #           smach.StateMachine.add("GoToDropLocation", goToLocationState,
 #               transitions={'done':'DisablePowerGraspState',
 #                            'exit':'exit'})
 #           smach.StateMachine.add("DisablePowerGraspState", disablePowerGraspState,
 #               transitions={'done':'GoToIdle',
 #                            'exit':'exit'})
 #           smach.StateMachine.add("GoToIdle", goToIdleState,
 #               transitions={'done':'exit',
 #                            'exit':'exit'})




    def run(self):

        if not self.dreamerInterface.connectToControlIt(DEFAULT_POSTURE):
            return

        self.createTrajectories()
        self.createFSM()

        # Create and start the introspection server
        sis = smach_ros.IntrospectionServer('server_name', self.fsm, '/SM_ROOT')
        sis.start()

        index = raw_input("Start demo? Y/n\n")
        if index == "N" or index == "n":
            return

        outcome = self.fsm.execute()

        print "Test trajectory done, waiting until ctrl+c is hit..."
        rospy.spin()  # just to prevent this node from exiting
        sis.stop()


# Main method
if __name__ == "__main__":
    rospy.init_node('TestTrajectory', anonymous=True)
    demo = TestTrajectory()
    demo.run()
