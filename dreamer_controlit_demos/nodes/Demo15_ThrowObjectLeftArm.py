#!/usr/bin/env python

'''
Makes Dreamer throw an object using her left arm. Uses 5-DOF end effector control.

Usage Notes:

  $ roslaunch dreamer_controlit simulate_cartesian_both_hands.launch
  $ rosrun dreamer_controlit_demos Demo15_ThrowObjectLeftArm.py

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

DEFAULT_READY_RH_CARTPOS = [0.25822435038901964, -0.1895604971725577, 1.0461857180093073]
DEFAULT_READY_RH_ORIENT = [0.5409881394605172, -0.8191390472602035, 0.19063854336595773]
DEFAULT_READY_LH_CARTPOS = [0.25822435038901964, 0.25, 1.0461857180093073]
DEFAULT_READY_LH_ORIENT = [0.5409881394605172, 0.8191390472602035, 0.19063854336595773]
DEFAULT_READY_POSTURE = [0.06796522908004803, 0.06796522908004803,                                            # torso
    -0.08569654146540764, 0.07021124925432169, 0,                    1.7194162945362514, 1.51, -0.07, -0.18,  # left arm
    -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18]  # right arm


# The time each trajectory should take
TIME_WINDUP = 5.0
TIME_THROW = 10.0
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

    def createTrajectories(self):

        # ==============================================================================================
        # Define the Windup trajectory
        self.trajWindUp = Trajectory.Trajectory("Windup", TIME_WINDUP)

        # These are the initial values as specified in the YAML ControlIt! configuration file
        self.trajWindUp.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        self.trajWindUp.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajWindUp.setInitRHOrientWP([1.0, 0.0, 0.0])
        self.trajWindUp.setInitLHOrientWP([1.0, 0.0, 0.0])
        self.trajWindUp.setInitPostureWP(DEFAULT_POSTURE)

        self.trajWindUp.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajWindUp.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajWindUp.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajWindUp.addRHCartWP(DEFAULT_READY_RH_CARTPOS)

        self.trajWindUp.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajWindUp.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajWindUp.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajWindUp.addRHOrientWP(DEFAULT_READY_RH_ORIENT)

        self.trajWindUp.addLHCartWP([0.0995847062304291, 0.19801801741887398, 0.8155106479182263])
        self.trajWindUp.addLHCartWP([-0.019374787306896842, 0.2055681101286158, 0.8227934746708964])
        self.trajWindUp.addLHCartWP([-0.191268546695847, 0.24031719464302984, 0.8455461607957308])
        self.trajWindUp.addLHCartWP([-0.3354688498321601, 0.24241563419149073, 0.9190208028990703])
        self.trajWindUp.addLHCartWP([-0.4699930914836071, 0.24114097922917663, 1.081677254267744])
        self.trajWindUp.addLHCartWP([-0.45346097028335175, 0.2941965293418364, 1.0416864880195948])

        self.trajWindUp.addLHOrientWP([0.11162942272878174, 0.9931975897591658, -0.03312732524401186])
        self.trajWindUp.addLHOrientWP([0.09148526111493414, 0.9939165277770623, -0.06132196026156375])
        self.trajWindUp.addLHOrientWP([0.09945933635980725, 0.9950182403706174, 0.006822150733242239])
        self.trajWindUp.addLHOrientWP([0.11083779250640635, 0.9937878489183526, -0.010034694541468317])
        self.trajWindUp.addLHOrientWP([0.10097205115380427, 0.9936112050318664, -0.050412479813197277])
        self.trajWindUp.addLHOrientWP([0.18533190630653829, 0.9701316962898558, -0.15651382162144967])

        self.trajWindUp.addPostureWP([0.0, 0.0,
            -0.10421757416543008, -0.006784497819703997, -0.09659828438002596, 0.5686092786711396, 1.5632948790303993, -0.022069623282861047, 0.07590091599361415,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajWindUp.addPostureWP([0.0, 0.0,
            -0.41846371086120077, 0.01652334460877522, -0.09715019145827647, 0.7488704686109962, 1.5863471767612942, -0.033791532765262505, -0.26315810494292424,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajWindUp.addPostureWP([0.0, 0.0,
            -0.6684212290255923, 0.07457856715213197, -0.0895252830170859, 0.582415673924257, 1.5878764394857898, 0.0024377981029377513, -0.27432903162820155,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajWindUp.addPostureWP([0.0, 0.0,
            -0.9139436345533417, 0.07270528372652063, -0.09364840533572998, 0.46746435471195863, 1.5826526070974956, 0.016336511642733167, -0.32941722520446115,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajWindUp.addPostureWP([0.0, 0.0,
            -1.327547896857091, 0.07257695413637195, -0.0935789476962944, 0.505081489870024, 1.5819262040282558, 0.021629324364269866, -0.38737095261532584,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajWindUp.addPostureWP([0.0, 0.0,
            -1.0975876705686198, 0.1525120716752228, -0.09222608490443436, 0.1998529335421232, 1.4505016354732394, -0.08036288282528328, -0.24709614750921563,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])

        # ==============================================================================================
        # Define the Throw trajectory
        self.trajThrow = Trajectory.Trajectory("Throw", TIME_THROW)
        self.trajThrow.setPrevTraj(self.trajWindUp)                        # This trajectory always starts where the GoToReady trajectory ends

        self.trajThrow.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajThrow.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajThrow.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajThrow.addRHCartWP(DEFAULT_READY_RH_CARTPOS)

        self.trajThrow.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajThrow.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajThrow.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajThrow.addRHOrientWP(DEFAULT_READY_RH_ORIENT)

        self.trajThrow.addLHCartWP([-0.43167247821127785, 0.296602206917838, 1.0093210121085392])
        self.trajThrow.addLHCartWP([-0.3144274806066712, 0.2820920678026165, 0.899788302949844])
        self.trajThrow.addLHCartWP([-0.1075310507148538, 0.2607566777406535, 0.8066404845731345])
        self.trajThrow.addLHCartWP([0.07555897844415246, 0.25784458791093434, 0.8011017646276007])
        self.trajThrow.addLHCartWP([0.27658567918754406, 0.22594834871993053, 0.8820771092368943])
        self.trajThrow.addLHCartWP([0.406620377366256, 0.2159898433097043, 1.0453859241778782])
        self.trajThrow.addLHCartWP([0.4549415677975072, 0.21368872831961108, 1.1778227984909866])

        self.trajThrow.addLHOrientWP([0.18925708481528716, 0.9719360841794492, -0.13972116560184386])
        self.trajThrow.addLHOrientWP([0.2891958503661352, 0.9571891522370404, 0.01243732168691452])
        self.trajThrow.addLHOrientWP([0.2422661720554381, 0.9668554789642354, 0.08060759687798638])
        self.trajThrow.addLHOrientWP([0.23042439544802462, 0.9492066143488008, 0.21426945946377712])
        self.trajThrow.addLHOrientWP([0.16563668818757002, 0.9456157783908007, 0.27995586649436977])
        self.trajThrow.addLHOrientWP([0.04823201935356781, 0.9648554639902868, 0.25831687114310914])
        self.trajThrow.addLHOrientWP([-0.032820460658610315, 0.9960107076304662, 0.0829788385530582])

        self.trajThrow.addPostureWP([0.0, 0.0,
            -0.9907625354841852, 0.15286447872398215, -0.09209587280371191, 0.13062388631006874, 1.4498180274372212, -0.08374983069564275, -0.25898490756131554,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajThrow.addPostureWP([0.0, 0.0,
            -0.7840377824270323, 0.13716477391965173, -0.09327244213961453, 0.309115957422396, 1.4467746907886727, 0.0447539151871275, -0.25469157809397275,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajThrow.addPostureWP([0.0, 0.0,
            -0.3637090095895495, 0.09722989834299803, -0.09135908821486008, 0.31143821613258066, 1.447493804076094, 0.030716398082253907, -0.21249595295113194,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajThrow.addPostureWP([0.0, 0.0,
            -0.029456970944792434, 0.09265425675333003, -0.09362248271274641, 0.3183362791424086, 1.4019006546853012, 0.08418747893064996, -0.3119836374304644,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajThrow.addPostureWP([0.0, 0.0,
            0.26553264914855856, 0.04468623413231656, -0.09212494443118274, 0.5372868430557058, 1.3498258333204074, 0.08441205078112499, -0.31804018029736625,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajThrow.addPostureWP([0.0, 0.0,
            0.48979876432528224, 0.04409878699906535, -0.09246822520345147, 0.8897420577950705, 1.3971446313686349, 0.04527617920964998, -0.28921751297424314,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajThrow.addPostureWP([0.0, 0.0,
            0.6974755670346813, 0.04554054793994541, -0.08896844639869776, 1.042991994216168, 1.5675002491964347, 0.07229444946246992, -0.17758078151861081,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])


        # ==============================================================================================
        # Define the GoToIdle trajectory
        self.trajGoToIdle = Trajectory.Trajectory("GoToIdle", TIME_GO_TO_IDLE)
        self.trajGoToIdle.setPrevTraj(self.trajThrow)                        # This trajectory always starts where the GoToReady trajectory ends

        self.trajGoToIdle.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajGoToIdle.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajGoToIdle.addRHCartWP(DEFAULT_READY_RH_CARTPOS)
        self.trajGoToIdle.addRHCartWP(DEFAULT_READY_RH_CARTPOS)

        self.trajGoToIdle.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajGoToIdle.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajGoToIdle.addRHOrientWP(DEFAULT_READY_RH_ORIENT)
        self.trajGoToIdle.addRHOrientWP(DEFAULT_READY_RH_ORIENT)

        self.trajGoToIdle.addLHCartWP([0.44401026721093223, 0.2118732568901639, 1.1656318004994706])
        self.trajGoToIdle.addLHCartWP([0.3870780383254859, 0.21230561385452268, 1.059453632433805])
        self.trajGoToIdle.addLHCartWP([0.2755066414666355, 0.211434756322385, 0.9479820976373647])
        self.trajGoToIdle.addLHCartWP([0.17461352357092289, 0.21599918849236854, 0.8615358950818717])
        self.trajGoToIdle.addLHCartWP([0.08232433795208735, 0.2226810297013515, 0.8113159695859907])
        self.trajGoToIdle.addLHCartWP([0.012912883004958874, 0.21643546999896202, 0.7944574920654524])
        self.trajGoToIdle.addLHCartWP([0.043342008588729, 0.2164044236199803, 0.7957362964500756])

        self.trajGoToIdle.addLHCartWP([0.25822435038901964, 0.1895604971725577, 1.0461857180093073])
        self.trajGoToIdle.addLHCartWP([0.21649227857092893, 0.3006839904787592, 1.1140502834793191])
        self.trajGoToIdle.addLHCartWP([0.11866831717348489, 0.4101100845056917, 1.209699047600146])
        self.trajGoToIdle.addLHCartWP([-0.03366873622218044, 0.40992725074781894, 1.1144948070701866])
        self.trajGoToIdle.addLHCartWP([-0.055152798770261954, 0.2907526623508046, 1.009663652974324])
        self.trajGoToIdle.addLHCartWP([0.019903910090688474, 0.28423307267223147, 0.9179288590591458])
        self.trajGoToIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82]) # Matches the start of trajectory GoToReady

        self.trajGoToIdle.addLHOrientWP([0.00960601026360149, 0.9956496824888905, 0.09267920115423435])
        self.trajGoToIdle.addLHOrientWP([0.0211119722318617, 0.9958364884086387, 0.08867904477627506])
        self.trajGoToIdle.addLHOrientWP([0.07225166658675553, 0.9952055393323725, 0.06592140136251745])
        self.trajGoToIdle.addLHOrientWP([0.2517702047397906, 0.9272072267454218, 0.27730582878177584])
        self.trajGoToIdle.addLHOrientWP([0.5980386409128169, 0.7475835065407074, 0.2889094749630949])
        self.trajGoToIdle.addLHOrientWP([0.9608012247008393, 0.17037495707605194, 0.21870843745658478])
        self.trajGoToIdle.addLHOrientWP([0.8954609737274308, -0.15499885046676978, 0.41728287873467934])


        self.trajGoToIdle.addPostureWP([0.0, 0.0,
            0.6444520865509912, 0.04494674561538298, -0.09134258267023784, 1.0857720510110065, 1.5629741691999843, 0.03565794348083835, -0.18472165238959634,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajGoToIdle.addPostureWP([0.0, 0.0,
            0.39647409574146864, 0.04479217124115421, -0.09038665209685341, 1.0766575194142864, 1.5626523388489713, 0.04598670816986003, -0.19139505441557894,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajGoToIdle.addPostureWP([0.0, 0.0,
            0.07091274368389647, 0.04350734854709134, -0.09246198899761747, 1.0669758162144527, 1.5631956268091818, 0.022039773622498364, -0.24041955887375677,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajGoToIdle.addPostureWP([0.0, 0.0,
            -0.0695861727898646, 0.04098058969446322, -0.09384228100519601, 0.8230063426156317, 1.2812658222886961, 0.06928917151987034, -0.168656589590271,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajGoToIdle.addPostureWP([0.0, 0.0,
            -0.1221742725152796, 0.03936644842091598, -0.09570827863009652, 0.536844884517059, 0.9468470685750172, 0.03651139093644122, -0.09621080997175521,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajGoToIdle.addPostureWP([0.0, 0.0,
            -0.1697038842551681, 0.018288561682723085, -0.0928623226839307, 0.37063487409137685, 0.2640334279515837, 0.026963932810918047, -0.10023404494803403,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])
        self.trajGoToIdle.addPostureWP([0.0, 0.0,
            -0.11137205086442183, 0.017600081654634808, -0.09111326840132516, 0.3650831021750848, -0.06972130361349083, 0.17903335075474963, -0.029155989068765556,
            -0.08569654146540764, 0.07021124925432169, -0.15649686418494702, 1.7194162945362514, 1.51, -0.07, -0.18])

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