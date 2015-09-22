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
	if len(sys.argv) < 8:
            print 'Input argument is not enough! Abort!'
            sys.exit(1)
        obj_pos = map(float, sys.argv[1:4])
        obj_ori = map(float, sys.argv[4:])

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
        self.trajGoGrasp.addRHCartWP([-0.07240654972621982, -0.18926498253418547, 0.777797533624643])
        self.trajGoGrasp.addRHCartWP([-0.06280671829675111, -0.40011642806917397, 0.82111143670776])
        self.trajGoGrasp.addRHCartWP([-0.06025639273866295, -0.5341569560363296, 0.901103752574618])
        self.trajGoGrasp.addRHCartWP([-0.007862037094822966, -0.6527893680147459, 1.03977919242589])
        self.trajGoGrasp.addRHCartWP([0.05417095868522431, -0.7093414332293393, 1.230334433742191])
        self.trajGoGrasp.addRHCartWP([0.18710400113181638, -0.5718990887963695, 1.22885703886998])
        self.trajGoGrasp.addRHCartWP([0.1960799140016424, -0.4369593571036633, 1.227548977796905])
        self.trajGoGrasp.addRHCartWP([0.1745164785222798, -0.31922521625343564, 1.19078543793328])
        self.trajGoGrasp.addRHCartWP([0.16428686379619362, -0.2372991785682392, 1.140699284449669])
        self.trajGoGrasp.addRHCartWP([0.21072186975754947, -0.20122576241017265, 1.11816156028849])
        self.trajGoGrasp.addRHCartWP([0.21614753772044126, -0.21760630847131135, 1.02462911060823])
        #self.trajGoGrasp.addRHCartWP([0.21882659167258459, -0.21523054829550878, 1.020498736759905])
        self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        self.trajGoGrasp.addRHOrientWP([-0.025085112105937016, 0.9979605031943115, 0.029350791912220515, -0.0508340656343224])
        self.trajGoGrasp.addRHOrientWP([0.16344839733755745, 0.98532467098872, 0.025428650310729557, -0.042110543716857])
        self.trajGoGrasp.addRHOrientWP([0.2936237989823493, 0.9546844720057241, -0.04352609498526412, -0.0216356797364524])
        self.trajGoGrasp.addRHOrientWP([0.3723771748255084, 0.8666500001754495, -0.2577525346817052, 0.2093242645514912])
        self.trajGoGrasp.addRHOrientWP([0.4069075683420487, 0.7206077965294737, -0.2802143507435294, 0.486446864564136])
        self.trajGoGrasp.addRHOrientWP([0.2766024741169928, 0.6767422455964259, -0.07939748193875634, 0.67764817139814])
        self.trajGoGrasp.addRHOrientWP([0.17135778722757275, 0.7961437696303232, -0.06633118235375171, 0.576534284394989])
        self.trajGoGrasp.addRHOrientWP([0.00372093668433573, 0.8323574462624181, 0.0710204621554687, 0.549657466279609])
        self.trajGoGrasp.addRHOrientWP([-0.029074287114389127, 0.8612613494086716, -0.09515964292192347, 0.4983254119574])
        self.trajGoGrasp.addRHOrientWP([0.009622762624631681, 0.774412162156797, -0.025168477873009958, 0.632107390610694])
        self.trajGoGrasp.addRHOrientWP([0.01440372407211505, 0.7644157870618747, -0.0009125707664297478, 0.644562025283855])
        #self.trajGoGrasp.addRHOrientWP([0.0041712811207543455, 0.7679679300827776, 0.0027818525625918763, 0.640468672203805])
        self.trajGoGrasp.addRHOrientWP(obj_ori)


        # left arm Position does not move
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])



        # left arm Orientation does not move
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoGrasp.addPostureWP([-0.11019353905,-0.11019353905, # torso down
             -0.310448992556,0.0420380640406,-0.151313398588,0.398792395899,0.201951205449,-0.390995204069,-0.154808336199, # left arm
             -0.238071464623,-0.0484254401648,-0.0427613091051,-0.00430498453681,-0.0152571980608,-0.0813024058166,-0.00352893659313]) # right arm
        self.trajGoGrasp.addPostureWP([-0.110352210274,-0.110352210274, # torso down
             -0.310378842473,0.0421124473579,-0.151197553146,0.398811435837,0.201845189856,-0.390913598083,-0.154791283391, # left arm
             -0.239523030853,0.341266058301,-0.0418051654107,0.0300699037606,-0.0166201198056,-0.0863917831259,-0.0145446149069]) # right arm
        self.trajGoGrasp.addPostureWP([-0.110259661055,-0.110259661055, # torso down
             -0.310161928392,0.0421209870143,-0.151106027945,0.398835214606,0.2018691817,-0.391039127106,-0.154953319575, # left arm
             -0.236790419898,0.622551743518,-0.0413839495724,0.0298440555147,0.119933500157,-0.0851660801189,-0.020823358247]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109792237347,-0.109792237347, # torso down
             -0.311410677242,0.0420785945391,-0.151288567653,0.39884572527,0.202014384439,-0.391004592321,-0.154971214661, # left arm
             -0.155720523088,0.9406558513,0.0830379901395,0.124874199188,0.534922798146,-0.0453560697765,-0.0749108987506]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109814585177,-0.109814585177, # torso down
             -0.312093229745,0.0420919749649,-0.151214367621,0.398748516857,0.201954700741,-0.391016785284,-0.154776155931, # left arm
             -0.12204163506,1.21146749624,0.385322296528,0.420667901458,0.644961119653,-0.0231121934628,-0.106436488727]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109741327166,-0.109741327166, # torso down
             -0.311829960527,0.0421815465587,-0.151160210676,0.398924842123,0.201989216105,-0.390848206984,-0.155109843091, # left arm
             -0.101642613064,0.901031396058,0.406276343471,1.16780755424,0.475612991642,-0.0499590094531,-0.0800531581967]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109541314212,-0.109541314212, # torso down
             -0.313381601215,0.0421314446411,-0.151192220971,0.398934821478,0.201988676347,-0.39137050593,-0.154816741533, # left arm
             -0.361871251159,0.958365895708,0.204452670483,1.68810279415,0.862551512743,-0.42189743697,0.294675496738]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109560111846,-0.109560111846, # torso down
             -0.313606396862,0.0421199066713,-0.151244464326,0.398675384434,0.201864172334,-0.391214204958,-0.154679992309, # left arm
             -0.365668167333,0.910046577572,-0.129917430988,1.96464367353,0.891507093805,-0.420885384524,0.296251232673]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109334803153,-0.109334803153, # torso down
             -0.313530450545,0.0421469226534,-0.151106419565,0.398752128764,0.201931263078,-0.39130583394,-0.154927027669, # left arm
             -0.367994005109,0.333507152518,-0.130341099864,1.9876902062,0.70412126966,-0.627637215931,0.513365062925]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109686914869,-0.109686914869, # torso down
             -0.312012930578,0.0420552094903,-0.151164483051,0.398743463302,0.201719296642,-0.391233351186,-0.154864171085, # left arm
             -0.199104736254,0.149377164232,-0.175906950144,1.77553249501,0.225884862837,-0.386958485519,0.275391257132]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109676001128,-0.109676001128, # torso down
             -0.312649737829,0.0423201728402,-0.151139582909,0.398937642282,0.202090738375,-0.391347399677,-0.154687561551, # left arm
             -0.199363437016,0.1480114811,-0.164257138729,1.43128497453,0.157709681196,-0.0289196995721,0.16271460221]) # right arm
        self.trajGoGrasp.addPostureWP([-0.109727263999,-0.109727263999, # torso down
             -0.313134767712,0.0422090375761,-0.151181020651,0.398863006282,0.20177613383,-0.391342762949,-0.154740102845, # left arm
             -0.187520702292,0.138806299814,-0.16652264446,1.40521542561,0.159380347644,-0.024740084984,0.142082085265]) # right arm


    # ==============================================================================================
    # Define the GoTOLocation trajectory
        self.trajGoLocation = Trajectory.Trajectory("GoToLocation", TIME_GO_TO_LOCATION)

	'''
        ==== END POSITION OF LAST ACTION ====
        self.trajGoGrasp.addPostureWP([-0.0326800201648,-0.0326800201648, # torso down
             -0.0618155402256,0.0102851437315,-0.0252052788739,0.114671676562,0.0467629891621,-0.0323641577321,-0.0368239745189, # left arm
             0.172226532381,-0.128085166684,-0.288259357413,1.09227174884,0.260798872242,0.0228264644753,-0.35851360908]) # right arm
        self.trajGoGrasp.addRHCartWP([0.3080369181313972, -0.0933579900905777, 1.01059794106796])
        self.trajGoGrasp.addRHOrientWP([-0.33079164055278293, 0.7242129795026276, 0.108525588846752, 0.595243351433504])
        ==== END POSITION OF LAST ACTION ====
	'''

        # These are the initial values as specified in the YAML ControlIt! configuration file
        #self.trajGoLocation.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        #self.trajGoLocation.setInitRHCartWP([0.3080369181313972, -0.0933579900905777, 1.01059794106796])
        # self.trajGoLocation.setInitRHCartWP(obj_pos)
        # self.trajGoLocation.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        # #self.trajGoLocation.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        # #self.trajGoLocation.setInitRHOrientWP([-0.33079164055278293, 0.7242129795026276, 0.108525588846752, 0.595243351433504])
        # self.trajGoLocation.setInitRHOrientWP(obj_ori)
        # self.trajGoLocation.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        # #self.trajGoLocation.setInitPostureWP(DEFAULT_POSTURE)
        # self.trajGoGrasp.addPostureWP([-0.0326800201648,-0.0326800201648, # torso down
        #      -0.0618155402256,0.0102851437315,-0.0252052788739,0.114671676562,0.0467629891621,-0.0323641577321,-0.0368239745189, # left arm
        #      0.172226532381,-0.128085166684,-0.288259357413,1.09227174884,0.260798872242,0.0228264644753,-0.35851360908]) # right arm
   
        self.trajGoLocation.setPrevTraj(self.trajGoGrasp)

        # right arm Position
        self.trajGoLocation.addRHCartWP([0.2142987951616763, -0.16789930344823858, 1.040141795541417])
        self.trajGoLocation.addRHCartWP([0.17260548338284623, -0.09582553676184068, 1.079261044317009])
        self.trajGoLocation.addRHCartWP([0.21274517398813994, -0.0784930840665991, 1.080368245019013])
        self.trajGoLocation.addRHCartWP([0.2269644686422892, -0.09208789705465098, 1.011355619138627])
        self.trajGoLocation.addRHCartWP([0.18074123429179909, -0.11333789346073705, 1.144531778647654])



        # right arm Orientation
        self.trajGoLocation.addRHOrientWP([-0.008297396192958701, 0.7376755456837832, 0.06373210620142455, 0.672089399675132])
        self.trajGoLocation.addRHOrientWP([-0.10563024417632112, 0.7539964233988904, 0.13694241034223914, 0.633702155011792])
        self.trajGoLocation.addRHOrientWP([-0.12817430572616825, 0.7396010049873987, 0.08715681654693694, 0.65495449468085])
        self.trajGoLocation.addRHOrientWP([-0.08305739329300292, 0.7395117234129913, 0.09315678187677309, 0.661472368541938])
        self.trajGoLocation.addRHOrientWP([-0.12089751611649303, 0.59902531997337, 0.06522913944699466, 0.788858425824728])
        


        # left arm Position does not move
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoLocation.addPostureWP([-0.109901968672,-0.109901968672, # torso down
             -0.312655627737,0.0421862230036,-0.151350456582,0.398709789928,0.201881990198,-0.391320271946,-0.154716359501, # left arm
             -0.196988301508,0.059877197384,-0.238214039572,1.49570744346,0.00325764481501,-0.0385455160001,0.131905964192]) # right arm
        self.trajGoLocation.addPostureWP([-0.110140823537,-0.110140823537, # torso down
             -0.313362981547,0.0422425200376,-0.151278702527,0.398653452019,0.201846649829,-0.391226474908,-0.154722866654, # left arm
             -0.186641504628,0.155597450274,-0.607982655761,1.68717653268,0.181101980414,-0.227013941122,0.27670773801]) # right arm
        self.trajGoLocation.addPostureWP([-0.109535918573,-0.109535918573, # torso down
             -0.312383731855,0.0423468775142,-0.151209125896,0.398741518878,0.201812471805,-0.391313763376,-0.15477258949, # left arm
             -0.0906141141237,0.0150316722145,-0.527789343279,1.54284170742,0.17691154292,-0.164949023834,0.234280017928]) # right arm
        self.trajGoLocation.addPostureWP([-0.109171748013,-0.109171748013, # torso down
             -0.312114074707,0.0422530071936,-0.151058761535,0.399126126493,0.201936856344,-0.391244444828,-0.154924797356, # left arm
             -0.0555196013846,0.00681409506773,-0.492177165783,1.25019064731,0.107952783788,0.0705839035561,0.217547039746]) # right arm
        self.trajGoLocation.addPostureWP([-0.10951222438,-0.10951222438, # torso down
             -0.31223472355,0.0421230687851,-0.151268396311,0.398977408692,0.201783135776,-0.391356056687,-0.154915388974, # left arm
             -0.159947853044,0.137691039955,-0.506499456794,1.88518980694,0.119583816739,-0.0176454141924,0.277207002575]) # right arm
        


        # ==============================================================================================
        #Define the GoToIdle trajectory
        self.trajGoIdle = Trajectory.Trajectory("GoToIdle", TIME_GO_TO_IDLE)

        # # These are the initial values as specified in the YAML ControlIt! configuration file
        # #self.trajGoIdle.setInitRHCartWP([0.033912978219317776, -0.29726881641499886, 0.82])
        # #self.trajGoIdle.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        # self.trajGoIdle.setInitRHOrientWP([-0.3417090583413905, 0.6227374186730135, 0.18086441069396716, 0.680236055921939])
        # self.trajGoIdle.setInitRHCartWP([0.2711818442848606, -0.11725385932626933, 1.06368001058201])
        # self.trajGoIdle.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        # self.trajGoIdle.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        # #self.trajGoIdle.setInitPostureWP(DEFAULT_POSTURE)
        # self.trajGoIdle.setInitPostureWP([-0.0332891145262,-0.0332891145262, # torso down
        #      -0.0622898657978,0.010319632391,-0.0252854737271,0.114916347834,0.0467076603524,-0.0324222505344,-0.0370686454575, # left arm
        #      0.0307719078756,0.00180269585053,-0.362223883914,1.47969587707,0.266268723735,0.0766932359528,-0.411311617134]) # right arm

        self.trajGoIdle.setPrevTraj(self.trajGoLocation)

        # right arm Position    
        self.trajGoIdle.addRHCartWP([0.19660653215808815, -0.19418452158154903, 1.162583779525237])
        self.trajGoIdle.addRHCartWP([0.1901731333041752, -0.2776233741354989, 1.129394401675841])
        self.trajGoIdle.addRHCartWP([0.19956191715889746, -0.3863668678624899, 1.083345866143807])
        self.trajGoIdle.addRHCartWP([0.1440961465813244, -0.48673098047494484, 1.02852434100105])
        self.trajGoIdle.addRHCartWP([0.06990914200368348, -0.5290927268180079, 0.976406300910272])
        self.trajGoIdle.addRHCartWP([0.013826735139140617, -0.4216914018405109, 0.856078693845317])
        self.trajGoIdle.addRHCartWP([-0.029052504847932595, -0.30631460630002666, 0.795150100133595])
        self.trajGoIdle.addRHCartWP([-0.07284125970018052, -0.2565276235226578, 0.781911822178236])
        self.trajGoIdle.addRHCartWP([-0.09063355366673864, -0.18916358784290918, 0.778267253907254])
        self.trajGoIdle.addRHCartWP([-0.08804507809050312, -0.1919562738205294, 0.778135199949866])



        # right arm Orientation 
        self.trajGoIdle.addRHOrientWP([-0.04447558984422483, 0.5859625120589901, -0.004178351069112669, 0.80910592492693])
        self.trajGoIdle.addRHOrientWP([-0.08451577433804146, 0.6099716358315463, 0.04683347404044698, 0.786510211680898])
        self.trajGoIdle.addRHOrientWP([0.12282965398027672, 0.34963278424567995, -0.06271805886140752, 0.926680223904857])
        self.trajGoIdle.addRHOrientWP([0.33562400688558414, 0.3487440299557773, -0.2721733459415044, 0.831658461948939])
        self.trajGoIdle.addRHOrientWP([0.42544912040153665, 0.3422066713954373, -0.48495328986512237, 0.68316026425064])
        self.trajGoIdle.addRHOrientWP([-0.41854183602444606, -0.4927854225738399, 0.6177471162400302, -0.4476312759103573])
        self.trajGoIdle.addRHOrientWP([-0.3896799165079565, -0.5929021667050804, 0.6780820769886617, -0.1918887184112592])
        self.trajGoIdle.addRHOrientWP([0.3076508945089083, 0.9021995024041238, -0.182665178822886, 0.2408742771970951])
        self.trajGoIdle.addRHOrientWP([0.21984202984611925, 0.942603246547382, -0.07851017148483685, 0.2387566846915389])
        self.trajGoIdle.addRHOrientWP([0.011454803721943112, 0.9975975464192318, 0.05067381133544077, 0.0458267137678503])

 
        # left arm Position does not move
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoIdle.addPostureWP([-0.10946915104,-0.10946915104, # torso down
             -0.312788607432,0.0421144943615,-0.151063958121,0.398789393916,0.201936037397,-0.391336392808,-0.154725080494, # left arm
             -0.179412663571,0.242985766277,-0.254287685681,1.93636966038,0.229946771281,-0.00902507239109,0.276235614529]) # right arm
        self.trajGoIdle.addPostureWP([-0.109690094503,-0.109690094503, # torso down
             -0.314147106941,0.0422342459471,-0.151214146804,0.396641970318,0.201792389876,-0.391150165408,-0.154581566063, # left arm
             -0.287557936099,0.526431068527,-0.167147582024,1.82512583682,0.514190125999,0.183897506236,0.0811403890676]) # right arm
        self.trajGoIdle.addPostureWP([-0.10965765681,-0.10965765681, # torso down
             -0.314623300578,0.0422264264785,-0.151014323708,0.396384088821,0.201828114886,-0.391440895994,-0.154709467611, # left arm
             -0.274546435614,0.49031820603,0.124331227757,1.48079617956,0.376916106739,0.915638568337,0.197808887176]) # right arm
        self.trajGoIdle.addPostureWP([-0.110108739875,-0.110108739875, # torso down
             -0.314110178149,0.0422208593044,-0.151027546604,0.396365865379,0.202001532531,-0.391391656508,-0.154661258599, # left arm
             -0.220503497346,0.337675132621,0.624805751243,1.16172088215,0.384001487506,0.896745360748,0.182850228809]) # right arm
        self.trajGoIdle.addPostureWP([-0.109914844184,-0.109914844184, # torso down
             -0.314923644821,0.0422696934788,-0.151193104755,0.39621674284,0.20183401121,-0.391155929596,-0.154656637274, # left arm
             -0.230249517377,0.345852992827,0.88586255411,0.893215173773,0.559154442999,0.826572075307,0.254374641592]) # right arm
        self.trajGoIdle.addPostureWP([-0.109921738922,-0.109921738922, # torso down
             -0.314834731069,0.042222340133,-0.151215678478,0.396394585017,0.201790267583,-0.391278406841,-0.154617421352, # left arm
             -0.270019264149,0.157806167509,0.887328352797,0.613659304297,0.556772075107,0.6985809056,0.373830708729]) # right arm
        self.trajGoIdle.addPostureWP([-0.110147121318,-0.110147121318, # torso down
             -0.314155649038,0.0423680526326,-0.151089097522,0.396174440851,0.201950379629,-0.391266751445,-0.154726310991, # left arm
             -0.263727657217,0.0292928830857,0.883743192191,0.333090296517,0.558229407793,0.558683993527,0.507622085692]) # right arm
        self.trajGoIdle.addPostureWP([-0.110096744088,-0.110096744088, # torso down
             -0.311052550826,0.0465252655301,-0.151238131785,0.396219594324,0.202053432985,-0.391290965438,-0.154890635658, # left arm
             -0.257235871749,0.0551459447258,0.599366258241,0.0432666473679,-0.328664738131,0.560906587474,0.499914477709]) # right arm
        self.trajGoIdle.addPostureWP([-0.110168244893,-0.110168244893, # torso down
             -0.312294930701,0.0454672581072,-0.151302176433,0.396142760985,0.201850634372,-0.391432024363,-0.154834278074, # left arm
             -0.270494472784,-0.0497059819383,0.114822026788,-0.0045554499616,-0.095347286215,0.562315674029,0.501668074017]) # right arm
        self.trajGoIdle.addPostureWP([-0.11042585257,-0.11042585257, # torso down
             -0.312141090201,0.0450946010403,-0.151239665265,0.396244668375,0.20197803806,-0.391388155881,-0.154770207092, # left arm
             -0.268384689433,-0.0450081014886,0.114735592811,-6.91863897046e-05,-0.224352654112,0.132910817624,0.0777436784938]) # right arm

    def createFSM(self):
        # define the states
        goToGraspState = TrajectoryState(self.dreamerInterface, self.trajGoGrasp)
        goToLocationState = TrajectoryState(self.dreamerInterface, self.trajGoLocation)
        goToIdleState = TrajectoryState(self.dreamerInterface, self.trajGoIdle)

        enablePowerGraspState = EnablePowerGraspState(self.dreamerInterface)
        disablePowerGraspState = DisablePowerGraspState(self.dreamerInterface)


        # wire the states into a FSM
        self.fsm = smach.StateMachine(outcomes=['exit'])
        with self.fsm:
            smach.StateMachine.add("GoToGrasp", goToGraspState,
                transitions={'done':'EnablePowerGraspState',
                             'exit':'exit'})
            smach.StateMachine.add("EnablePowerGraspState", enablePowerGraspState,
                transitions={'done':'GoToDropLocation',
                             'exit':'exit'})
            smach.StateMachine.add("GoToDropLocation", goToLocationState,
                transitions={'done':'DisablePowerGraspState',
                             'exit':'exit'})
            smach.StateMachine.add("DisablePowerGraspState", disablePowerGraspState,
                transitions={'done':'GoToIdle',
                             'exit':'exit'})
            smach.StateMachine.add("GoToIdle", goToIdleState,
                transitions={'done':'exit',
                             'exit':'exit'})




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
