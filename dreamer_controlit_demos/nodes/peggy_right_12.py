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
    #    obj_pos = map(float, sys.argv[1:4])
    #    obj_ori = map(float, sys.argv[4:])

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
        self.trajGoGrasp.addRHCartWP([-0.09473078575929389, -0.20822564550971, 0.772323495869846])
        self.trajGoGrasp.addRHCartWP([-0.09504168148829503, -0.3940192030453265, 0.812749165213753])
        self.trajGoGrasp.addRHCartWP([-0.102233936942719, -0.49451038498576755, 0.866238952322406])
        self.trajGoGrasp.addRHCartWP([-0.11967374612061657, -0.571056777886559, 0.928915659791564])
        self.trajGoGrasp.addRHCartWP([-0.10166468670298712, -0.6719141673386102, 1.0596700338825])
        self.trajGoGrasp.addRHCartWP([0.09947194702905834, -0.6765411991582408, 1.2353619995333])
        self.trajGoGrasp.addRHCartWP([0.14924125979293434, -0.4831262196488924, 1.175160444398975])
        self.trajGoGrasp.addRHCartWP([0.13881015064622967, -0.3703351343798719, 1.13555879421501])
        self.trajGoGrasp.addRHCartWP([0.1346278824696966, -0.22975885130598253, 1.111029010020508])
        self.trajGoGrasp.addRHCartWP([0.20231312088099884, -0.1968861962744175, 1.03725309492427])
        self.trajGoGrasp.addRHCartWP([0.20005707861233782, -0.194072876288009, 1.008541676154782])
        #self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        self.trajGoGrasp.addRHOrientWP([-0.12384223040225073, 0.9776359927589541, 0.0392613755265698, 0.1653768787435894])
        self.trajGoGrasp.addRHOrientWP([0.04215288485438046, 0.9846828184585636, 0.06839550221104486, 0.1547415154645053])
        self.trajGoGrasp.addRHOrientWP([0.13950973019123014, 0.9767115370254504, 0.08533311385968372, 0.138887970329912])
        self.trajGoGrasp.addRHOrientWP([0.22143093590565532, 0.9635869389746344, 0.10170282764104167, 0.110113970554404])
        self.trajGoGrasp.addRHOrientWP([0.3781208985376035, 0.9106047943484852, 0.06652881306527617, 0.1529621248243957])
        self.trajGoGrasp.addRHOrientWP([0.42655644691723066, 0.7301548580006972, 0.15685072325265192, 0.510216945569675])
        self.trajGoGrasp.addRHOrientWP([0.2347223621462274, 0.5971623515617582, 0.32229157530415126, 0.696010545231675])
        self.trajGoGrasp.addRHOrientWP([-0.0888121808855495, 0.619226035289227, 0.26328090251394537, 0.734407707011425])
        self.trajGoGrasp.addRHOrientWP([-0.10321595943857707, 0.7043541018266666, -0.0326037471892106, 0.701547404404253])
        self.trajGoGrasp.addRHOrientWP([0.020620398223877488, 0.7145379223448409, 0.013888867673367965, 0.699154815518679])
        self.trajGoGrasp.addRHOrientWP([0.008995528844771875, 0.7492821812954445, -0.00647852717983082, 0.662158079267725])
        #self.trajGoGrasp.addRHOrientWP(obj_ori)


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


        # trajectory
        self.trajGoGrasp.addPostureWP([-0.153827720821,-0.153827720821, # torso down
             -0.329062172414,0.110135223995,-0.241696356301,0.47417742698,0.264108856955,-0.44833610171,-0.140138683562, # left arm
             -0.299477374182,-0.0154186239256,-0.0495444429272,-0.0441303609373,0.0066328032908,0.375023154679,-0.230114209352]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153687575207,-0.153687575207, # torso down
             -0.328743019215,0.110153688529,-0.241935879637,0.473866331574,0.26386328547,-0.448386272002,-0.139998137923, # left arm
             -0.297945074834,0.327724322602,-0.049740608956,-0.0462576921282,0.00067153942554,0.374304028887,-0.23499089365]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153560934488,-0.153560934488, # torso down
             -0.328755399592,0.110190193533,-0.241861616889,0.473903557227,0.263891950766,-0.448243391009,-0.140137694576, # left arm
             -0.309811082619,0.532529988293,-0.0498459570852,-0.0507177502626,-0.00324502885478,0.375867970826,-0.239608347785]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153259723293,-0.153259723293, # torso down
             -0.329203073532,0.110258982082,-0.241747862742,0.473845598351,0.263960887955,-0.448270281235,-0.14011066146, # left arm
             -0.352377129706,0.709435131609,-0.0498232751562,-0.0523548322822,-0.00141872331822,0.377392391828,-0.243265640042]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153047444051,-0.153047444051, # torso down
             -0.328714491294,0.110300630642,-0.241920466546,0.474009071362,0.263943056469,-0.448254714665,-0.139983373711, # left arm
             -0.307113829838,1.00737666465,-0.0987863413443,-0.0529445184838,0.173842489174,0.381124909043,-0.241729610532]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153362552754,-0.153362552754, # torso down
             -0.329375620565,0.11024732272,-0.241851170958,0.473806371505,0.263816174074,-0.448200718668,-0.14004639348, # left arm
             -0.0131973518769,1.20400142761,0.061954259532,0.577487152554,0.165227778295,0.337236261519,-0.283906349274]) # right arm
        self.trajGoGrasp.addPostureWP([-0.15313979185,-0.15313979185, # torso down
             -0.348245265126,0.110285905549,-0.241740570602,0.451815435254,0.263802439533,-0.447769464869,-0.140185974845, # left arm
             -0.493961457196,0.814677606485,0.347183230648,1.53048051993,0.0821593633146,0.336051415546,-0.283914301204]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153008057046,-0.153008057046, # torso down
             -0.348161848237,0.110316655412,-0.241622245943,0.45192039832,0.263955567809,-0.447807361814,-0.140091211318, # left arm
             -0.594789762435,0.86791302086,-0.0104146142227,1.78744523818,0.582939050412,0.330007760871,-0.276436057166]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153572792658,-0.153572792658, # torso down
             -0.349031296901,0.110336045338,-0.24165193388,0.451620954045,0.263884605534,-0.447781702683,-0.140169819903, # left arm
             -0.483694880979,0.297031912611,-0.150721139423,1.93418591739,0.454463376609,-0.0970269984142,0.164248206784]) # right arm
        self.trajGoGrasp.addPostureWP([-0.15338854945,-0.15338854945, # torso down
             -0.347950035502,0.110268423019,-0.241588365048,0.451943789839,0.263977940257,-0.44748216004,-0.140128198763, # left arm
             -0.255882742998,0.0484900473709,-0.122672767135,1.47086255265,0.00854761027394,0.0276229008471,0.128682628559]) # right arm
        self.trajGoGrasp.addPostureWP([-0.153420027581,-0.153420027581, # torso down
             -0.348050530978,0.110406208481,-0.241637226292,0.451829751977,0.263856798437,-0.447448488541,-0.140196934554, # left arm
             -0.254187033929,0.0310678796476,-0.118631038105,1.36533788282,0.0490832039362,0.0341368922587,0.13191856346]) # right arm


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
        self.trajGoLocation.addRHCartWP([0.20815614740944108, -0.20318220088621045, 1.081092851710467])
        self.trajGoLocation.addRHCartWP([0.20133546417913162, -0.16121084488478876, 1.082842499771597])
        self.trajGoLocation.addRHCartWP([0.1896457227202583, -0.13325366042678888, 1.03093090221414])
        self.trajGoLocation.addRHCartWP([0.1732937528225132, -0.13396720505688386, 0.994389792818594])
        
        # right arm Orientation
        self.trajGoLocation.addRHOrientWP([-0.050226630549623646, 0.6188659053170257, 0.07527557631660013, 0.780266534224026])
        self.trajGoLocation.addRHOrientWP([-0.18004265085997573, 0.6370201313837719, 0.18101250951187592, 0.727340681856396])
        self.trajGoLocation.addRHOrientWP([-0.18621647680025183, 0.6762388866087569, 0.17430567938124028, 0.691116431682603])
        self.trajGoLocation.addRHOrientWP([-0.2153272558388692, 0.6975536600311728, 0.12928131363472975, 0.671073324022715])


        # left arm Position does not move
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        self.trajGoLocation.addPostureWP([-0.153507733386,-0.153507733386, # torso down
             -0.347984589971,0.110216380717,-0.241696577523,0.451857490776,0.264089164156,-0.447425187714,-0.140248294689, # left arm
             -0.229714674138,0.0824522784885,-0.126108736127,1.59957903277,0.0471159399061,0.132595837725,-0.0494620033936]) # right arm
        self.trajGoLocation.addPostureWP([-0.15320196612,-0.15320196612, # torso down
             -0.347987826399,0.110346185389,-0.241590685386,0.451953802681,0.263803819407,-0.447551089049,-0.140187346913, # left arm
             -0.237989942933,0.0194655247703,-0.215194166314,1.6204178565,0.0371143357733,0.0155218842854,-0.299913248171]) # right arm
        self.trajGoLocation.addPostureWP([-0.153224821054,-0.153224821054, # torso down
             -0.348131627638,0.110251114914,-0.241732181038,0.451791944806,0.263957107327,-0.447417100724,-0.140258809237, # left arm
             -0.266444632627,-0.0120349002355,-0.286687184105,1.45942560123,0.015705135973,0.0906351206599,-0.229617066887]) # right arm
        self.trajGoLocation.addPostureWP([-0.15333529171,-0.15333529171, # torso down
             -0.34875242194,0.110400891108,-0.241602247499,0.451886237431,0.263873431859,-0.447423639728,-0.140041028989, # left arm
             -0.305782605054,-0.00727618838567,-0.293197729081,1.36661144371,0.118036055322,0.168251600287,-0.221865521599]) # right arm
        
        
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
        self.trajGoIdle.addRHCartWP([0.17438051148547712, -0.16583380315430551, 1.082567862872993])
        self.trajGoIdle.addRHCartWP([0.18671519177547274, -0.29010721192059535, 1.07113449520243])
        self.trajGoIdle.addRHCartWP([0.1287389235667348, -0.4290096882263904, 1.133764254287488])
        self.trajGoIdle.addRHCartWP([0.03950539758603787, -0.4946747529285347, 1.027228878852787])
        self.trajGoIdle.addRHCartWP([-0.018772552461195243, -0.4409056433362983, 0.879088395783766])
        self.trajGoIdle.addRHCartWP([-0.04647093226727902, -0.3381514377235927, 0.808940149824313])
        self.trajGoIdle.addRHCartWP([-0.08549033061998956, -0.2235962023451242, 0.773281968376306])
        self.trajGoIdle.addRHCartWP([-0.056683082120331005, -0.20827016483710944, 0.773246282970213])
        self.trajGoIdle.addRHCartWP([-0.05861508738534457, -0.19785555063956684, 0.772724452746076])


        # right arm Orientation 
        self.trajGoIdle.addRHOrientWP([-0.29924306165418124, 0.4615380266374902, 0.16922499275254824, 0.817801407340011])
        self.trajGoIdle.addRHOrientWP([-0.1516879714824705, 0.5135417105932378, 0.08350513537641022, 0.840412138866563])
        self.trajGoIdle.addRHOrientWP([0.008993876874136462, 0.3308418705050461, -0.1466076447615481, 0.932185048902291])
        self.trajGoIdle.addRHOrientWP([0.16746360005092478, 0.3989058204999882, -0.36202656619988205, 0.825691742961462])
        self.trajGoIdle.addRHOrientWP([0.15844726671549772, 0.6716073993509497, -0.38494495366375503, 0.612907291077738])
        self.trajGoIdle.addRHOrientWP([0.07294738136943908, 0.7499587285584202, -0.5089633832849599, 0.416169267829216])
        self.trajGoIdle.addRHOrientWP([-0.015539408934197247, 0.8325833287943651, -0.525066533275163, 0.1756948007713152])
        self.trajGoIdle.addRHOrientWP([-0.11056762677535222, 0.926686141269133, 0.3514182692879097, 0.074382763458218])
        self.trajGoIdle.addRHOrientWP([-0.11676641773034312, 0.9866497175702347, 0.014334873914959016, 0.112616383794016])
 
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

        # trajectory
        self.trajGoIdle.addPostureWP([-0.153297053744,-0.153297053744, # torso down
             -0.347879115408,0.110370142069,-0.241522810618,0.451967311673,0.263976895349,-0.447434879845,-0.140075857532, # left arm
             -0.318490912116,0.106372253221,-0.276101327766,1.71189826343,0.325998539962,0.395790751801,-0.44791283438]) # right arm
        self.trajGoIdle.addPostureWP([-0.153318764112,-0.153318764112, # torso down
             -0.351660934535,0.110262427956,-0.241562753684,0.45194276275,0.263861690795,-0.447443960032,-0.140131375049, # left arm
             -0.318179548814,0.198556615839,0.0761533152334,1.60140247712,0.415348626818,0.394241691508,-0.452665290509]) # right arm
        self.trajGoIdle.addPostureWP([-0.153393242018,-0.153393242018, # torso down
             -0.351518966079,0.11037059852,-0.241699738096,0.451759357314,0.263992870086,-0.447337953396,-0.140081158349, # left arm
             -0.296260522068,0.1774710397,0.672150383505,1.72039584344,0.755445504485,0.403279283783,-0.445751170051]) # right arm
        self.trajGoIdle.addPostureWP([-0.15332644122,-0.15332644122, # torso down
             -0.351205222791,0.110161787748,-0.241696586711,0.45180551921,0.263894392171,-0.447598689511,-0.140220457313, # left arm
             -0.301536232122,0.10518753457,1.08333085861,1.37321987524,0.688242139758,0.364332390015,-0.429147715019]) # right arm
        self.trajGoIdle.addPostureWP([-0.153613142589,-0.153613142589, # torso down
             -0.351188020344,0.11021431338,-0.241733887266,0.451816678083,0.26398278644,-0.447386047409,-0.140288273515, # left arm
             -0.364672609036,0.0837959606679,1.08280494746,0.836560719793,0.305579907229,0.392995623626,-0.453906916102]) # right arm
        self.trajGoIdle.addPostureWP([-0.153697324273,-0.153697324273, # torso down
             -0.351009658207,0.110254141863,-0.241780782581,0.451919785042,0.263937230308,-0.447540808249,-0.139945133706, # left arm
             -0.360268021964,-0.0304795735419,1.0822443535,0.573931306526,0.243322089252,0.261169873906,-0.330679166205]) # right arm
        self.trajGoIdle.addPostureWP([-0.153704886979,-0.153704886979, # torso down
             -0.350753571706,0.110179537619,-0.241681857116,0.451840066846,0.263899456017,-0.447627527737,-0.140222377498, # left arm
             -0.361141191361,-0.110847565553,1.08131021864,0.263774083527,0.0684526619991,0.139689685841,-0.206039622735]) # right arm
        self.trajGoIdle.addPostureWP([-0.153461617965,-0.153461617965, # torso down
             -0.350665647556,0.110201133386,-0.241669790746,0.446036382193,0.263901756845,-0.447183666881,-0.140092143226, # left arm
             -0.28748351451,-0.0214674577016,0.15398889934,0.0712355767521,-0.853267204959,0.141892410719,-0.207953929113]) # right arm
        self.trajGoIdle.addPostureWP([-0.153628319324,-0.153628319324, # torso down
             -0.350954910885,0.110017087939,-0.241391138157,0.44635490788,0.2640488586,-0.447266711535,-0.140157688158, # left arm
             -0.291603533916,-0.0408242770715,0.168906655845,0.0714131598892,-0.177453331704,0.141338026545,-0.206523431102]) # right arm

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
