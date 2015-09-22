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
        #self.trajGoGrasp.addRHCartWP([-0.061263905007396265, -0.18777870178877373, 0.777894344041236])
        #self.trajGoGrasp.addRHCartWP([-0.08042755693631962, -0.3114940849117037, 0.792344545233222])
        #self.trajGoGrasp.addRHCartWP([0.07047450934753619, -0.46218278567511617, 0.878041575593679])
        self.trajGoGrasp.addRHCartWP([0.1280866044013502, -0.5980521581850233, 1.068394262789365])
        self.trajGoGrasp.addRHCartWP([0.18509627032981188, -0.4550589628007665, 1.24629409504973])
        self.trajGoGrasp.addRHCartWP([0.21376691505023146, -0.3138971956641147, 1.22126887413446])
        self.trajGoGrasp.addRHCartWP([0.2097754229579469, -0.25303211212839655, 1.18084121032211])
        self.trajGoGrasp.addRHCartWP([0.23390235790401098, -0.2347535674008328, 1.103775879472771])
        self.trajGoGrasp.addRHCartWP([0.2426459118915965, -0.24344941687504615, 1.063783188431663])
        #self.trajGoGrasp.addRHCartWP([0.24244936474348786, -0.2389456664716109, 1.031461115664558])
        #self.trajGoGrasp.addRHCartWP([0.24134531672053958, -0.23669306303691837, 1.026910005484385])
        self.trajGoGrasp.addRHCartWP(obj_pos)


        # right arm Orientation	
        #self.trajGoGrasp.addRHOrientWP([0.0040363246429925, 0.9989954795970447, -0.04437403391897209, -0.00476287117239843])
        #self.trajGoGrasp.addRHOrientWP([0.11651802268236117, 0.9927301364097549, -0.015801389184588743, -0.02570491692068364])
        #self.trajGoGrasp.addRHOrientWP([0.18996441242495202, 0.9155783957722214, -0.27300710212676965, 0.2260461134272202])
        #self.trajGoGrasp.addRHOrientWP([0.14913428618532656, 0.6991772237140462, -0.6220670206419745, 0.31928482011071])
        self.trajGoGrasp.addRHOrientWP([0.0040363246429925, 0.9989954795970447, -0.04437403391897209, -0.00476287117239843]) #copy from init
        self.trajGoGrasp.addRHOrientWP([-0.09006014011097914, 0.42280214706421054, -0.4056922144406264, 0.805320645918968])
        self.trajGoGrasp.addRHOrientWP([-0.18826895909164512, 0.5960467565114382, -0.05290891217252465, 0.778770640244907])
        self.trajGoGrasp.addRHOrientWP([-0.23503913357165532, 0.690485590817773, 0.07001238767846728, 0.680503137490545])
        self.trajGoGrasp.addRHOrientWP([-0.1264906837428913, 0.7542880284012754, 0.05540779125155967, 0.641856412140126])
        self.trajGoGrasp.addRHOrientWP([-0.11361012551298492, 0.7882662909629717, -0.007895728635246377, 0.604703771595416])
        #self.trajGoGrasp.addRHOrientWP([-0.07308925010876131, 0.8139922406338245, -0.007264305193084221, 0.576213349009311])
        #self.trajGoGrasp.addRHOrientWP([-0.0765402999005207, 0.8185092139778348, -0.010136372973771535, 0.569281567475587])
        self.trajGoGrasp.addRHOrientWP(obj_ori)


        # left arm Position does not move
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        # trajectory
        #self.trajGoGrasp.addPostureWP([-0.112649727574,-0.112649727574, # torso down
        #    -0.314924416986,0.0866750834562,-0.147485456538,0.416361751514,0.18371038664,-0.401421455065,-0.165489784665, # left arm
        #     -0.257900517284,-0.0445306321636,-0.155262705175,0.0724392044318,0.243259024898,-0.0434396286619,0.0674969809341]) # right arm
        #self.trajGoGrasp.addPostureWP([-0.113047158574,-0.113047158574, # torso down
        #     -0.315606745987,0.0864774401881,-0.147510716326,0.416322090512,0.18383136279,-0.401368150949,-0.165506023411, # left arm
        #     -0.279478535789,0.177578088108,-0.155517358205,0.0429914014172,0.193670698577,-0.0426987618229,0.0634470097841]) # right arm
        #self.trajGoGrasp.addPostureWP([-0.112534057034,-0.112534057034, # torso down
        #     -0.31487632393,0.0865140032666,-0.147596051277,0.416447780333,0.183620782035,-0.401564828668,-0.16532594985, # left arm
        #     -0.113206421579,0.46936920907,0.0420029241152,0.298196947095,0.559270713236,-0.05397371202,0.062896852507]) # right arm
        self.trajGoGrasp.addPostureWP([-0.112308385693,-0.112308385693, # torso down
             -0.316035411963,0.0865307439811,-0.147629515872,0.416252460096,0.183737575963,-0.401375311234,-0.16525357628, # left arm
             -0.096036814061,0.70030204586,0.549989592087,0.718910763746,1.06214901639,-0.429909268845,0.446721354137]) # right arm
        self.trajGoGrasp.addPostureWP([-0.112171462333,-0.112171462333, # torso down
             -0.315603256731,0.0865393460258,-0.14738510027,0.41646876743,0.183640073421,-0.401350208682,-0.16540092764, # left arm
             0.0413505686786,0.348500228893,0.658240725785,1.72965298541,1.01622691145,-0.170947690155,0.198354353573]) # right arm
        self.trajGoGrasp.addPostureWP([-0.112260274402,-0.112260274402, # torso down
             -0.315895837232,0.0865797829732,-0.147706704399,0.416206270689,0.183678544594,-0.40133659256,-0.165255882969, # left arm
             -0.0972318518964,0.392240427147,0.116947287012,1.91037930608,0.787579522134,-0.222621369165,-0.038047350764]) # right arm
        self.trajGoGrasp.addPostureWP([-0.112002565868,-0.112002565868, # torso down
             -0.315825846912,0.0864940394881,-0.14756931052,0.416348094204,0.183748211305,-0.401328355133,-0.165241649213, # left arm
             -0.128687618394,0.506101455592,-0.221474796927,1.89568344458,0.776029920314,-0.243071119471,-0.0198405241864]) # right arm
        self.trajGoGrasp.addPostureWP([-0.111818793416,-0.111818793416, # torso down
             -0.315702373255,0.0865860829754,-0.147714803538,0.416201332269,0.183511874015,-0.40136261593,-0.165267216969, # left arm
             -0.1192850989,0.296186456863,-0.214124719791,1.62294326061,0.429645704315,-0.23737482238,-0.0185900145963]) # right arm
        self.trajGoGrasp.addPostureWP([-0.11197904755,-0.11197904755, # torso down
             -0.314909812664,0.0865738350022,-0.14767510257,0.416021818482,0.183543951451,-0.401243635534,-0.165236454438, # left arm
             -0.111049140914,0.223211934989,-0.149273635032,1.4591836246,0.423235170181,-0.222525387717,-0.0137931655928]) # right arm
        self.trajGoGrasp.addPostureWP([-0.112117116166,-0.112117116166, # torso down
             -0.31546789783,0.0864695058547,-0.147674147512,0.416053352298,0.183558964268,-0.401366462207,-0.165255564683, # left arm
             -0.110955387425,0.155524645838,-0.109723776121,1.34584527487,0.290315512879,-0.209142600609,-0.0179012692189]) # right arm
        #self.trajGoGrasp.addPostureWP([-0.112020615065,-0.112020615065, # torso down
        #     -0.314768600317,0.0866284009373,-0.147597050877,0.416286951572,0.183732740457,-0.401384142523,-0.165355313533, # left arm
        #     -0.113592208008,0.147548744092,-0.110058517873,1.33365286966,0.291365646196,-0.211126430395,-0.0188171779067]) # right arm



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
        self.trajGoLocation.addRHCartWP([0.28883824418290543, -0.14405629503939205, 0.999584811185456])
        self.trajGoLocation.addRHCartWP([0.2772515572866687, -0.06550500032876867, 1.015580822684924])
        self.trajGoLocation.addRHCartWP([0.27521028676187226, -0.012900374916451662, 1.041249066881016])
        self.trajGoLocation.addRHCartWP([0.2716053128570153, -0.020591395546707494, 1.00369923204009])

        # right arm Orientation
        self.trajGoLocation.addRHOrientWP([-0.3376460043882763, 0.6715982043955052, 0.08237679282442024, 0.654343252106513])
        self.trajGoLocation.addRHOrientWP([-0.4475142222321699, 0.6072542882881371, 0.18322193332655326, 0.630399058853742])
        self.trajGoLocation.addRHOrientWP([-0.4783712340814896, 0.57760352868288, 0.2755212230296299, 0.601351130136707])
        self.trajGoLocation.addRHOrientWP([-0.47311275134691994, 0.655005207183941, 0.2203445148265389, 0.54642547329048])


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
        self.trajGoLocation.addPostureWP([-0.0663117345843,-0.0663117345843, # torso down
             -0.278287560766,0.086067037334,-0.110558948578,0.539100025607,0.155069011802,-0.368305953131,-0.190699397664, # left arm
             0.0798877599587,-0.0840737354825,-0.152953479153,1.10580355632,0.310863271514,0.157436091974,-0.506464192244]) # right arm
        self.trajGoLocation.addPostureWP([-0.0662534533853,-0.0662534533853, # torso down
             -0.277350771609,0.0860679282623,-0.110472109103,0.539041581354,0.155074405309,-0.368290884565,-0.190794939774, # left arm
             0.109102862192,-0.11244546775,-0.431956221484,1.11494694164,0.385903288468,0.198866641767,-0.572591642859]) # right arm
        self.trajGoLocation.addPostureWP([-0.0656793929371,-0.0656793929371, # torso down
             -0.277188133885,0.0859684895316,-0.110772546359,0.539044921928,0.155028556167,-0.368386520124,-0.190593082396, # left arm
             0.20051037886,-0.124479721422,-0.65585392054,1.10829452975,0.395592728694,0.173719618292,-0.535368903705]) # right arm
        self.trajGoLocation.addPostureWP([-0.0656571719253,-0.0656571719253, # torso down
             -0.277395383862,0.0861746292738,-0.110544851512,0.539151624809,0.155080092715,-0.368286019862,-0.190575150443, # left arm
             0.199906896463,-0.124447569203,-0.655740456969,0.970132249503,0.485733308607,0.131035214385,-0.491288488393]) # right arm
        
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
        self.trajGoIdle.addRHCartWP([0.2732797786391307, -0.011781985161585753, 1.053449248228718])
        self.trajGoIdle.addRHCartWP([0.2866419713758848, -0.13475678591282125, 1.016968588688599])
        self.trajGoIdle.addRHCartWP([0.2364098732210676, -0.24508974889203625, 0.968192491663854])
        self.trajGoIdle.addRHCartWP([0.15823706410292315, -0.33813623247638386, 0.93566112558224])
        self.trajGoIdle.addRHCartWP([0.05733644311166543, -0.370075341103974, 0.905560345526331])
        self.trajGoIdle.addRHCartWP([-0.0031595827411109215, -0.32738468811051774, 0.83698754518787])
        self.trajGoIdle.addRHCartWP([-0.052568894462058924, -0.27463208735477374, 0.80040609625637])
        self.trajGoIdle.addRHCartWP([-0.08715124957487375, -0.22696789675647344, 0.78823545634331])


        # right arm Orientation 
        self.trajGoIdle.addRHOrientWP([0.5191511749493414, -0.5530606801453738, -0.24438924335579507, -0.604052844838791])
        self.trajGoIdle.addRHOrientWP([-0.46962201425932115, 0.4806107095912671, 0.199305869613428, 0.713264102480162])
        self.trajGoIdle.addRHOrientWP([-0.3693163619699166, 0.4780650050047376, 0.1499428451645672, 0.782672612881708])
        self.trajGoIdle.addRHOrientWP([-0.22679365149735858, 0.5016173458771365, 0.06675035337162827, 0.832159280595028])
        self.trajGoIdle.addRHOrientWP([-0.14796871443347098, 0.5852958884182097, -0.002568149839317317, 0.797199715978403])
        self.trajGoIdle.addRHOrientWP([-0.1621679226247192, 0.7295241056045406, -0.20109888382426602, 0.633289336036943])
        self.trajGoIdle.addRHOrientWP([-0.16214107265036964, 0.8128904873945442, -0.3925644854153719, 0.3985128013675498])
        self.trajGoIdle.addRHOrientWP([-0.1537214557569671, 0.8350911417412259, -0.488612600061028, 0.2006245899357714])


 
        # left arm Position does not move
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm Orientation does not move
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])

        # trajectory
        self.trajGoIdle.addPostureWP([-0.0658933454842,-0.0658933454842, # torso down
             -0.27726932896,0.0859560441633,-0.1105738978,0.539102918217,0.155124323905,-0.368441346217,-0.190648521287, # left arm
             0.19315663738,-0.120984460862,-0.657077118246,1.16276149382,0.495210674225,0.167789177992,-0.563579903976]) # right arm
        self.trajGoIdle.addPostureWP([-0.0664059496526,-0.0664059496526, # torso down
             -0.277594116452,0.086083220478,-0.110523428847,0.539155516181,0.154885668161,-0.368368836821,-0.190708901317, # left arm
             0.109711100239,0.0464786838164,-0.393996088082,1.16446662445,0.613996637159,0.432553239165,-0.825521223386]) # right arm
        self.trajGoIdle.addPostureWP([-0.0667583557335,-0.0667583557335, # torso down
             -0.27828872837,0.0860112201771,-0.110597243355,0.538980186479,0.15506960637,-0.368483851315,-0.190633324568, # left arm
             -0.0851957546712,0.142719537914,-0.10204379143,1.16389957858,0.703275896209,0.614684315699,-1.0070286928]) # right arm
        self.trajGoIdle.addPostureWP([-0.0668191745596,-0.0668191745596, # torso down
             -0.277616989924,0.0861100534157,-0.110455121632,0.538650536342,0.155158962432,-0.368238061377,-0.19060072644, # left arm
             -0.266120509407,0.152352526223,0.247766171553,1.14834546613,0.798597879491,0.717378944483,-1.11266498839]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669997865479,-0.0669997865479, # torso down
             -0.278308708136,0.0860346985709,-0.110547338161,0.539107237914,0.155178321208,-0.368513690572,-0.190656750261, # left arm
             -0.458932654121,0.115011213128,0.440543461974,1.12536386762,0.785362147435,0.714493531282,-1.10608034612]) # right arm
        self.trajGoIdle.addPostureWP([-0.0670090973714,-0.0670090973714, # torso down
             -0.278539324143,0.0861031232014,-0.110505908326,0.539088282133,0.155116953222,-0.368267599971,-0.19084694381, # left arm
             -0.425246503101,0.0545721987857,0.442388540068,0.773963430602,0.69607075193,0.539697367894,-0.905284367816]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669876810473,-0.0669876810473, # torso down
             -0.277582286456,0.0860492615745,-0.110500851816,0.538915679189,0.155183908404,-0.368344226949,-0.190571434924, # left arm
             -0.378816463076,0.0049036750284,0.438210923442,0.472029428606,0.671445895022,0.284673962644,-0.659176435363]) # right arm
        self.trajGoIdle.addPostureWP([-0.0671572217398,-0.0671572217398, # torso down
             -0.278566240185,0.0860228028407,-0.110587824492,0.538945889011,0.155259931144,-0.368483057565,-0.190552623774, # left arm
             -0.354552329343,-0.0445709484692,0.436526878184,0.283318176516,0.670772308,0.102201180003,-0.480077185228]) # right arm

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
