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
TIME_GO_TO_GRASP = 10.0
TIME_TEST_TRAJECTORY = 2.0
TIME_GO_TO_LOCATION = 3.0
TIME_DROP = 2.0
TIME_GO_TO_IDLE = 10.0


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
        self.trajGoGrasp.setInitLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.setInitRHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.setInitPostureWP(DEFAULT_POSTURE)


        self.trajGoGrasp.addRHCartWP([-0.07024262305122916, -0.3778760129931865, 0.81526463808579])
        self.trajGoGrasp.addRHCartWP([-0.06157122153345717, -0.5088570496998758, 0.88467929061838])
        self.trajGoGrasp.addRHCartWP([-0.03619758851840669, -0.5981174618147724, 0.965752050778073])
        self.trajGoGrasp.addRHCartWP([0.10482378870185628, -0.6419503915716834, 1.108285660413317])
        self.trajGoGrasp.addRHCartWP([0.20210817764043548, -0.5268676698196983, 1.15351627009211])
        self.trajGoGrasp.addRHCartWP([0.22973113347951693, -0.38420238326977346, 1.119688443072485])
        self.trajGoGrasp.addRHCartWP([0.20587241896563152, -0.24712910065251942, 1.070480723607068])
        self.trajGoGrasp.addRHCartWP([0.2039677484264281, -0.213417885604496, 1.01710856638995])
        self.trajGoGrasp.addRHCartWP([0.20201096802150922, -0.21739456121304582, 1.003816676455791])




        self.trajGoGrasp.addRHOrientWP([0.12437570225149638, 0.9886087198046791, -0.04638179297347371, 0.0709380934066201])
        self.trajGoGrasp.addRHOrientWP([0.25081418146807943, 0.9645526120109505, -0.020573821943214006, 0.0794180262157083])
        self.trajGoGrasp.addRHOrientWP([0.3477176492895149, 0.9237283581136132, -0.07333804685418042, 0.1429681351662542])
        self.trajGoGrasp.addRHOrientWP([0.3862746261023898, 0.7996059406991121, -0.02711603342216955, 0.459006507099403])
        self.trajGoGrasp.addRHOrientWP([0.2827468065665801, 0.6845044287763354, 0.09318588188691157, 0.665450465308231])
        self.trajGoGrasp.addRHOrientWP([0.09354241690040364, 0.6553629082944883, 0.15111306874102373, 0.73410769995137])
        self.trajGoGrasp.addRHOrientWP([-0.012536103792510307, 0.8007708470457688, 0.07834712984250916, 0.593692364671086])
        self.trajGoGrasp.addRHOrientWP([0.011493595489784588, 0.7578822081545429, 0.030055664197746478, 0.651597354871199])
        self.trajGoGrasp.addRHOrientWP([0.020833672825148752, 0.7721109066855332, 0.02589796853844556, 0.634617996183089])


        # left arm doesn't move
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoGrasp.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm doesn't move
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        self.trajGoGrasp.addPostureWP([-0.088976914431,-0.088976914431, # torso down
            -0.286514007146,0.108137376656,-0.167299425944,0.535480609083,0.198553938162,-0.431220781867,-0.172013479135, # left arm
            -0.271816563517,0.297054349871,0.0167555717902,0.0990820513372,0.126962591092,0.114390955322,-0.053029724809]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0889291904195,-0.0889291904195, # torso down
            -0.286233849534,0.107955662752,-0.167374279793,0.535286567283,0.198696033887,-0.431163194627,-0.172095443091, # left arm
            -0.268459139897,0.565043166671,0.0161139394864,0.101363905856,0.120226345711,0.114280904528,-0.0585666206075]) # right arm
        self.trajGoGrasp.addPostureWP([-0.088661160532,-0.088661160532, # torso down
            -0.286465712818,0.108106265738,-0.167281290207,0.535387669376,0.198534605145,-0.431052015344,-0.172082605697, # left arm
            -0.251585675858,0.783111098046,0.0214874275513,0.144441050912,0.282378334996,0.111027215348,-0.0625131072048]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0888220899141,-0.0888220899141, # torso down
            -0.286613940818,0.108172409493,-0.167301416373,0.535431189864,0.198660344371,-0.431049723241,-0.172080112468, # left arm
            -0.168839642953,0.929970115342,0.350322692484,0.621534601287,0.193094881387,0.165612933707,-0.133080542358]) # right arm
        self.trajGoGrasp.addPostureWP([-0.088398975285,-0.088398975285, # torso down
            -0.287058641529,0.108164448993,-0.167306754051,0.535449368804,0.198680161876,-0.431171832324,-0.172235191893, # left arm
            -0.162750477301,0.77448252313,0.313627166386,1.22337712422,0.197646994154,0.162351609983,-0.131737606324]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0883916296311,-0.0883916296311, # torso down
            -0.287688384025,0.108098128161,-0.167146992639,0.535345255388,0.198657494981,-0.431062572625,-0.171974097973, # left arm
            -0.170230799264,0.582853878279,0.0502942900658,1.51918942021,0.245245126352,0.15521108732,-0.125232689146]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0887729640568,-0.0887729640568, # torso down
            -0.287963451396,0.108136603593,-0.167458396526,0.535245515838,0.198632374098,-0.431113781156,-0.172153750528, # left arm
            -0.194855800391,0.485055079434,-0.357460751689,1.63460760126,0.441832051013,-0.18501124807,0.218021079535]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0885143607739,-0.0885143607739, # torso down
            -0.287200616868,0.108190252802,-0.167160838657,0.535455155473,0.198589648516,-0.431058045261,-0.172117283896, # left arm
            -0.200063554416,0.288737813682,-0.339549036774,1.4605865345,0.256477335503,0.0620279728635,0.273326825043]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0883616196612,-0.0883616196612, # torso down
            -0.287035299331,0.108117599248,-0.167206496176,0.535474729159,0.198508454676,-0.431003216479,-0.17204529741, # left arm
            -0.200856055583,0.283307662785,-0.334871616871,1.4096085806,0.258906841067,0.0643045593302,0.275104782436]) # right arm



    # ==============================================================================================
    # Define the GoTOLocation trajectory
        self.trajGoLocation = Trajectory.Trajectory("GoToLocation", TIME_GO_TO_LOCATION)

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

 

        self.trajGoLocation.addRHCartWP([0.23056402714289576, -0.19895934988248798, 1.063697753132595])
        self.trajGoLocation.addRHCartWP([0.21047175508992497, -0.12243323390356464, 1.055585365081222])
        self.trajGoLocation.addRHCartWP([0.205612988541002, -0.11115853120815658, 1.05039318786393])
        self.trajGoLocation.addRHCartWP([0.19611549558992444, -0.12457365240585423, 0.995901594126343])
        #self.trajGoLocation.addRHCartWP([0.20478038259434617, -0.1166073971523455, 1.05035342651171])
        


        self.trajGoLocation.addRHOrientWP([-0.05181837687465661, 0.7151304877036943, -0.01911197219757728, 0.696805549557027])
        self.trajGoLocation.addRHOrientWP([-0.1057189540493638, 0.7062406306180067, 0.06737854837987378, 0.696783901677707])
        self.trajGoLocation.addRHOrientWP([-0.11625147856725133, 0.7118760961253141, 0.07192721733669559, 0.688871898760945])
        self.trajGoLocation.addRHOrientWP([-0.10610225115625511, 0.7444686118412538, 0.1263059551008173, 0.646958734378632])
        #self.trajGoLocation.addRHOrientWP([-0.13146221291520213, 0.6804303497243245, 0.11581478420146711, 0.711561073633064])


        # left arm doesn't move
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        #self.trajGoLocation.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm doesn't move
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        #self.trajGoLocation.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        self.trajGoLocation.addPostureWP([-0.0889257150772,-0.0889257150772, # torso down
            -0.287191284519,0.108100054806,-0.167356399231,0.535314788613,0.198595917923,-0.431144611476,-0.172020753783, # left arm
            -0.109485396247,0.276396866128,-0.353301240218,1.54808404226,0.386776300183,0.0574344092668,0.280746810102]) # right arm
        self.trajGoLocation.addPostureWP([-0.0888284385964,-0.0888284385964, # torso down
            -0.287329620963,0.10816343411,-0.167290429668,0.5354029497,0.198659466627,-0.431105571167,-0.171999862463, # left arm
            -0.101461077336,0.175698326522,-0.541324525225,1.54719431501,0.245342815593,0.0570858258143,0.279416720974]) # right arm
        self.trajGoLocation.addPostureWP([-0.088552963537,-0.088552963537, # torso down
            -0.287820124422,0.107922478275,-0.167241567255,0.535436888651,0.198691790378,-0.431064118559,-0.172050782159, # left arm
            -0.10752861537,0.153444119474,-0.563668021136,1.53285295917,0.237579003725,0.0557076986705,0.279072060472]) # right arm
        self.trajGoLocation.addPostureWP([-0.0883521704329,-0.0883521704329, # torso down
            -0.287853711552,0.108270222342,-0.16715719321,0.53538877608,0.19869913023,-0.431219240561,-0.172008065098, # left arm
            -0.142325708654,0.14175219529,-0.534255924055,1.35690277141,0.162062432629,0.114492480454,0.164056193046]) # right arm
        #self.trajGoLocation.addPostureWP([-0.0886218843961,-0.0886218843961, # torso down
        #    -0.287820780168,0.108312005286,-0.167399346108,0.535370043955,0.198797109976,-0.431142634786,-0.171952702008, # left arm
        #    -0.127071877454,0.146327069036,-0.529778973306,1.54372722199,0.157924079334,0.106145375128,0.167506141143]) # right arm


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
        self.trajGoIdle.addRHCartWP([0.23806006449534947, -0.16812808429318704, 1.08977442798251])
        self.trajGoIdle.addRHCartWP([0.24678239048155023, -0.2445280348210488, 1.093925295017103])
        self.trajGoIdle.addRHCartWP([0.23175663723007722, -0.34317208305306757, 1.085831091538392])
        self.trajGoIdle.addRHCartWP([0.1957634933653613, -0.4228106980352024, 1.08590972231969])
        self.trajGoIdle.addRHCartWP([0.13577429601771288, -0.4976886386836021, 1.055015127397333])
        self.trajGoIdle.addRHCartWP([0.034022985578650286, -0.5500466314741769, 1.021705335659044])
        self.trajGoIdle.addRHCartWP([-0.013072050180572442, -0.43403671618190826, 0.851903641868219])
        self.trajGoIdle.addRHCartWP([-0.046992563669959894, -0.33078964089801405, 0.802442249215622])
        self.trajGoIdle.addRHCartWP([-0.0690257483812168, -0.23790979841313592, 0.783805030222298])
        self.trajGoIdle.addRHCartWP([-0.07977168995910655, -0.26297665853730173, 0.788066034782979])
        self.trajGoIdle.addRHCartWP([-0.06938358563815773, -0.22269637578042162, 0.783576126666514])
        self.trajGoIdle.addRHCartWP([-0.06132186654886571, -0.18859770420957386, 0.782043031563492])

        # right arm orientation
        self.trajGoIdle.addRHOrientWP([-0.05804422045319227, 0.6892886456791986, 0.027319761542346006, 0.721640950916026])
        self.trajGoIdle.addRHOrientWP([0.0420907332329711, 0.6803451262341618, -0.06910491683285373, 0.728411552527043])
        self.trajGoIdle.addRHOrientWP([0.14622895901099847, 0.6628024636557794, -0.18053506440957415, 0.71184062558765])
        self.trajGoIdle.addRHOrientWP([0.261850482405012, 0.6003411654426861, -0.2770590930145522, 0.703038454792576])
        self.trajGoIdle.addRHOrientWP([0.43300177832457637, 0.5018771263307832, -0.3924482408379325, 0.63766228389087])
        self.trajGoIdle.addRHOrientWP([0.5075736779971044, 0.44556192753897006, -0.5235165625260111, 0.519397669318219])
        self.trajGoIdle.addRHOrientWP([0.39757569878534316, 0.6954334888209789, -0.5368653197077603, 0.264729021561127])
        self.trajGoIdle.addRHOrientWP([0.3300880088361215, 0.8665539036627806, -0.33955591924444484, 0.1575690838186889])
        self.trajGoIdle.addRHOrientWP([0.20920290764559468, 0.959904736921256, -0.15243802801683287, 0.1076089544745338])
        self.trajGoIdle.addRHOrientWP([0.1849490259271349, 0.9750419662749787, 0.04331816595489282, 0.1149371928904289])
        self.trajGoIdle.addRHOrientWP([0.10848498616024886, 0.9502729976186355, 0.24820287279699615, 0.1536475567980877])
        self.trajGoIdle.addRHOrientWP([0.13695786485364428, 0.9798541844580554, -0.08929968522778944, 0.1146903948593245])


        # left arm doesn't move
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
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])
        self.trajGoIdle.addLHCartWP([0.033912978219317776, 0.29726881641499886, 0.82])


        # left arm doesn't move
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
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoIdle.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        self.trajGoIdle.addPostureWP([-0.0669569934005,-0.0669569934005, # torso down
            -0.277346717644,0.0625617953671,-0.122911801737,0.545553771345,0.172198200909,-0.362218817077,-0.176389408324, # left arm
            -0.0828279393923,0.174905223639,-0.339518798998,1.65209848697,0.210827669057,-0.0140059461786,0.229339100035]) # right arm
        self.trajGoIdle.addPostureWP([-0.0668356331335,-0.0668356331335, # torso down
            -0.278059286633,0.0625583163478,-0.123033065504,0.545647683565,0.172132538926,-0.362271085326,-0.176658058859, # left arm
            -0.109102114669,0.181464974327,-0.0649566641951,1.64339461228,0.210281859941,-0.0105670855834,0.235870513922]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669364413641,-0.0669364413641, # torso down
            -0.277335034852,0.0623533949709,-0.122892258994,0.54560380799,0.172138697952,-0.36237979686,-0.176377288196, # left arm
            -0.136810993524,0.20342921713,0.259670003135,1.56923090683,0.24775861969,0.0208999380518,0.212466533211]) # right arm
        self.trajGoIdle.addPostureWP([-0.0667615308683,-0.0667615308683, # torso down
            -0.278123690122,0.0625046520159,-0.122976882533,0.545738812265,0.172136298507,-0.362138311689,-0.176503482748, # left arm
            -0.117939113333,0.167777869546,0.604001566993,1.51971288173,0.180902147432,0.0803449826369,0.178614506884]) # right arm
        self.trajGoIdle.addPostureWP([-0.0670801364241,-0.0670801364241, # torso down
            -0.277922021023,0.0624457805303,-0.122979849186,0.545710016693,0.172276779691,-0.362264161415,-0.176430157406, # left arm
            -0.120214967272,0.21496108816,0.89043092093,1.32384986737,0.122913489356,0.27037365636,0.342388242262]) # right arm
        self.trajGoIdle.addPostureWP([-0.0669533149885,-0.0669533149885, # torso down
            -0.277543420791,0.0625100191736,-0.123006935487,0.545784265346,0.172084818395,-0.362339731543,-0.176624084725, # left arm
            -0.12102407016,0.243561193453,1.29177111832,1.13080160352,0.127254035661,0.282357513999,0.309297317776]) # right arm
        self.trajGoIdle.addPostureWP([-0.0673218768229,-0.0673218768229, # torso down
            -0.277213443375,0.0624860194135,-0.122847414174,0.545649967851,0.172034700149,-0.362262725587,-0.176628655393, # left arm
            -0.19277931445,0.175408584304,1.16279582674,0.511526702287,0.0499538675104,0.27049721285,0.306251859392]) # right arm
        self.trajGoIdle.addPostureWP([-0.0673558711853,-0.0673558711853, # torso down
            -0.276961492108,0.0625948694083,-0.123168401629,0.545642788456,0.172188522861,-0.362076345223,-0.176450299657, # left arm
            -0.211720465819,0.0612837535295,1.16366097415,0.305813987218,-0.450347540539,0.275658893106,0.301685071753]) # right arm
        self.trajGoIdle.addPostureWP([-0.0674228199809,-0.0674228199809, # torso down
            -0.277837520036,0.0623686270456,-0.122826114885,0.545728453527,0.172060941064,-0.36230606451,-0.17653223036, # left arm
            -0.215284492784,-0.0217748645154,1.11537911177,0.11614566853,-0.827524281504,0.279503420012,0.30277813675]) # right arm
        self.trajGoIdle.addPostureWP([-0.0675698501565,-0.0675698501565, # torso down
            -0.277816935041,0.0624014702687,-0.123088108519,0.545803707466,0.172060147599,-0.36211085995,-0.176358418516, # left arm
            -0.228575760554,0.0636863274989,0.724487861251,0.0432980502212,-0.828225137412,0.280941538462,0.301625777588]) # right arm
        self.trajGoIdle.addPostureWP([-0.0676580463937,-0.0676580463937, # torso down
            -0.277908923,0.062334910171,-0.122844839816,0.5455252224,0.172018450288,-0.362362365705,-0.176528563738, # left arm
            -0.216608393365,0.00240349729189,0.280177966864,0.0449444777162,-0.828021983447,0.280759641616,0.302878616289]) # right arm
        self.trajGoIdle.addPostureWP([-0.067634978329,-0.067634978329, # torso down
            -0.277179721789,0.0626350983513,-0.123517478015,0.545506826506,0.171819278972,-0.362064828929,-0.175687942545, # left arm
            -0.203697850154,-0.0548539569602,0.152004692106,0.0466337024757,-0.0181904042239,0.280543921912,0.304179239269]) # right arm




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