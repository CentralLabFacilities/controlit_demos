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


        self.trajGoGrasp.addRHCartWP([-0.05315049977394091, -0.2518524008692323, 0.784235249961833])
        self.trajGoGrasp.addRHCartWP([-0.052370244321496597, -0.352128532076911, 0.805959045860243])
        self.trajGoGrasp.addRHCartWP([-0.05333996888415419, -0.45842107725140235, 0.852141473280871])
        self.trajGoGrasp.addRHCartWP([-0.053361958936085724, -0.504322659594482, 0.881105095173649])
        self.trajGoGrasp.addRHCartWP([-0.05342235281940607, -0.553834407217619, 0.920158856377428])
        self.trajGoGrasp.addRHCartWP([-0.05320572714876385, -0.5987665382883445, 0.964776303865292])
        self.trajGoGrasp.addRHCartWP([-0.054515097748979856, -0.6215143646755491, 0.991787103170205])
        self.trajGoGrasp.addRHCartWP([-0.051691580505462444, -0.6784446760413949, 1.07952246663411])
        self.trajGoGrasp.addRHCartWP([0.0971231860796643, -0.6803655327411001, 1.152294128273108])
        self.trajGoGrasp.addRHCartWP([0.19984442730902638, -0.6026846715332725, 1.17339972958458])
        self.trajGoGrasp.addRHCartWP([0.24872652503623788, -0.5149876336054383, 1.176813338949021])
        self.trajGoGrasp.addRHCartWP([0.2664043271349292, -0.4021777836854898, 1.163992854461617])
        self.trajGoGrasp.addRHCartWP([0.24434881933948555, -0.3008891120595256, 1.142302687100362])
        self.trajGoGrasp.addRHCartWP([0.22261543917778406, -0.20971614287897308, 1.115528916741094])
        self.trajGoGrasp.addRHCartWP([0.23513806269408313, -0.19558953428619413, 1.041639606275])
        self.trajGoGrasp.addRHCartWP([0.23392997222033968, -0.19092455150336562, 1.009552769447711])



        self.trajGoGrasp.addRHOrientWP([0.003689687310915867, 0.9957416715200532, -0.026441594039272404, 0.088236907871142])
        self.trajGoGrasp.addRHOrientWP([0.09317768786853839, 0.9912912961773385, -0.017225418119813738, 0.0914481797365218])
        self.trajGoGrasp.addRHOrientWP([0.19387090859986816, 0.9767512365272679, -0.006969389332788913, 0.0912278485630495])
        self.trajGoGrasp.addRHOrientWP([0.24028577621266115, 0.9663475579948857, -0.0031973687518336872, 0.091787361549751])
        self.trajGoGrasp.addRHOrientWP([0.29345721553935034, 0.951533816957999, 0.0014151939342728261, 0.0920013861817364])
        self.trajGoGrasp.addRHOrientWP([0.34600695034509765, 0.9336555788950605, 0.005980269923038519, 0.0923617165424198])
        self.trajGoGrasp.addRHOrientWP([0.37459733722374444, 0.9227836463779604, 0.0102342921728125, 0.0896796308310709])
        self.trajGoGrasp.addRHOrientWP([0.4569600474531073, 0.8841918581622882, 0.019389284152198896, 0.094954350354469])
        self.trajGoGrasp.addRHOrientWP([0.47730334635874505, 0.8174406338295004, 0.075323530693632, 0.31352622130057])
        self.trajGoGrasp.addRHOrientWP([0.4095248957897833, 0.7362601427657772, 0.13828063928006362, 0.520661912091888])
        self.trajGoGrasp.addRHOrientWP([0.24288821377611014, 0.6882104842618239, 0.11149950757191958, 0.674492034624557])
        self.trajGoGrasp.addRHOrientWP([0.01778007173669493, 0.6724766554420316, 0.13351937093688304, 0.72775792302053])
        self.trajGoGrasp.addRHOrientWP([-0.023260154655741418, 0.8002446535016311, 0.0640769206102517, 0.595786545662582])
        self.trajGoGrasp.addRHOrientWP([-0.0902955891605179, 0.8210907062266253, 0.07228385012796965, 0.558955994454924])
        self.trajGoGrasp.addRHOrientWP([-0.07425934111414212, 0.7519038387863897, 0.03055572248170427, 0.65436420692089])
        self.trajGoGrasp.addRHOrientWP([-0.06639134871272698, 0.7673017701954629, 0.04154021094583732, 0.636486129578319])


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
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])
        self.trajGoGrasp.addLHOrientWP([0.0, 1.0, 0.0, 0.0])


        self.trajGoGrasp.addPostureWP([-0.0881059099651,-0.0881059099651, # torso down
            -0.283556957266,0.0924124708761,-0.156471723844,0.513551658216,0.203800407183,-0.427164424216,-0.171294250963, # left arm
            -0.173655419619,0.0640732592637,0.0706005418841,-0.0229792734714,-0.00655058553818,0.193395389032,-0.0618370509101]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0879945528668,-0.0879945528668, # torso down
            -0.283044750473,0.0924099118135,-0.156416655842,0.5136229984,0.203892588588,-0.427243182094,-0.171369210651, # left arm
            -0.172141990723,0.248665536718,0.0703523943134,-0.0226324993829,-0.00853566056217,0.194331068932,-0.0662735234457]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0878695311444,-0.0878695311444, # torso down
            -0.284008061626,0.0926758337913,-0.156346703415,0.513719928638,0.203770264577,-0.427237022471,-0.171310337789, # left arm
            -0.17321348658,0.457076889205,0.0702670123979,-0.0230701421286,-0.00986113589105,0.194253578907,-0.0703754368218]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0876865407094,-0.0876865407094, # torso down
            -0.283378445883,0.0924982915002,-0.15629400643,0.51361808865,0.203900692651,-0.427240271818,-0.171395043397, # left arm
            -0.173163584244,0.554576881179,0.0704844147751,-0.0230091000144,-0.00874753998033,0.194083803009,-0.072291429352]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0878687002111,-0.0878687002111, # torso down
            -0.283742188992,0.0925716161899,-0.156398298169,0.513581254265,0.203712485933,-0.427272243638,-0.171380357058, # left arm
            -0.17320084132,0.667979665271,0.070801210872,-0.0230236110855,-0.0079666885217,0.193661767709,-0.0750548327298]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0877020162288,-0.0877020162288, # torso down
            -0.282972103658,0.0925739161327,-0.156387420828,0.513596423584,0.203759299153,-0.42720312638,-0.171331759681, # left arm
            -0.172253689773,0.781648802029,0.0712147575324,-0.0231102121905,-0.00711940712971,0.194062784205,-0.0773689885559]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0874977961199,-0.0874977961199, # torso down
            -0.28399333394,0.092548920162,-0.156316525554,0.513609763099,0.203738191397,-0.427272447569,-0.171386140498, # left arm
            -0.174023855334,0.845166926717,0.070738774406,-0.0248392150589,-0.00966037289474,0.19408242318,-0.0790057841242]) # right arm
        self.trajGoGrasp.addPostureWP([-0.087993898538,-0.087993898538, # torso down
            -0.283044467672,0.0924062158455,-0.156477673045,0.513716011946,0.203760901686,-0.427306673493,-0.171414312947, # left arm
            -0.172965615638,1.03284070487,0.0710571338158,-0.0154923425894,-0.00555577393801,0.196065885887,-0.0850279786461]) # right arm
        self.trajGoGrasp.addPostureWP([-0.08774357093,-0.08774357093, # torso down
            -0.284004366541,0.0923241910495,-0.156397714009,0.513619987329,0.203832341817,-0.427038858455,-0.171247973217, # left arm
            0.0706008341321,1.074222666,0.144798651548,0.355971277445,-0.123036733341,0.151284552372,-0.123966585288]) # right arm
        self.trajGoGrasp.addPostureWP([-0.087807720574,-0.087807720574, # torso down
            -0.283978389665,0.0922118167759,-0.156350387013,0.513595210346,0.203836779784,-0.427473909344,-0.171226828859, # left arm
            0.0727425305949,0.955179638145,0.142383629069,0.815348451163,-0.012281193382,0.147194033393,-0.123596400222]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0874090311378,-0.0874090311378, # torso down
            -0.284066997814,0.0923172874063,-0.156371117469,0.513664127406,0.203841277515,-0.427243052803,-0.171263106221, # left arm
            0.0651764306798,0.870846335433,0.0279726719504,1.12127628696,0.329131016388,0.141326569968,-0.118137498386]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0874591612476,-0.0874591612476, # torso down
            -0.283541851927,0.0922295564425,-0.156330275237,0.513640695884,0.203985616469,-0.427399471018,-0.171346200318, # left arm
            0.0551161806858,0.833672315301,-0.232310084168,1.40906699471,0.643537100794,0.137186212939,-0.111447603828]) # right arm
        self.trajGoGrasp.addPostureWP([-0.087403156317,-0.087403156317, # torso down
            -0.283205617145,0.0920876818059,-0.156427669763,0.513689465556,0.203643609151,-0.427342899912,-0.171216342533, # left arm
            0.0221314450808,0.776656864949,-0.453768923255,1.64064200994,0.816491502764,-0.217814665646,0.247371869283]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0876223023967,-0.0876223023967, # torso down
            -0.284050664983,0.0923607884769,-0.156329827855,0.513616417675,0.203826203965,-0.427248925141,-0.171409672784, # left arm
            -0.0150513901303,0.552670668425,-0.559964914849,1.72192492799,0.73835976493,-0.313526401788,0.347623983081]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0871561744444,-0.0871561744444, # torso down
            -0.283258609096,0.0923230146153,-0.156323402944,0.513555135954,0.203811903762,-0.427306181001,-0.171512522932, # left arm
            -0.0256839637462,0.36949986674,-0.534026699088,1.43474736486,0.506983257135,0.0730854062449,0.280160315243]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0872820677674,-0.0872820677674, # torso down
            -0.28325283778,0.0921687501927,-0.156383046385,0.513796398711,0.203781210372,-0.42734195797,-0.17117057729, # left arm
            -0.0313514981525,0.305895847147,-0.520006526856,1.31019447436,0.44229625194,0.104980691651,0.234523937053]) # right arm



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

        # right arm Position
        self.trajGoLocation.addRHCartWP([0.20449948887226677, -0.20948957210456626, 1.110838964113786])
        self.trajGoLocation.addRHCartWP([0.18046087910581973, -0.12616919149374142, 1.0977443345495])
        self.trajGoLocation.addRHCartWP([0.18926321706100066, -0.12145223903886486, 1.020784914017576])
        self.trajGoLocation.addRHCartWP([0.20192896676298455, -0.12921288071480855, 0.99309761820511])
        
        # right arm Orientation
        self.trajGoLocation.addRHOrientWP([-0.0376407795858349, 0.6054625959647735, 0.017750957077122514, 0.794784952123915])
        self.trajGoLocation.addRHOrientWP([-0.14066474540191656, 0.6110313040996426, 0.12757551832285424, 0.768491159308914])
        self.trajGoLocation.addRHOrientWP([-0.11594049292493272, 0.7000383298679741, 0.1799950555299789, 0.681253197277444])
        self.trajGoLocation.addRHOrientWP([-0.09890769038515215, 0.7304672920003714, 0.1853055230246751, 0.649843571359192])


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
        self.trajGoLocation.addPostureWP([-0.0347624942598,-0.0347624942598, # torso down
             -0.0407696996257,0.0326417990446,-0.0371064791657,0.123777453542,0.061098271116,-0.038446811658,-0.0357243759958, # left arm
             -0.259891212719,0.0895713031901,-0.0879624082974,1.91719530325,0.0796745650036,0.121841943646,0.0352053123559]) # right arm
        self.trajGoLocation.addPostureWP([-0.0350361733095,-0.0350361733095, # torso down
             -0.041726178668,0.0326195074871,-0.0370895495974,0.123418961899,0.0610157367814,-0.0387867304878,-0.0355807997238, # left arm
             -0.271498787932,0.0872719435067,-0.410784011647,1.88917980127,-0.0335515977597,0.122061093967,0.0355879385164]) # right arm
        self.trajGoLocation.addPostureWP([-0.0348512513022,-0.0348512513022, # torso down
             -0.0403793276396,0.0326711782364,-0.0370230780832,0.124099681556,0.0609643050087,-0.0384627093953,-0.0357502982058, # left arm
             -0.219156644198,0.0895702524116,-0.43949100394,1.57238614075,-0.0655707212456,0.123876532,0.0347084864628]) # right arm
        self.trajGoLocation.addPostureWP([-0.0346021902322,-0.0346021902322, # torso down
             -0.0411943398595,0.0326231408419,-0.0368738284792,0.123673435785,0.0609701631141,-0.0386208809603,-0.0359758723907, # left arm
             -0.16137803856,0.0915041318105,-0.430660772042,1.42454059265,-0.0488862084445,0.128783479435,0.037967981243]) # right arm
        
        
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