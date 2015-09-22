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


        self.trajGoGrasp.addRHCartWP([-0.04559452000611031, -0.19008056323543748, 0.782085952951363])
        self.trajGoGrasp.addRHCartWP([-0.04278645859334078, -0.24387389542266155, 0.785241608607163])
        self.trajGoGrasp.addRHCartWP([-0.04253375570168321, -0.28613126787927595, 0.79138818911386])
        self.trajGoGrasp.addRHCartWP([-0.0416100711698767, -0.33616810400090774, 0.803084565731383])
        self.trajGoGrasp.addRHCartWP([-0.039679016861252775, -0.3969831850459549, 0.824146821227679])
        self.trajGoGrasp.addRHCartWP([-0.03746433954853037, -0.44681585644880945, 0.84766828536423])
        self.trajGoGrasp.addRHCartWP([-0.03732034451822403, -0.5329283263806296, 0.904412588517898])
        self.trajGoGrasp.addRHCartWP([-0.035045231251828485, -0.5835864195006035, 0.95043646188269])
        self.trajGoGrasp.addRHCartWP([-0.03263204687045223, -0.6256876174667032, 0.999028329715836])
        self.trajGoGrasp.addRHCartWP([-0.029512177591144148, -0.664053846458048, 1.055825041237128])
        self.trajGoGrasp.addRHCartWP([-0.027335284868662314, -0.6818325398372871, 1.088491934493627])
        self.trajGoGrasp.addRHCartWP([0.13975794980631698, -0.663398597374016, 1.20527858760325])
        self.trajGoGrasp.addRHCartWP([0.22612730885709253, -0.5164774897104093, 1.2087090577995])
        self.trajGoGrasp.addRHCartWP([0.24374586281677063, -0.3639808581987248, 1.204322821913439])
        self.trajGoGrasp.addRHCartWP([0.2300879814484577, -0.2963886799053862, 1.155781062156949])
        self.trajGoGrasp.addRHCartWP([0.20558266357761731, -0.2418535707701937, 1.11126350990108])
        self.trajGoGrasp.addRHCartWP([0.20803213187652422, -0.2126564519748027, 1.052012866674892])
        self.trajGoGrasp.addRHCartWP([0.20445390596749197, -0.2095795869562649, 1.013159152090175])


        self.trajGoGrasp.addRHOrientWP([0.13194576846018757, 0.9879963065434669, -0.022340313171328133, 0.0771655548112327])
        self.trajGoGrasp.addRHOrientWP([0.17892964579031265, 0.9803485863864143, -0.01627005218591102, 0.0814623620414976])
        self.trajGoGrasp.addRHOrientWP([0.2156243421893463, 0.9729399113682952, -0.010397937692061567, 0.0823769070420682])
        self.trajGoGrasp.addRHOrientWP([0.25940122343748706, 0.9621245644251778, -0.005074339509709056, 0.083675437777982])
        self.trajGoGrasp.addRHOrientWP([0.31324180533898066, 0.9458298064713894, 0.001986813763718163, 0.0853322983947958])
        self.trajGoGrasp.addRHOrientWP([0.3595473281759007, 0.928992732509931, 0.006017762720145398, 0.0875328982562957])
        self.trajGoGrasp.addRHOrientWP([0.44146677824729685, 0.8930783423473367, 0.01845558716138862, 0.0847204192424324])
        self.trajGoGrasp.addRHOrientWP([0.4855025315919408, 0.8686714465002588, 0.027491533560590767, 0.0945591108008707])
        self.trajGoGrasp.addRHOrientWP([0.5340891266855825, 0.8392357415066238, 0.01831763105487989, 0.100482034845023])
        self.trajGoGrasp.addRHOrientWP([0.5803060551838528, 0.807084743913817, 0.029947748913991084, 0.104700672364165])
        self.trajGoGrasp.addRHOrientWP([0.6065250565012114, 0.7873170530897268, 0.03447969202672497, 0.1052158000875003])
        self.trajGoGrasp.addRHOrientWP([0.5436848276306436, 0.660838043653104, -0.06292066249054026, 0.513557083970932])
        self.trajGoGrasp.addRHOrientWP([0.25057371669488665, 0.5493656900348325, -0.18377333746968613, 0.775652958190398])
        self.trajGoGrasp.addRHOrientWP([0.04904451262392741, 0.5485804847623899, -0.06442172260184843, 0.83216820966458])
        self.trajGoGrasp.addRHOrientWP([-0.04707107698846003, 0.6192319444433716, -0.010820181539668171, 0.783721274665602])
        self.trajGoGrasp.addRHOrientWP([-0.07898679981316209, 0.7101584725048463, -0.008849635847211174, 0.699541073368984])
        self.trajGoGrasp.addRHOrientWP([-0.030810415933303355, 0.7425987307398529, -0.014613680225357265, 0.668867912015404])
        self.trajGoGrasp.addRHOrientWP([-0.0010511506388146514, 0.7778843341970075, -0.009327557048340427, 0.628337373051072])


        self.trajGoGrasp.addLHCartWP([0.03248539937472964, 0.20920933710392725, 0.808302642220199])
        self.trajGoGrasp.addLHCartWP([0.032590628384931146, 0.20926476892692178, 0.808310083554658])
        self.trajGoGrasp.addLHCartWP([0.03248174680527754, 0.20925421003506542, 0.808295780046209])
        self.trajGoGrasp.addLHCartWP([0.032440167459528185, 0.2093666753752737, 0.80829019003586])
        self.trajGoGrasp.addLHCartWP([0.032578476333800475, 0.20928889637691656, 0.808298854340603])
        self.trajGoGrasp.addLHCartWP([0.03299821742398054, 0.20930377209804563, 0.808363600803685])
        self.trajGoGrasp.addLHCartWP([0.032436208081242035, 0.20927004841953264, 0.808295197330724])
        self.trajGoGrasp.addLHCartWP([0.03232025806017382, 0.20932304913888586, 0.80827139524876])
        self.trajGoGrasp.addLHCartWP([0.03253982624892695, 0.20928685725858062, 0.808301780680069])
        self.trajGoGrasp.addLHCartWP([0.03263083138604955, 0.20930895101239771, 0.808326042374362])
        self.trajGoGrasp.addLHCartWP([0.032293449800666556, 0.20928248692516957, 0.808276130904483])
        self.trajGoGrasp.addLHCartWP([0.032320605212310914, 0.2092021839092438, 0.808265843761162])
        self.trajGoGrasp.addLHCartWP([0.03252150877479045, 0.20926482156160633, 0.808298018361703])
        self.trajGoGrasp.addLHCartWP([0.032232135545876464, 0.20935901437143573, 0.808263319320634])
        self.trajGoGrasp.addLHCartWP([0.03243023862889071, 0.2092616410375087, 0.80828905207682])
        self.trajGoGrasp.addLHCartWP([0.03231138493695246, 0.2092906019952438, 0.808285713329053])
        self.trajGoGrasp.addLHCartWP([0.03257029087594509, 0.20923671073347125, 0.80830002643027])
        self.trajGoGrasp.addLHCartWP([0.03259744861351244, 0.20922782052719024, 0.808300909314801])


        self.trajGoGrasp.addLHOrientWP([0.12233682808063064, 0.992152663881907, 0.01358231138610832, 0.02196162254061684])
        self.trajGoGrasp.addLHOrientWP([0.12235106735020151, 0.9921497078380347, 0.013704949705195469, 0.02193964240244])
        self.trajGoGrasp.addLHOrientWP([0.12227027893632283, 0.9921595813186089, 0.013604788190322884, 0.02200576798961761])
        self.trajGoGrasp.addLHOrientWP([0.12210999272490072, 0.9921794154337277, 0.013707498168441244, 0.02193767900834666])
        self.trajGoGrasp.addLHOrientWP([0.12221540546401082, 0.9921659371110456, 0.013521479361277873, 0.02207526896123145])
        self.trajGoGrasp.addLHOrientWP([0.12227313278424906, 0.9921465720554631, 0.013676382388492827, 0.022525921123966])
        self.trajGoGrasp.addLHOrientWP([0.12219478515951333, 0.9921734790402865, 0.013619211092966686, 0.0217885074780173])
        self.trajGoGrasp.addLHOrientWP([0.12218403841137364, 0.9921783289632189, 0.013655886507318008, 0.0216041907263801])
        self.trajGoGrasp.addLHOrientWP([0.12227274236189287, 0.9921616539810666, 0.013539545606478059, 0.02193876818845184])
        self.trajGoGrasp.addLHOrientWP([0.12227969687163809, 0.9921568412865811, 0.013690692607021515, 0.02202369081589938])
        self.trajGoGrasp.addLHOrientWP([0.12231441307618826, 0.9921620411294101, 0.013806835901723909, 0.02151835909481018])
        self.trajGoGrasp.addLHOrientWP([0.122355886392722, 0.9921576613891137, 0.013607405559010443, 0.02161135178300808])
        self.trajGoGrasp.addLHOrientWP([0.12223276990178543, 0.9921642409738441, 0.013536139857485842, 0.02204635599575466])
        self.trajGoGrasp.addLHOrientWP([0.12221065055749229, 0.9921749777507102, 0.013836694337065717, 0.02149223826210111])
        self.trajGoGrasp.addLHOrientWP([0.12225520406565518, 0.9921633491598199, 0.013565705794359265, 0.0219436844931593])
        self.trajGoGrasp.addLHOrientWP([0.12226952690280717, 0.9921649846855142, 0.013787238798289957, 0.02164989610312572])
        self.trajGoGrasp.addLHOrientWP([0.12233117398274938, 0.9921562958070735, 0.013675126642225817, 0.02177061032006391])
        self.trajGoGrasp.addLHOrientWP([0.12234333810414197, 0.9921512061924607, 0.013659103310647971, 0.0219435769447489])



        self.trajGoGrasp.addPostureWP([-0.0664593114019,-0.0664593114019, # torso down
             -0.279748405139,0.0245071739362,-0.13147786998,0.553355936602,0.177556686494,-0.359316353222,-0.181112958402, # left arm
             -0.190366257489,-0.0490754592928,0.0371977847825,0.0781818757063,-0.0233902399671,0.139230261165,0.310658948318]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0662143767978,-0.0662143767978, # torso down
             -0.279324181289,0.0245413796482,-0.131378511202,0.553205398005,0.177605104538,-0.35932927595,-0.181176276975, # left arm
             -0.187465264594,0.0477469015612,0.0373877955123,0.082016488505,-0.0234932228156,0.138834336774,0.30982306187]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0664642383169,-0.0664642383169, # torso down
             -0.279797466465,0.0246938222923,-0.131471115052,0.5532016293,0.177672149034,-0.359335961113,-0.180988711852, # left arm
             -0.188148507714,0.124733202402,0.0376628203678,0.0820315784047,-0.0250353631441,0.138462722212,0.309141615691]) # right arm
        self.trajGoGrasp.addPostureWP([-0.066479788978,-0.066479788978, # torso down
             -0.279852111779,0.0246687499604,-0.131272965995,0.553187370895,0.177676559462,-0.35928424061,-0.180988462894, # left arm
             -0.187654134062,0.216978998341,0.038315984088,0.0828955874936,-0.0236182293256,0.13864219354,0.30697433682]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0665122091448,-0.0665122091448, # torso down
             -0.279529831551,0.0245477800497,-0.131398267722,0.553082722329,0.177344114631,-0.359286563922,-0.181163112563, # left arm
             -0.186480700795,0.332594344246,0.0383677284081,0.0832116883515,-0.0223492672319,0.138285851178,0.304924035357]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0665230591407,-0.0665230591407, # torso down
             -0.278774944697,0.0245804263956,-0.131273804703,0.553105497552,0.177604694559,-0.359245693008,-0.181109013504, # left arm
             -0.184947631463,0.431829563718,0.0396007306881,0.0839047778114,-0.0185065812945,0.138835411693,0.304799508606]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0663497070999,-0.0663497070999, # torso down
             -0.27979164691,0.0245054280623,-0.131468706385,0.553131037068,0.177526260848,-0.359389886883,-0.181017302127, # left arm
             -0.187262915214,0.617622197599,0.0405194646211,0.0777529862952,-0.0184127785936,0.140099116291,0.299679788023]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0662970804462,-0.0662970804462, # torso down
             -0.279788899288,0.0246588961341,-0.13142802539,0.553190758347,0.177634166429,-0.359494883234,-0.181031420055, # left arm
             -0.187711816022,0.740788779381,0.0416614151095,0.0775586534954,-0.0145036426501,0.160119683011,0.27803697089]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0664372348648,-0.0664372348648, # torso down
             -0.279305058193,0.0246026665825,-0.131459954011,0.553186362366,0.177409377854,-0.359361162577,-0.1811941548, # left arm
             -0.187832471177,0.856617720837,0.0403608824514,0.0767235058015,0.0236984922783,0.162805579699,0.275241378158]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0663317725547,-0.0663317725547, # torso down
             -0.279412737698,0.0245923364575,-0.131353083115,0.553368558017,0.177610262578,-0.359436421876,-0.181129136778, # left arm
             -0.188034417607,0.98014344079,0.040180679737,0.0768073833338,0.0227979023352,0.171680050873,0.265639563744]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0660940142817,-0.0660940142817, # torso down
             -0.279792564708,0.0245542520901,-0.131416437845,0.553413215893,0.177868297868,-0.35955095467,-0.181162247393, # left arm
             -0.187285682829,1.0469633941,0.0406069265102,0.0774524192875,0.0234076958358,0.171439570025,0.264978115452]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0663285895689,-0.0663285895689, # torso down
             -0.27969481101,0.0244291589957,-0.131493177999,0.5531860688,0.177495927792,-0.359487601829,-0.181215096305, # left arm
             -0.0921295154001,1.10526624453,0.317910909112,0.698937163972,0.26848186615,0.186287413049,0.180557993526]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0666303352853,-0.0666303352853, # torso down
             -0.27951012207,0.024554676251,-0.131434910738,0.55321034578,0.177398228367,-0.359264922559,-0.181128527516, # left arm
             -0.0634075425193,0.775661772776,0.375794566307,1.35400537811,0.625029359137,0.168738275457,0.209145187722]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0662043170916,-0.0662043170916, # torso down
             -0.279843244075,0.0246609445509,-0.131344529136,0.553230910798,0.177797425106,-0.359511772787,-0.181066631257, # left arm
             -0.0822977269868,0.646467439001,0.0789349306569,1.75912886393,0.629767908192,0.155721761995,0.21506162473]) # right arm
        self.trajGoGrasp.addPostureWP([-0.066319233001,-0.066319233001, # torso down
             -0.279453491774,0.0245421118255,-0.131492463452,0.553422977391,0.177449259539,-0.359213849414,-0.181068143906, # left arm
             -0.135910947705,0.61582054955,-0.167312177961,1.80856913387,0.631173461706,0.150442541135,0.220412502198]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0661713003276,-0.0661713003276, # torso down
             -0.279820372261,0.0246500462262,-0.131408323587,0.553430445605,0.177782323327,-0.359389453664,-0.181111216609, # left arm
             -0.184908210618,0.553174164006,-0.356207197472,1.81144660858,0.614574055669,0.0544979848269,0.322156117168]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0659548462766,-0.0659548462766, # torso down
             -0.279167372089,0.0244654615247,-0.13149570234,0.553257711757,0.177661558312,-0.359207396043,-0.181094118893, # left arm
             -0.190319103603,0.350587268231,-0.35769531115,1.61987720726,0.403850328264,0.0521862673476,0.32369172933]) # right arm
        self.trajGoGrasp.addPostureWP([-0.0661714311264,-0.0661714311264, # torso down
             -0.278842746972,0.0245695395198,-0.13141572432,0.553327495851,0.177598382028,-0.359435943164,-0.181143630501, # left arm
             -0.191729744247,0.297882423255,-0.36000150893,1.47806008241,0.346614486518,0.0528598710709,0.322944990828]) # right arm


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
        self.trajGoIdle.addRHCartWP([0.2002293878094432, -0.12837318014228544, 1.09450408805717])
        self.trajGoIdle.addRHCartWP([0.22403461378231257, -0.20273700952067492, 1.130796315102099])
        self.trajGoIdle.addRHCartWP([0.22732227056397292, -0.2438027116706776, 1.1889611966628])
        self.trajGoIdle.addRHCartWP([0.19719140259395582, -0.28076005944485605, 1.077636954223212])
        self.trajGoIdle.addRHCartWP([0.17219922018303624, -0.3194794214282456, 1.112709053661094])
        self.trajGoIdle.addRHCartWP([0.2284038839801888, -0.44829928361815075, 1.155696354514254])
        self.trajGoIdle.addRHCartWP([0.14932392927918303, -0.6433094359761335, 1.264314997179006])
        self.trajGoIdle.addRHCartWP([0.05288667935588165, -0.6802431030883713, 1.124799854908889])
        self.trajGoIdle.addRHCartWP([-0.023142186861017676, -0.5535134542113634, 0.926477637649365])
        self.trajGoIdle.addRHCartWP([-0.03617413368521693, -0.20998774738341777, 0.78768780968166])


        # right arm Orientation 
        self.trajGoIdle.addRHOrientWP([-0.12312530433506581, 0.5977249881389018, 0.18461082095617484, 0.77037902539765])
        self.trajGoIdle.addRHOrientWP([-0.008635022506701034, 0.5596304138174526, 0.1662452644911143, 0.811850816561204])
        self.trajGoIdle.addRHOrientWP([0.0651135936484889, 0.474618929152166, 0.15830035018184652, 0.863387567170697])
        self.trajGoIdle.addRHOrientWP([0.00010204019590855813, 0.6900086439827745, 0.12535340532689765, 0.712863650770469])
        self.trajGoIdle.addRHOrientWP([0.024447278249258124, 0.6793328589574478, 0.31592758966942747, 0.661890440641477])
        self.trajGoIdle.addRHOrientWP([0.20157604118615075, 0.6796625232065113, 0.14158026565936666, 0.690927624678664])
        self.trajGoIdle.addRHOrientWP([0.46055322943055377, 0.5827798457090784, -0.12144071092349171, 0.658415163879741])
        self.trajGoIdle.addRHOrientWP([0.4682335993234306, 0.7134756199277661, -0.27530990058771226, 0.442622067764022])
        self.trajGoIdle.addRHOrientWP([0.34769445913830066, 0.8878660121676856, -0.22085236421132523, 0.2050042456737029])
        self.trajGoIdle.addRHOrientWP([0.025172530517963726, 0.9809712267488497, -0.15582018376544834, 0.1130569163274894])
 
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
        self.trajGoIdle.addPostureWP([-0.035041443134,-0.035041443134, # torso down
             -0.0407938985819,0.0327043698482,-0.0371591510346,0.123915737246,0.0609767434975,-0.0386349383146,-0.0356138545595, # left arm
             -0.179480252261,0.140838997472,-0.447038898547,1.81943263423,-0.0655767457119,0.127778324088,0.0286252274714]) # right arm
        self.trajGoIdle.addPostureWP([-0.0346013725692,-0.0346013725692, # torso down
             -0.0404276070508,0.0326096193082,-0.0370083284433,0.124028545697,0.0609984931353,-0.0384280135755,-0.0357642111359, # left arm
             -0.146027689485,0.251388358947,-0.232203949477,1.90005361554,-0.0513857933674,0.129786261731,0.0390046186706]) # right arm
        self.trajGoIdle.addPostureWP([-0.0345444140843,-0.0345444140843, # torso down
             -0.0408204666527,0.0325550443775,-0.0370701716297,0.123773549571,0.0609623286641,-0.0385800715487,-0.035672170807, # left arm
             -0.0993800251952,0.317442107887,-0.0809764081246,2.03134614125,-0.0515803320387,0.137990266046,0.0364327115095]) # right arm
        self.trajGoIdle.addPostureWP([-0.0350942651603,-0.0350942651603, # torso down
             -0.0404285738926,0.0327091768615,-0.0369835681776,0.12370366762,0.0608421595783,-0.0386801051297,-0.0357241439553, # left arm
             -0.311628719567,0.456048093946,-0.12624257182,1.76653735742,0.226369950121,0.12722600649,0.0430162324847]) # right arm
        self.trajGoIdle.addPostureWP([-0.0349197633609,-0.0349197633609, # torso down
             -0.0413138722786,0.0328072487967,-0.0370163224079,0.123843439849,0.0611263177167,-0.0387616592708,-0.0357010561678, # left arm
             -0.456388219065,0.842630031907,-0.168616756594,1.88598448024,0.214888123908,0.153789352191,0.0352335861887]) # right arm
        self.trajGoIdle.addPostureWP([-0.0343278810064,-0.0343278810064, # torso down
             -0.0413880236935,0.0326048016834,-0.0371658593811,0.123923651384,0.0610555078597,-0.0385485555454,-0.0356854405447, # left arm
             -0.256418718685,0.849020261651,0.145856509265,1.53980822332,0.371534872788,0.137581085806,0.0263760834605]) # right arm
        self.trajGoIdle.addPostureWP([-0.0341463515117,-0.0341463515117, # torso down
             -0.0404569307052,0.0325994245575,-0.0370695634867,0.123774989011,0.0610809131643,-0.0386458581761,-0.0358773650739, # left arm
             -0.195047807143,1.00106466904,0.761624417157,1.05482685768,0.375881337508,0.137297918564,0.0264710183834]) # right arm
        self.trajGoIdle.addPostureWP([-0.0341683006142,-0.0341683006142, # torso down
             -0.0406763568062,0.0325935831562,-0.0369652797727,0.123877272408,0.0611546940425,-0.038423034386,-0.0357614731687, # left arm
             -0.224044349834,0.948675644312,0.719916775619,0.513043161672,0.37622555557,0.136723503909,0.0232947679167]) # right arm
        self.trajGoIdle.addPostureWP([-0.0345810799983,-0.0345810799983, # torso down
             -0.0408206111579,0.0327227355273,-0.036932288479,0.124126768744,0.0609640517321,-0.0385643170974,-0.0357349031189, # left arm
             -0.247007706696,0.620736809446,0.411472164973,0.264743923635,0.254647726822,0.141811527278,0.0286762958092]) # right arm
        self.trajGoIdle.addPostureWP([-0.0348973213927,-0.0348973213927, # torso down
             -0.0413644317422,0.0326808167838,-0.0370411723782,0.123833236007,0.0609405463001,-0.0385111206705,-0.0358528466127, # left arm
             -0.234308607784,-0.0367051476743,0.178780095273,0.262349686097,0.127307862519,0.139309280536,0.0334679656265]) # right arm



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