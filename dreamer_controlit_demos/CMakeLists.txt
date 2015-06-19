cmake_minimum_required(VERSION 2.8.3)
project(dreamer_controlit_demos)

################
## ControlIt! ##
################

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    cmake_modules
    # controlit_core
    controlit_cmake
    controlit_dependency_addons
    geometry_msgs
    interactive_markers
    roscpp
    std_msgs
    tf
)

list(APPEND CMAKE_MODULE_PATH ${controlit_cmake_DIR}/../../../../src/controlit/controlit_cmake/cmake)

include(controlitbuild)
controlit_build_init()

find_package(Eigen REQUIRED)
find_package(Rbdl REQUIRED)

# find_package(Eigen REQUIRED)
# find_package(OpenMP)
# find_package(Boost COMPONENTS components)

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  QuatInterpMsg.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   getControlItParameters.srv
#   getControllerConfig.srv
#   get_command.srv
#   get_constraint_set_jacobians.srv
#   get_inertia_matrix.srv
#   get_parameters.srv
#   lock.srv
#   tare.srv
#   unlock.srv
#   updateControlItParameters.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  # sensor_msgs
  # diagnostic_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

# # Add the ControlIt!-specific build options and macros
# rosbuild_find_ros_package(controlit_cmake)
# list(APPEND CMAKE_MODULE_PATH ${controlit_cmake_PACKAGE_PATH}/cmake)
# include(controlitbuild)
# controlit_build_init()

# # find Eigen/RBDL
# controlit_build_depend_rbdl()

# #set the default path for built executables to the "bin" directory
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# #set the default path for built libraries to the "lib" directory
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

###########
## Build ##
###########

# get_cmake_property(_variableNames VARIABLES)
# foreach (_variableName ${_variableNames})
#     message(STATUS "+++ ${_variableName}=${${_variableName}}")
# endforeach()

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    # /usr/realtime/include
    # ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
)

add_executable(QuaternionInterpolator src/QuaternionInterpolator.cpp)

add_dependencies(QuaternionInterpolator dreamer_controlit_demos_generate_messages_cpp)

target_link_libraries(QuaternionInterpolator
    # ${Boost_LIBRARIES}
    # ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${RBDL_LIBRARY}
)

add_executable(DreamerInteractiveMarkers src/DreamerInteractiveMarkers.cpp)

target_link_libraries(DreamerInteractiveMarkers
    # ${Boost_LIBRARIES}
    # ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)
