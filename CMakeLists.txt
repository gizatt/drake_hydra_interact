cmake_minimum_required(VERSION 2.8.3)
project(drake_hydra_interact)

find_package(catkin REQUIRED COMPONENTS
             std_msgs geometry_msgs
             roscpp tf)

find_package(PkgConfig)

catkin_package(
    INCLUDE_DIRS
    LIBRARIES
    CATKIN_DEPENDS std_msgs geometry_msgs tf message_runtime
)
include_directories(include ${catkin_INCLUDE_DIRS})

catkin_python_setup()
catkin_install_python(PROGRAMS
    src/drake_hydra_interact/run_interaction.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".git" EXCLUDE)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
