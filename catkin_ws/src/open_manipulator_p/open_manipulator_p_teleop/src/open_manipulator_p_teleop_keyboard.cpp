/*******************************************************************************
* Copyright 2019 ROBOTIS CO., LTD.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Authors: Darby Lim, Hye-Jong KIM, Ryan Shim, Yong-Ho Na */
#include "ros/ros.h"
#include "open_manipulator_p_teleop/Angle.h"
#include "open_manipulator_p_teleop/Grab.h"
#include "open_manipulator_p_teleop/open_manipulator_p_teleop_keyboard.h"

OpenManipulatorTeleop::OpenManipulatorTeleop()
: node_handle_(""),
  priv_node_handle_("~")
{
  /************************************************************
  ** Initialize ROS parameters
  ************************************************************/
  with_gripper_ = priv_node_handle_.param<bool>("with_gripper", false);

  /************************************************************
  ** Initialize variables
  ************************************************************/
  present_joint_angle_.resize(NUM_OF_JOINT);
  present_kinematic_position_.resize(3);

  /************************************************************
  ** Initialize ROS Subscribers and Clients
  ************************************************************/
  initSubscriber();
  initClient();

  disableWaitingForEnter();
  ROS_INFO("Start OpenManipulator-P Teleop Keyboard");
}

OpenManipulatorTeleop::~OpenManipulatorTeleop()
{
  restoreTerminalSettings();
  ROS_INFO("Run Once");
  // ros::shutdown();
}

void OpenManipulatorTeleop::initClient()
{
  goal_joint_space_path_client_ = node_handle_.serviceClient<open_manipulator_msgs::SetJointPosition>("goal_joint_space_path");
  goal_joint_space_path_from_present_client_ = node_handle_.serviceClient<open_manipulator_msgs::SetJointPosition>("goal_joint_space_path_from_present");
  goal_task_space_path_from_present_position_only_client_ = node_handle_.serviceClient<open_manipulator_msgs::SetKinematicsPose>("goal_task_space_path_from_present_position_only");
  goal_tool_control_client_ = node_handle_.serviceClient<open_manipulator_msgs::SetJointPosition>("goal_tool_control");
}

void OpenManipulatorTeleop::initSubscriber()
{
  joint_states_sub_ = node_handle_.subscribe("joint_states", 10, &OpenManipulatorTeleop::jointStatesCallback, this);
  kinematics_pose_sub_ = node_handle_.subscribe("kinematics_pose", 10, &OpenManipulatorTeleop::kinematicsPoseCallback, this);
}

void OpenManipulatorTeleop::jointStatesCallback(const sensor_msgs::JointState::ConstPtr &msg)
{
  std::vector<double> temp_angle;
  temp_angle.resize(NUM_OF_JOINT);
  for (std::vector<int>::size_type i = 0; i < msg->name.size(); i++)
  {
    if (!msg->name.at(i).compare("joint1"))  temp_angle.at(0) = (msg->position.at(i));
    else if (!msg->name.at(i).compare("joint2"))  temp_angle.at(1) = (msg->position.at(i));
    else if (!msg->name.at(i).compare("joint3"))  temp_angle.at(2) = (msg->position.at(i));
    else if (!msg->name.at(i).compare("joint4"))  temp_angle.at(3) = (msg->position.at(i));
    else if (!msg->name.at(i).compare("joint5"))  temp_angle.at(4) = (msg->position.at(i));
    else if (!msg->name.at(i).compare("joint6"))  temp_angle.at(5) = (msg->position.at(i));
  }
  present_joint_angle_ = temp_angle;
}

void OpenManipulatorTeleop::kinematicsPoseCallback(const open_manipulator_msgs::KinematicsPose::ConstPtr &msg)
{
  std::vector<double> temp_position;
  temp_position.push_back(msg->pose.position.x);
  temp_position.push_back(msg->pose.position.y);
  temp_position.push_back(msg->pose.position.z);
  present_kinematic_position_ = temp_position;
}

std::vector<double> OpenManipulatorTeleop::getPresentJointAngle()
{
  return present_joint_angle_;
}

std::vector<double> OpenManipulatorTeleop::getPresentKinematicsPose()
{
  return present_kinematic_position_;
}

bool OpenManipulatorTeleop::setJointSpacePathFromPresent(std::vector<std::string> joint_name, std::vector<double> joint_angle, double path_time)
{
  open_manipulator_msgs::SetJointPosition srv;
  srv.request.joint_position.joint_name = joint_name;
  srv.request.joint_position.position = joint_angle;
  srv.request.path_time = path_time;

  if (goal_joint_space_path_from_present_client_.call(srv))
  {
    return srv.response.is_planned;
  }
  return false;
}

bool OpenManipulatorTeleop::setJointSpacePath(std::vector<std::string> joint_name, std::vector<double> joint_angle, double path_time)
{
  open_manipulator_msgs::SetJointPosition srv;
  srv.request.joint_position.joint_name = joint_name;
  srv.request.joint_position.position = joint_angle;
  srv.request.path_time = path_time;

  if (goal_joint_space_path_client_.call(srv))
  {
    return srv.response.is_planned;
  }
  return false;
}

bool OpenManipulatorTeleop::setToolControl(std::vector<double> joint_angle)
{
  open_manipulator_msgs::SetJointPosition srv;
  srv.request.joint_position.joint_name.push_back(priv_node_handle_.param<std::string>("end_effector_name", "gripper"));
  srv.request.joint_position.position = joint_angle;

  if (goal_tool_control_client_.call(srv))
  {
    return srv.response.is_planned;
  }
  return false;
}

bool OpenManipulatorTeleop::setTaskSpacePathFromPresentPositionOnly(std::vector<double> kinematics_pose, double path_time)
{
  open_manipulator_msgs::SetKinematicsPose srv;
  srv.request.planning_group = priv_node_handle_.param<std::string>("end_effector_name", "gripper");
  srv.request.kinematics_pose.pose.position.x = kinematics_pose.at(0);
  srv.request.kinematics_pose.pose.position.y = kinematics_pose.at(1);
  srv.request.kinematics_pose.pose.position.z = kinematics_pose.at(2);
  srv.request.path_time = path_time;

  if (goal_task_space_path_from_present_position_only_client_.call(srv))
  {
    return srv.response.is_planned;
  }
  return false;
}

void OpenManipulatorTeleop::printText()
{
  printf("\n");
  printf("---------------------------------\n");
  printf("Control Your OpenMANIPULATOR-P!\n");
  printf("---------------------------------\n");
  printf("w : increase x axis in task space\n");
  printf("s : decrease x axis in task space\n");
  printf("a : increase y axis in task space\n");
  printf("d : decrease y axis in task space\n");
  printf("z : increase z axis in task space\n");
  printf("x : decrease z axis in task space\n");
  printf("\n");
  printf("r : increase joint 1 angle\n");
  printf("f : decrease joint 1 angle\n");
  printf("t : increase joint 2 angle\n");
  printf("g : decrease joint 2 angle\n");
  printf("y : increase joint 3 angle\n");
  printf("h : decrease joint 3 angle\n");
  printf("u : increase joint 4 angle\n");
  printf("j : decrease joint 4 angle\n");
  printf("i : increase joint 5 angle\n");
  printf("k : decrease joint 5 angle\n");
  printf("o : increase joint 6 angle\n");
  printf("l : decrease joint 6 angle\n");
  printf("       \n");
  if (with_gripper_)
  {
    printf("\n");
    printf("v : gripper open\n");
    printf("b : gripper close\n");    
  }
  printf("1 : init pose\n");
  printf("2 : home pose\n");
  printf("       \n");
  printf("q to quit\n");
  printf("-------------------------------------------------------------------------------\n");
  printf("Present Joint Angle J1: %.3lf J2: %.3lf J3: %.3lf J4: %.3lf J5: %.3lf J6: %.3lf\n",
         getPresentJointAngle().at(0),
         getPresentJointAngle().at(1),
         getPresentJointAngle().at(2),
         getPresentJointAngle().at(3),
         getPresentJointAngle().at(4),
         getPresentJointAngle().at(5));
  printf("Present Kinematics Position X: %.3lf Y: %.3lf Z: %.3lf\n",
         getPresentKinematicsPose().at(0),
         getPresentKinematicsPose().at(1),
         getPresentKinematicsPose().at(2));
  printf("-------------------------------------------------------------------------------\n");
}

void OpenManipulatorTeleop::setGoal(int joint_0, int joint_1, int joint_2, int joint_4, int joint_5)
{
  std::vector<double> goalPose;  goalPose.resize(3, 0.0);
  std::vector<double> goalJoint; goalJoint.resize(6, 0.0);
  double path_time = 2.0;

  if (joint_0 == 0, joint_1 == 0, joint_2 == 0, joint_4 == 0, joint_5 == 0) {
    path_time = 4.0;
  }

  std::vector<std::string> joint_name;
  std::vector<double> joint_angle;

  joint_name.push_back("joint1"); joint_angle.push_back(float(joint_0)/1000);
  joint_name.push_back("joint2"); joint_angle.push_back(float(joint_1)/1000);
  joint_name.push_back("joint3"); joint_angle.push_back(float(joint_2)/1000);
  joint_name.push_back("joint4"); joint_angle.push_back(0.0);
  joint_name.push_back("joint5"); joint_angle.push_back(float(joint_4)/1000);
  joint_name.push_back("joint6"); joint_angle.push_back(float(joint_5)/1000);

  setJointSpacePath(joint_name, joint_angle, path_time);
}

void OpenManipulatorTeleop::restoreTerminalSettings(void)
{
  tcsetattr(0, TCSANOW, &oldt_);  /* Apply saved settings */
}

void OpenManipulatorTeleop::disableWaitingForEnter(void)
{
  struct termios newt;

  tcgetattr(0, &oldt_);  /* Save terminal settings */
  newt = oldt_;  /* Init new settings */
  newt.c_lflag &= ~(ICANON | ECHO);  /* Change settings */
  tcsetattr(0, TCSANOW, &newt);  /* Apply settings */
}

bool acclaim(open_manipulator_p_teleop::Angle::Request &req, open_manipulator_p_teleop::Angle::Response &res)
{
  OpenManipulatorTeleop openManipulatorTeleop;
  openManipulatorTeleop.setGoal((int)req.joint_0, (int)req.joint_1, (int)req.joint_2, (int)req.joint_4, (int)req.joint_5);

  return true;
}

bool judge(open_manipulator_p_teleop::Grab::Request &req, open_manipulator_p_teleop::Grab::Response &res)
{
  OpenManipulatorTeleop openManipulatorTeleop;

  if (req.grab == 1) {
      std::vector<double> joint_angle;
      joint_angle.push_back(1.1);
      openManipulatorTeleop.setToolControl(joint_angle);
  } else {
      std::vector<double> joint_angle;
      joint_angle.push_back(0.0);
      openManipulatorTeleop.setToolControl(joint_angle);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "open_manipulator_p_teleop_keyboard");
  ros::NodeHandle nh;

  ros::ServiceServer angle_server = nh.advertiseService("path", acclaim);
  ros::ServiceServer grab_server = nh.advertiseService("grab", judge);
  
  ROS_INFO("Ready!");
  ros::spin();
}