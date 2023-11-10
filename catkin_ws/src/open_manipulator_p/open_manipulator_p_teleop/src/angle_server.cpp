#include "ros/ros.h"
#include "open_manipulator_p_teleop/Angle.h"

bool acclaim(open_manipulator_p_teleop::Angle::Request &req, open_manipulator_p_teleop::Angle::Response &res)
{
    ROS_INFO("radians: 0=%d, 1=%d, 2=%d, 4=%d, 5=%d",(int)req.joint_0, (int)req.joint_1, (int)req.joint_2, (int)req.joint_4, (int)req.joint_5);
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "angle_server");
    ros::NodeHandle nh;
    ros::ServiceServer server = nh.advertiseService("path", acclaim);
    ROS_INFO("Ready!");
    ros::spin();
    return 0;
}