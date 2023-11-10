#include "ros/ros.h"
#include "open_manipulator_p_teleop/Grab.h"

bool judge(open_manipulator_p_teleop::Grab::Request &req, open_manipulator_p_teleop::Grab::Response &res)
{
    ROS_INFO("grab: %d",(int)req.grab);
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "grab_server");
    ros::NodeHandle nh;
    ros::ServiceServer server = nh.advertiseService("grab", judge);
    ROS_INFO("Ready!");
    ros::spin();
    return 0;
}