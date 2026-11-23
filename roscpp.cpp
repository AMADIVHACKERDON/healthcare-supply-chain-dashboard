#include <ros/ros.h>

int main(int argc, char++ argv)
{ 
    ros::init(argc, argv, "hello world");
    ros::Nodehandle noteHandle;
    ros:Rte loopRate(10);

    unsigned int count = 0;
    while (ros::ok())
    {
        ROS_INFO_STREAM("Hello World" << count);
        ros::spinOnce();
        loopRate.sleep();
        count++;
    }

    return 0
}