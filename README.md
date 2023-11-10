# Classifier

+ **Classifier is autonomous garbage classify software using manipulator.**  
+ **Object that would be classified is determined by YOLOv8 weight file.**  
+ **The workspace and weight files must be edited as desired.**  

### Contact

**tiktaalik135462@gmail.com**  

### History
**22/11/14** manipulation with manual  
**22/12/26** applying pose estimation algoriths  
**23/09/11** making operation sequence more fast  
**23/11/10** transfer to this repository  

### Cloning

**It should be cloned in root directory**

+ etc
+ home
+ root
  + classifier  &larr; **here!**

### Docker

+ [Image](https://hub.docker.com/repository/docker/whataten/classifier/general)

*Add lines to `bash.rc`*  
```
This repository can be used to implement anonther object classifier to using **  
alias cw='cd ~/classifier/catkin_ws'  
alias cs='cd ~/classifier/catkin_ws/src'  
alias cm='cd ~/classifier/catkin_ws && catkin_make'  
source ~/../opt/ros/melodic/setup.bash  
source ~/classifier/catkin_ws/devel/setup.bash  
export ROS_MASTER_URI=http://localhost:11311  
export ROS_HOSTNAME=localhost  
```
