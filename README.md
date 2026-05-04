# How to run this code on a virtual duckiebot
- Navigate to the root of this project
- Make the ros_communication script executable: `chmod +x ./packages/my_package/src/ros_communication.py`
- Create and start virtual duckiebot with name `ROBOT_NAME`
- Run matrix on custom map: `dts matrix run --standalone --map custom_map`
- Attach bot to matrix: `dts matrix attach ROBOT_NAME map_0/vehicle_0`
- Build: `dts devel build -f`
- Run while connecting to the bot's ROS network: `dts devel run -R ROBOT_NAME`
- You can now observe the robot's behavior in the matrix window

# Debugging
- Type `dts gui --vnc ROBOT_NAME` and open the image viewer to get a graphical view of
certain aspects (e.g., lane detection) which are published to corresponding ROS nodes.
- Sometimes, the ROS communication stops working. In this case, restart the robot.

# Template
This repo is based on the [template-ros](https://github.com/duckietown/template-ros/) Duckietown template.
This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

**NOTE:** If you want to develop software that does not use
ROS, check out [this template](https://github.com/duckietown/template-basic).


## How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Create a new repository

Create a new repository on github.com while
specifying the newly forked template repository as
a template for your new repository.


### 3. Define dependencies

List the dependencies in the files `dependencies-apt.txt` and
`dependencies-py3.txt` (apt packages and pip packages respectively).


### 4. Place your code

Place your code in the directory `/packages/` of
your new repository.


### 5. Setup launchers

The directory `/launchers` can contain as many launchers (launching scripts)
as you want. A default launcher called `default.sh` must always be present.

If you create an executable script (i.e., a file with a valid shebang statement)
a launcher will be created for it. For example, the script file 
`/launchers/my-launcher.sh` will be available inside the Docker image as the binary
`dt-launcher-my-launcher`.

When launching a new container, you can simply provide `dt-launcher-my-launcher` as
command.