# Project README

## Overview

The aim of this project is to leverage Artificial Intelligence for state estimation. Specifically, we will implement various neural network architectures such as LSTM and Transformers to predict changes in heading angle and displacement. This prediction will be based on IMU data recorded using the ROS Gazebo simulator.

We will utilize TurtleBot3 as the mobile robot to gather measurements within the Gazebo environment. For ease of use, we have developed a Python script, `turtlebot3_waypoints.py`, to facilitate the recording of bagfiles from the simulations.

## Steps

To achieve our objectives, we will follow these steps:

1. **Launch TurtleBot3 inside Gazebo**: Begin by initializing TurtleBot3 within the Gazebo simulation environment.

2. **Run Navigation Node**: Execute the Navigation node to perform Initial Pose estimation and gather information about the surrounding environment.

3. **Launch Recording Script**: Utilize the script `turtlebot3_waypoints.py` to record the bagfile generated during the simulation.

4. **Index Bagfiles**: Index all obtained bagfiles to optimize storage space and facilitate access.

5. **Launch Neural Network**: Run the neural network with the collected data and record the results for analysis.

By following these steps, we aim to develop a robust system for state estimation using AI techniques, with a focus on neural network architectures like LSTM and Transformers. This README will guide users through the process and provide necessary instructions to replicate the project setup and results.
