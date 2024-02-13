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

## Results

The best results obtained are the following:
| run | model                                               | epochs | batch size | hidden size | normalized (Y/N) | loss                  | optimizer       | lr                        | train loss | test loss |
| --- | --------------------------------------------------- | ------ | ---------- | ----------- | ---------------- | --------------------- | --------------- | ------------------------- | ---------- | --------- |
| 1   | LSTM                                                | 10     | 32         | 128         | Y                | MSE, reduction='mean' | adam, lr=0.0001 | step_size=200, gamma=0.1  | 0.2224     | 0.4682    |
| 2   | LSTM                                                | 15     | 32         | 256         | Y                | MSE, reduction='mean' | adam, lr=0.0002 | step_size=200, gamma=0.01 | 0.1834     | 0.3798    |
| 3   | LSTM                                                | 50     | 16         | 128         | Y                | MSE, reduction='mean' | adam, lr=0.001  | step_size=300, gamma=0.2  | 0.1353     | 0.3702    |
| 4   | LSTM                                                | 10     | 64         | 128         | Y                | MSE, reduction='mean' | adam, lr=0.0005 | step_size=100, gamma=0.1  | 0.2053     | 0.4048    |
| 5   | LSTM                                                | 25     | 32         | 256         | Y                | MSE, reduction='mean' | adam, lr=0.001  | step_size=300, gamma=0.2  | 0.1603     | 0.3823    |
| 6   | LSTM, num_layers=2, dropout=0.5, bidirectional=True | 10     | 16         | 128         | Y                | MSE, reduction='mean' | adam, lr=0.001  | step_size=300, gamma=0.2  | 0.196      | 0.3863    |
| 7   | LSTM                                                | 20     | 32         | 256         | Y                | MSE, reduction='mean' | adam, lr=0.0002 | step_size=200, gamma=0.01 | 0.1668     | 0.3618    |
| 8   | LSTM, num_layers=3, dropout=0.7, bidirectional=True | 10     | 128        | 64          | Y                | MSE, reduction='mean' | adam, lr=0.0001 | step_size=50, gamma=0.01  | 0.2552     | 0.3938    |
| 9   | Transformer, dropout = 0.5                          | 10     | 128        | 6           | Y                | MSE, reduction='mean' | adam, lr=0.0001 | step_size=200, gamma=0.1  | 0.7458     | 0.6386    |
| 10  | Transformer, dropout = 0.5                          | 20     | 32         | 6           | Y                | MSE, reduction='mean' | adam, lr=0.0001 | step_size=200, gamma=0.01 | 0.448      | 0.504     |
| 11  | Transformer, dropout = 0.5                          | 100    | 32         | 6           | Y                | MSE, reduction='mean' | adam, lr=0.0001 | step_size=200, gamma=0.01 | 0.448      | 0.502     |
| 12  | Transformer, dropout = 0.3                          | 20     | 16         | 6           | Y                | MSE, reduction='mean' | adam, lr=0.001  | step_size=300, gamma=0.2  | 0.389      | 0.4496    |
📋 Copy
Clear
Buy Me a Coffee at ko-fi.com
For the complete view, check the file `results.pdf`.
