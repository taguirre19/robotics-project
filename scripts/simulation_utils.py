import numpy as np
import pinocchio as pin
from pink import solve_ik
from pink.tasks import FrameTask
from pinocchio.utils import rotate


def get_parameters_robot(configuration):
    l_ankle_position = configuration.get_transform_frame_to_world("l_ankle").copy().translation
    r_ankle_position = configuration.get_transform_frame_to_world("r_ankle").copy().translation
    pelvis_position = configuration.get_transform_frame_to_world("root_joint").copy().translation
    leg_length = (pelvis_position - l_ankle_position)[-1]
    x_left_foot = l_ankle_position[0] * (-1)
    x_right_foot = r_ankle_position[0] * (-1)
    h_ankle = l_ankle_position[2]
    return leg_length, x_left_foot, x_right_foot, h_ankle

def compute_initial_position(configuration, h_com, T):
    root_task = FrameTask("root_joint", position_cost=1.0, orientation_cost=1.0)
    root_pose = configuration.get_transform_frame_to_world("root_joint").copy()

    root_pose.translation[2] = h_com
    root_task.set_target(pin.SE3(rotate('z', np.pi/2), root_pose.translation))
    velocity = solve_ik(configuration, [root_task], T, solver='daqp')
    configuration.integrate_inplace(velocity, T)

def draw_feet_footprints(foot_trajectory, foot_name, foot_length, foot_width, viz):
    for i,(x,y) in enumerate(foot_trajectory):
        step_name = f'{foot_name}_feet_{i}'
        y+=0.045
        pos = np.array([-x,y,-foot_length/10])
        dims = [foot_width, foot_length, foot_length/10]
        color = 'red'
        viz.addBox(step_name, dims, color)
        placement = pin.SE3(np.eye(3), np.array(pos))
        viz.applyConfiguration(step_name, placement)
