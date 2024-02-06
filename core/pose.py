import numpy as np


# todo
def is_vacant(image, goal_pose):
    """
    포즈를 추정한 곳에 이미 차가 세워져있는지 판단하는 작업

    Args:
        image ([type]): [description]
        goal_pose ([type]): [description]

    Returns:
        [type]: [description]
    """
    loc = goal_pose["loc"]
    print(image[np.int32(loc[1])][np.int32(loc[0])])
    
    if np.array_equal(image[np.int32(loc[1])][np.int32(loc[0])], [128, 64, 128]):
        return True
    return False


def infer_goal_pose(slot, type="front"):
    """
    주차 공간에 대한 정보를 받아 주차가 완료되었을 때의 예상 pose를 return

    전면 주차 상황: 차의 머리가 inner를 향하게
    후면 주차 상황: 차의 머리가 outer를 향하게
    return { 'loc': (x, y), 'direction': float }
    """
    locations = np.array([corner.loc for corner in slot])
    outer_corners_locations = np.array([locations[0], locations[2]])
    inner_corners_locations = np.array([locations[1], locations[3]])

    direction_vector = None
    center_of_slot = np.sum(locations, 0) / 4
    if type == "front":  # 전면 주차 상황: inner pair를 향하게
        center_of_inner_pair = np.sum(outer_corners_locations, 0) / 2
        direction_vector = np.subtract(
            center_of_inner_pair, center_of_slot
        ) / np.linalg.norm(np.subtract(center_of_slot, center_of_inner_pair))
    else:  # 후면 주차 상황: outer pair를 향하게
        center_of_outer_pair = np.sum(inner_corners_locations, 0) / 2
        direction_vector = np.subtract(
            center_of_outer_pair, center_of_slot
        ) / np.linalg.norm(np.subtract(center_of_slot, center_of_outer_pair))

    return {
        "loc": center_of_slot,
        "direction": [direction_vector[0], direction_vector[1]],
    }
