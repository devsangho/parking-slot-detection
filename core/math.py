import numpy as np
from shapely.geometry import Polygon

THRESHOLD_OF_ANGLE = 20


def is_opposite_direction(
    direction_vector1,
    direction_vector2,
) -> bool:
    """
    방향벡터(크기가 1인) 두개를 받아 서로 반대 방향인지 판단.
    두 벡터의 내적이 -1이면 서로 반대 방향임을 의미.

    Args:
        direction_vector1 ((np.float32, np.float32)): 크기가 1인 방향벡터
        direction_vector2 ((np.float32, np.float32)): 크기가 1인 방향벡터

    Returns:
        bool: [is_opposite_direction]
    """
    if (
        -1 - 1 * THRESHOLD_OF_ANGLE * 0.01
        < np.dot(direction_vector1, direction_vector2)
        < -1 + 1 * THRESHOLD_OF_ANGLE * 0.01
    ):
        return True
    return False


def is_same_direction(
    direction_vector1,
    direction_vector2,
) -> bool:
    """
    방향벡터(크기가 1인) 두개를 받아 서로 같은 방향인지 판단.
    두 벡터의 내적이 1이면 서로 같은 방향임을 의미.

    Args:
        direction_vector1 ((np.float32, np.float32)): 크기가 1인 방향벡터
        direction_vector2 ((np.float32, np.float32)): 크기가 1인 방향벡터

    Returns:
        bool: [is_same_direction]
    """
    if (
        1 - 1 * THRESHOLD_OF_ANGLE * 0.01
        < np.dot(direction_vector1, direction_vector2)
        < 1 + 1 * THRESHOLD_OF_ANGLE * 0.01
    ):
        return True
    return False


def get_distance_between_two_points(pt1, pt2) -> np.float32:
    """두개의 포인트를 받아서, 거리를 계산하여 리턴

    Args:
        pt1 ((np.float32, np.float32)): Corner.loc
        pt2 ((np.float32, np.float32)): Corner.loc

    Returns:
        np.float32: distance
    """
    d = np.linalg.norm(np.array(pt1) - np.array(pt2))
    return d


def get_area_from_four_points(points) -> np.float32:
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    pgon = Polygon(zip(x, y))
    return pgon.area
