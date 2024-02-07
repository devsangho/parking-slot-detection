import numpy as np
import cv2

from core.paring import (
    pairing_between_outer_corners,
    pairing_between_outer_pair_and_inner_vectors,
)
from core.corner import convert_to_corners
from core.pose import infer_goal_pose, is_vacant
from draw_cv import draw_using_cv

from data.sample_ideal import (
    ideal_parallel_situation_without_aux_corner,
    ideal_perpendicular_situation_without_aux_corner,
    ideal_opposite_situation_without_aux_corner,
    ideal_other_situation_without_aux_corner,
)
from data.sample_aux import (
    ideal_parallel_situation_with_aux_corner,
)

#####
# 1. 최종 필터링을 area가 아니라 겹치는 여부로 판단해야할 것 같다.
# 2. 기타 주차 상황에선 방향이 서로 같고 거리로 paring


# todo
def get_image():
    """
    # ros2로 들어온 이미지를 받아 리턴하는 함수
    """
    return np.zeros(480, 480, 3)


# todo
def get_boxes():
    """
    yolo로 들어온 boxes를 리턴하는 함수
    return (visible_outer_boxes, visible_inner_boxes, auxiliary_inner_boxes)
    """
    pass


def is_pre_parking_situation(
    visible_outer_corners, visible_inner_corners, auxiliary_inner_corners
) -> bool:
    """
    주차를 하기 위해 주차장을 돌면서 탐색하고 있는 상황인지를 확인한다.
    이 경우에는 visible outer vectors가 있고, visible_inner_vectors 혹은 auxiliary_inner_vectors가 있는 상황일 것이다.

    Args:
        visible_outer_corners ([Corner.loc]): [description]
        visible_inner_corners ([Corner.loc]): [description]
        auxiliary_inner_corners ([Corner.loc]): [description]

    Returns:
        bool: [description]
    """
    # visible_outer_vectors가 있고,
    if len(visible_outer_corners) < 2:
        return False
    # visible_inner_vectors 혹은 auxiliary_inner_vectors가 있는 상황
    if len(visible_inner_corners) + len(auxiliary_inner_corners) < 2:
        return False
    return True


# todo
def publish_goal_pose_to_ros2(goals):
    """
    rviz로 추론할 goal 포즈 출력
    """

    def fit_coordintate_to_real_world(*args):
        """
        opencv2로 정의된 좌표계를 차를 중심으로 한 xyz 좌표계로 변경
        """
        pass

    def get_quaternion_from_vector(loc, m):
        """
        opencv2로 정의된 좌표계를 차를 중심으로 한 xyz 좌표계로 변경
        """
        q1 = 123
        q2 = 123
        q3 = 123
        q4 = 123
        return [q1, q2, q3, q4]

    msg = [
        [{"loc": 123123, "quaternion": get_quaternion_from_vector(*goal)}]
        for goal in goals
    ]
    pass  # node.pub(~~~)


if __name__ == "__main__":
    # frame = get_image()
    # boxes = get_boxes()
    # visible_outer_boxes, visible_inner_boxes, auxiliary_inner_boxes = get_boxes(frame)

    # ----------------------------------------#
    #
    # ----------------------------------------#
    # visible_outer_corners = set(
    #     [convert_to_corner(image, "out") for image in visible_outer_boxes]
    # )
    # visible_inner_corners = set(
    #     [convert_to_corner(image, "in") for image in visible_inner_boxes]
    # )
    # auxiliary_inner_corners = set(
    #     [convert_to_corner(image, "aux_in") for image in auxiliary_inner_boxes]
    # )

    # ----------------------------------------#
    # FOR TEST
    # ----------------------------------------#
    image = np.array(cv2.imread("data/images/3/image_356.png"))
    label = []

    with open("data/labels/3/image_356.txt", "r") as file:
        for line in file:
            values = [np.float32(val) for val in line.split()]
            label.append(values)
    (
        visible_outer_corners,
        visible_inner_corners,
        auxiliary_inner_corners,
    ) = convert_to_corners(image, label)

    # visible_outer_corners = ideal_parallel_situation_without_aux_corner["out"]
    # visible_inner_corners = ideal_parallel_situation_without_aux_corner["in"]
    # auxiliary_inner_corners = ideal_parallel_situation_without_aux_corner["aux_in"]

    # visible_outer_corners = ideal_perpendicular_situation_without_aux_corner["out"]
    # visible_inner_corners = ideal_perpendicular_situation_without_aux_corner["in"]
    # auxiliary_inner_corners = ideal_perpendicular_situation_without_aux_corner["aux_in"]

    # visible_outer_corners = ideal_opposite_situation_without_aux_corner["out"]
    # visible_inner_corners = ideal_opposite_situation_without_aux_corner["in"]
    # auxiliary_inner_corners = ideal_opposite_situation_without_aux_corner["aux_in"]

    # visible_outer_corners = ideal_other_situation_without_aux_corner["out"]
    # visible_inner_corners = ideal_other_situation_without_aux_corner["in"]
    # auxiliary_inner_corners = ideal_other_situation_without_aux_corner["aux_in"]

    # visible_outer_corners = ideal_parallel_situation_with_aux_corner["out"]
    # visible_inner_corners = ideal_parallel_situation_with_aux_corner["in"]
    # auxiliary_inner_corners = ideal_parallel_situation_with_aux_corner["aux_in"]

    # ----------------------------------------#

    # ----------------------------------------#
    while is_pre_parking_situation(
        visible_outer_corners, visible_inner_corners, auxiliary_inner_corners
    ):
        outer_pairs = pairing_between_outer_corners(visible_outer_corners)
        # start = time.time()
        slots = list(
            filter(
                None,
                [
                    pairing_between_outer_pair_and_inner_vectors(
                        outer_pair, visible_inner_corners, auxiliary_inner_corners
                    )
                    for outer_pair in outer_pairs
                ],
            )
        )
        goals = list(filter(lambda pose: is_vacant(image, pose), [infer_goal_pose(slot) for slot in slots]))
        # end = time.time()
        # print("걸린 시간:", end - start, "sec")
        draw_using_cv(
            image,
            visible_outer_corners,
            visible_inner_corners,
            auxiliary_inner_corners,
            outer_pairs,
            slots,
            goals,
        )
        # goals = [infer_goal_pose(slot) for slot in slots]
        # publish_goal_pose_to_ros2(goals)
        # pass
        break  # 임시

    exit()
