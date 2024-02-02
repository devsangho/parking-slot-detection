from typing import List
from corner_vector import Corner
import numpy as np
from data.sample_ideal import (
    ideal_parallel_situation_without_aux_corner,
    ideal_perpendicular_situation_without_aux_corner,
    ideal_opposite_situation_without_aux_corner,
    ideal_other_situation_without_aux_corner,
)
from data.sample_aux import (
    ideal_parallel_situation_with_aux_corner,
)
import cv2
import random
import time
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

#####
# 1. 최종 필터링을 area가 아니라 겹치는 여부로 판단해야할 것 같다.
# 2. 기타 주차 상황에선 방향이 서로 같고 거리로 paring


def exit():
    pass


THRESHOLD_OF_ANGLE = 20  # [%]
MIN_AREA = 2750
MAX_AREA = 8000


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


def convert_to_corners(image, labels) -> Corner:
    """이미지와 box를 받아서 box의 중점(loc)과 line detection 알고리즘을 통해 directions를 도출한다.

    Args:
        image ([type]): [description]
        label ([type]): [description]
        box ([type]): [description]

    Returns:
        Corner: [description]
    """
    visible_outer_corners = []
    visible_inner_corners = []
    auxiliary_inner_corners = []

    def bgrExtraction(image, bgrLower, bgrUpper):
        img_mask = cv2.inRange(image, bgrLower, bgrUpper)
        result = cv2.bitwise_and(image, image, mask=img_mask)
        return result

    target_color = [157, 243, 50]
    tolerance = 20
    lower_color = np.array(
        [
            max(0, target_color[0] - tolerance),
            max(0, target_color[1] - tolerance),
            max(0, target_color[2] - tolerance),
        ]
    )
    upper_color = np.array(
        [
            min(255, target_color[0] + tolerance),
            min(255, target_color[1] + tolerance),
            min(255, target_color[2] + tolerance),
        ]
    )

    extracted_image = bgrExtraction(image, lower_color, upper_color)
    for cls, cx, cy, w, h in labels:
        center_x = 480 * cx
        center_y = 480 * cy
        size = 480 * w

        top_left_x = int(center_x - size / 2)
        top_left_y = int(center_y - size / 2)
        bottom_right_x = int(center_x + size / 2)
        bottom_right_y = int(center_y + size / 2)

        cropped_image = extracted_image[
            top_left_y:bottom_right_y, top_left_x:bottom_right_x
        ]
        # print('cropped_image shape:', cropped_image.shape)

        type = None
        if cls == 0:
            type = "out"
        elif cls == 1:
            type = "in"
        elif cls == 3:
            type = "aux_in"
        else:
            raise ("Not supported type.")
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 50, 200, apertureSize=3)

        # cv2.imshow(f'{type}, [{center_x:.2f}, {center_y:.2f}]', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        angles = []
        for y in range(gray.shape[0]):  # ok
            for x in range(gray.shape[1]):  # ok
                if gray[y, x] >= 170:
                    dx = x - 10  # 원점에서부터 계산
                    dy = y - 10
                    angle = np.arctan2(dy, dx)
                    angle = np.degrees(angle)
                    angles.append(angle)
        # for y in range(edges.shape[0]):
        #     for x in range(edges.shape[1]):
        #         if edges[y, x] == 255:
        #             dx = 10 - x
        #             dy = 10 - y  # 이미지는 위에서 아래로 진행하므로 y 방향 반전 필요
        #             angle = np.arctan2(dy, dx)
        #             # 각도를 0부터 2*pi까지의 범위로 변환
        #             angle = (
        #                 np.degrees(angle) + 180
        #             )  # 각도를 라디안에서 도로 변환한 후, 180을 더하여 양수로 만듦
        #             # angle = np.degrees(angle)
        #             angles.append(angle)

        # 각도 히스토그램 계산 (0부터 360도까지의 범위로)
        histogram, bins = np.histogram(angles, bins=120, range=(-180.00, 180.00))

        # 임계값을 넘는 각도의 인덱스 찾기
        # threshold = 2.5  # 임의의 임계값 (원하는 값으로 설정 가능)
        threshold = 2
        threshold_indices = np.where(histogram > threshold)[0]

        # 인덱스에 해당하는 각도와 빈도수 추출
        selected_angles = bins[:-1][threshold_indices]
        selected_frequencies = histogram[threshold_indices]

        # 빈도수를 기준으로 내림차순으로 정렬하여 상위 3개의 각도 선택
        # sorted_indices = np.argsort(selected_frequencies)[::-1][:4]
        sorted_indices = np.argsort(selected_frequencies)[::-1]
        top_3_angles = selected_angles[sorted_indices]
        top_3_frequencies = selected_frequencies[sorted_indices]

        # print("상위 3개의 각도 (degrees):", top_3_angles)
        # print("상위 3개의 빈도수:", top_3_frequencies)

        # 히스토그램 그리기
        # plt.bar(bins[:-1], histogram, width=1.0)
        # plt.title('Angle Distribution around Center (Frequency > 1)')
        # plt.xlabel(f'Angle {type}')
        # plt.ylabel('Frequency')
        # plt.xlim(-180, 180)
        # plt.show()
        corner = Corner(
            type=type,
            loc=(np.float32(center_x), np.float32(center_y)),
            directions=[
                [
                    np.float32(np.cos(np.deg2rad(angle))),
                    np.float32(np.sin(np.deg2rad(angle))),
                ]
                for angle in top_3_angles
            ],
        )
        if type == "out":
            visible_outer_corners.append(corner)
        elif type == "in":
            visible_inner_corners.append(corner)
        elif type == "aux_in":
            auxiliary_inner_corners.append(corner)
    return visible_outer_corners, visible_inner_corners, auxiliary_inner_corners


def is_pairing_possible_between_two_outer_corners(
    corner1: Corner, corner2: Corner
) -> bool:
    """두 개의 outer 코너가 paring 가능 여부

    Args:
        corner1 ([type]): [description]
        corner2 ([type]): [description]

    Returns:
        bool: [description]
    """

    if len(corner1.directions) == 0 and len(corner2.directions) == 0:
        raise ("direction is empty.")

    distance = get_distance_between_two_points(corner1.loc, corner2.loc)

    # 기타 주차 상황에선 방향이 서로 같고 거리로 paring
    if len(corner1.directions) == 1 and len(corner2.directions) == 1:
        MIN_DISTANCE = 45
        MAX_DISTANCE = 70
        if MIN_DISTANCE < distance < MAX_DISTANCE and is_same_direction(
            corner1.directions[0], corner2.directions[0]
        ):
            return True
        else:
            return False

    # 일반적인 상황(직각, 평행, 대향, 교차)
    else:
        direction_from_corner1_to_corner2 = np.float32(
            np.float32(np.array(corner2.loc) - np.array(corner1.loc)) / distance
        )
        for corner1_direction in corner1.directions:
            for corner2_direction in corner2.directions:
                if is_opposite_direction(
                    corner1_direction, corner2_direction
                ) and is_same_direction(
                    direction_from_corner1_to_corner2, corner1_direction
                ):
                    return True
        return False


def is_pairing_possible_between_outer_corner_and_inner_corner(
    outer_corner: Corner, inner_corner: Corner
) -> bool:
    """outer, inner 코너 간 paring 가능 여부

    Args:
        corner1 ([type]): [description]
        corner2 ([type]): [description]

    Returns:
        bool: [description]
    """

    if len(outer_corner.directions) == 0 and len(inner_corner.directions) == 0:
        raise ("direction is empty.")

    distance = get_distance_between_two_points(outer_corner.loc, inner_corner.loc)

    direction_from_outer_corner_to_inner_corner = np.float32(
        np.float32(np.array(inner_corner.loc) - np.array(outer_corner.loc)) / distance
    )
    for outer_direction in outer_corner.directions:
        for inner_direction in inner_corner.directions:
            if is_opposite_direction(
                outer_direction, inner_direction
            ) and is_same_direction(
                direction_from_outer_corner_to_inner_corner, outer_direction
            ):
                return True
    return False


def pairing_between_outer_corners(corners: List[Corner]):
    """
    outer vector들을 pairing 한다.

    Args:
        corners (List[Corner]): [description]

    Returns:
        [type]: [description]
    """
    pairs = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            # 비교하기 전에 같은 방향으로 이미 min 된게 있는지 판단해야함
            if is_pairing_possible_between_two_outer_corners(
                corner1=corners[i], corner2=corners[j]
            ):
                indices = [
                    index for index, pair in enumerate(pairs) if pair[0] == corners[i]
                ]
                # 같은 선상에 있는 pair가 여러개일 수 있다. 이럴 때에는 가장 거리가 짧은 것을 택한다.
                if len(indices) > 0:  # 이미 있는 경우
                    index_of_already_paired = indices[0]
                    already_paired = pairs[index_of_already_paired]
                    distance_of_already_paired = get_distance_between_two_points(
                        corners[i].loc, already_paired[1].loc
                    )
                    distance_of_new_paired = get_distance_between_two_points(
                        corners[i].loc, corners[j].loc
                    )
                    if distance_of_new_paired < distance_of_already_paired:
                        removed = pairs.pop(index_of_already_paired)
                        new_pair = [corners[i], corners[j]]
                        pairs.append(new_pair)
                else:
                    pairs.append([corners[i], corners[j]])
    return pairs


def estimate_inner_corner(outer_corner, auxiliary_inner_corner, is_outer_pair_long):
    length = 45 if is_outer_pair_long else 100  # 임시
    dx, dy = np.array(auxiliary_inner_corner.loc) - np.array(outer_corner.loc)
    angle = np.arctan2(dy, dx)
    estimated_loc = np.array(outer_corner.loc) + np.array(
        (
            length * np.cos(angle),
            length * np.sin(angle),
        )
    )
    estimated_inner_corner = Corner(
        type="in",
        loc=(estimated_loc[0], estimated_loc[1]),
        directions=[],
    )
    return estimated_inner_corner


def estimate_inner_corner_from_other_pair(
    outer_corner, vector_from_outer_to_inner_point
):
    estimated_loc = np.array(outer_corner.loc) + np.array(
        vector_from_outer_to_inner_point
    )
    estimated_inner_corner = Corner(
        type="in",
        loc=(estimated_loc[0], estimated_loc[1]),
        directions=[],
    )
    return estimated_inner_corner


def pairing_between_outer_pair_and_inner_vectors(
    outer_pair: List[Corner],
    inner_corners: List[Corner],
    auxiliary_inner_corners: List[Corner],
):
    """
    outer pair와 inner vectors 중에 각각 outer pair의 방향과 거리를 고려하여 slot을 결정한다.
    return [slot]
    slot = [pt1, pt2, pt3, pt4]
    """
    slot = []
    pair1 = []
    pair2 = []

    outer_corner1, outer_corner2 = outer_pair

    LENGTH_OF_SHORT_OUTER_PAIR = 100
    is_outer_pair_long = (
        True
        if LENGTH_OF_SHORT_OUTER_PAIR
        < get_distance_between_two_points(outer_corner1.loc, outer_corner2.loc)
        else False
    )
    MIN_DISTANCE_BETWEEN_OUTER_AND_INNER = 50 if is_outer_pair_long else 80
    MAX_DISTANCE_BETWEEN_OUTER_AND_INNER = 100 if is_outer_pair_long else 100
    if len(inner_corners) == 0 and len(auxiliary_inner_corners) == 0:
        return None

    # 우선 inner_corner 중에 매칭되는 것이 있는지 파악한다.
    for inner_corner in inner_corners:
        if is_pairing_possible_between_outer_corner_and_inner_corner(
            outer_corner1, inner_corner
        ):
            distance_of_new_paired = get_distance_between_two_points(
                outer_corner1.loc, inner_corner.loc
            )
            if len(pair1) > 0:
                already_paired = pair1
                distance_of_already_paired = get_distance_between_two_points(
                    outer_corner1.loc, already_paired[1].loc
                )
                if (
                    MIN_DISTANCE_BETWEEN_OUTER_AND_INNER
                    < distance_of_new_paired
                    < distance_of_already_paired
                ):
                    pair1 = [outer_corner1, inner_corner]
            elif (
                MIN_DISTANCE_BETWEEN_OUTER_AND_INNER
                < distance_of_new_paired
                < MAX_DISTANCE_BETWEEN_OUTER_AND_INNER
            ):
                pair1 = [outer_corner1, inner_corner]
        if is_pairing_possible_between_outer_corner_and_inner_corner(
            outer_corner2, inner_corner
        ):
            distance_of_new_paired = get_distance_between_two_points(
                outer_corner2.loc, inner_corner.loc
            )
            if len(pair2) > 0:
                already_paired = pair2
                distance_of_already_paired = get_distance_between_two_points(
                    outer_corner2.loc, already_paired[1].loc
                )
                if (
                    MIN_DISTANCE_BETWEEN_OUTER_AND_INNER
                    < distance_of_new_paired
                    < distance_of_already_paired
                ):
                    pair2 = [outer_corner2, inner_corner]
            elif (
                MIN_DISTANCE_BETWEEN_OUTER_AND_INNER
                < distance_of_new_paired
                < MAX_DISTANCE_BETWEEN_OUTER_AND_INNER
            ):
                pair2 = [outer_corner2, inner_corner]
    # 만약 inner_corner로 페어링 안됐다면, aux 점들 중에 찾아본다.
    if len(pair1) == 0:
        if len(pair2) == 2:  # 만약 다른 pair가 결정된 것이 있다면 여기서 사용한 거리 데이터를 그대로 사용
            vector_from_outer_to_inner_point = np.array(pair2[1].loc) - np.array(
                pair2[0].loc
            )
            estimated_inner_corner = estimate_inner_corner_from_other_pair(
                outer_corner1,
                vector_from_outer_to_inner_point,
            )
            # print("이미 맺어진 페어가 있군요.. 이것을 참고하겠습니다..")
            pair1 = [outer_corner1, estimated_inner_corner]
        else:  # 다른 pair도 비어있다면, aux 탐색
            already_paired_aux_inner_corner = None
            for auxiliary_inner_corner in auxiliary_inner_corners:
                if is_pairing_possible_between_outer_corner_and_inner_corner(
                    outer_corner1, auxiliary_inner_corner
                ):
                    estimated_inner_corner = estimate_inner_corner(
                        outer_corner1, auxiliary_inner_corner, is_outer_pair_long
                    )
                    already_paired = pair1
                    if len(already_paired) == 0:  # 추정한 것이 없다면,
                        # print("추정해둔 것이 없군요.. 추가하겠습니다..")
                        pair1 = [outer_corner1, estimated_inner_corner]
                        already_paired_aux_inner_corner = auxiliary_inner_corner
                    else:  # 추정한 것이 이미 있다면,
                        distance_of_already_paired = np.round(
                            get_distance_between_two_points(
                                outer_corner1.loc, already_paired[1].loc
                            ),
                            5,
                        )
                        distance_of_new_paired = np.round(
                            get_distance_between_two_points(
                                outer_corner1.loc, estimated_inner_corner.loc
                            ),
                            5,
                        )
                        # distance_of_already_paired = get_distance_between_two_points(
                        #     outer_corner1.loc, already_paired_aux_inner_corner.loc
                        # )
                        # distance_of_new_paired = get_distance_between_two_points(
                        #     outer_corner1.loc, auxiliary_inner_corner.loc
                        # )
                        # print("추정해둔 것이 있군요...", distance_of_new_paired, distance_of_already_paired)
                        if distance_of_new_paired < distance_of_already_paired:
                            pair1 = [outer_corner1, estimated_inner_corner]
                            already_paired_aux_inner_corner = auxiliary_inner_corner
                            # print("기존 것보다 더 좋은 페어군요.. 업데이트합니다...")
                        else:
                            pass
                            # print("기존 것보다 안좋군요.. 무시합니다..")
    if len(pair2) == 0:
        if len(pair1) == 2:  # 만약 다른 pair가 결정된 것이 있다면 여기서 사용한 거리 데이터를 그대로 사용
            vector_from_outer_to_inner_point = np.array(pair1[1].loc) - np.array(
                pair1[0].loc
            )
            estimated_inner_corner = estimate_inner_corner_from_other_pair(
                outer_corner2,
                vector_from_outer_to_inner_point,
            )
            pair2 = [outer_corner2, estimated_inner_corner]
        else:  # 다른 pair도 비어있다면, aux 탐색
            already_paired_aux_inner_corner = None
            for auxiliary_inner_corner in auxiliary_inner_corners:
                if is_pairing_possible_between_outer_corner_and_inner_corner(
                    outer_corner2, auxiliary_inner_corner
                ):
                    estimated_inner_corner = estimate_inner_corner(
                        outer_corner2, auxiliary_inner_corner, is_outer_pair_long
                    )
                    already_paired = pair2
                    if len(already_paired) == 0:  # 추정한 것이 없다면
                        pair1 = [outer_corner2, estimated_inner_corner]
                        already_paired_aux_inner_corner = auxiliary_inner_corner
                    else:  # 추정한 것이 이미 있다면,
                        distance_of_already_paired = np.round(
                            get_distance_between_two_points(
                                outer_corner2.loc, already_paired[1].loc
                            ),
                            5,
                        )
                        distance_of_new_paired = np.round(
                            get_distance_between_two_points(
                                outer_corner2.loc, estimated_inner_corner.loc
                            ),
                            5,
                        )
                        # distance_of_already_paired = get_distance_between_two_points(
                        #     outer_corner2.loc, already_paired_aux_inner_corner.loc
                        # )
                        # distance_of_new_paired = get_distance_between_two_points(
                        #     outer_corner2.loc, auxiliary_inner_corner.loc
                        # )
                        if distance_of_new_paired < distance_of_already_paired:
                            pair1 = [outer_corner2, estimated_inner_corner]
                            already_paired_aux_inner_corner = auxiliary_inner_corner
    slot = [*pair1, *pair2]
    # 최종적으로 slot안에 들어가있는 코너의 갯수가 4개인 것들만 return
    if len(slot) == 4:
        # outer_corner1, _, _, inner_corner2 = slot
        # 4점이 모두 같은 선상에 있는 경우는 엣지 케이스이므로 제외
        # if is_pairing_possible_between_outer_corner_and_inner_corner(
        #     outer_corner1,
        #     inner_corner2,
        # ):
        #     print("같은 선상에 있소이다.")
        #     return None
        pts = [corner.loc for corner in slot]
        area = get_area_from_four_points([pts[0], pts[2], pts[3], pts[1]])
        print("area", area)
        if MIN_AREA < area < MAX_AREA:
            return slot
        # return slot
    return None


# todo
def infer_goal_pose(slot, type="front"):
    """
    주차 공간에 대한 정보를 받아 주차가 완료되었을 때의 예상 pose를 return

    전면 주차 상황: 차의 머리가 inner를 향하게
    후면 주차 상황: 차의 머리가 outer를 향하게
    return { 'loc': (x, y), 'direction': float }
    """
    if type == "front":  # 전면 주차 상황
        pass
    else:  # 후면 주차 상황
        pass


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


def draw_using_cv(
    img,
    visible_outer_corners,
    visible_inner_corners,
    auxiliary_inner_corners,
    outer_pairs,
    slots,
):
    if img is None:
        img = np.zeros((480, 480, 3))
    for corner in visible_outer_corners:
        # cv2.circle(img, np.int32(corner.loc), 10, (0, 0, 255), 1)
        for direction in corner.directions:
            dx = direction[0] * 20
            dy = direction[1] * 20
            destination_x, destination_y = corner.loc[0] + dx, corner.loc[1] + dy
            cv2.arrowedLine(
                img,
                np.int32(corner.loc),
                np.int32([destination_x, destination_y]),
                (0, 0, 255),
                1,
                tipLength=0.3,
            )
        cv2.putText(
            img,
            f"{np.int32(corner.loc)}",
            np.int32(np.array(corner.loc) - 10),
            0,
            0.5,
            (255, 255, 255),
            1,
        )
    for corner in visible_inner_corners:
        # cv2.circle(img, np.int32(corner.loc), 10, (0, 255, 255), 1)
        for direction in corner.directions:
            dx = direction[0] * 20
            dy = direction[1] * 20
            destination_x, destination_y = corner.loc[0] + dx, corner.loc[1] + dy
            cv2.arrowedLine(
                img,
                np.int32(corner.loc),
                np.int32([destination_x, destination_y]),
                (0, 255, 255),
                1,
                tipLength=0.3,
            )
        cv2.putText(
            img,
            f"{np.int32(corner.loc)}",
            np.int32(corner.loc),
            0,
            0.3,
            (255, 255, 255),
            1,
        )
    for corner in auxiliary_inner_corners:
        # cv2.circle(img, np.int32(corner.loc), 10, (0, 120, 255), 1)
        for direction in corner.directions:
            dx = direction[0] * 20
            dy = direction[1] * 20
            destination_x, destination_y = corner.loc[0] + dx, corner.loc[1] + dy
            cv2.arrowedLine(
                img,
                np.int32(corner.loc),
                np.int32([destination_x, destination_y]),
                (0, 255, 255),
                1,
                tipLength=0.3,
            )
        cv2.putText(
            img,
            f"aux {np.int32(corner.loc)}",
            np.int32(corner.loc),
            0,
            0.3,
            (255, 255, 255),
            1,
        )
    for slot in slots:
        pts = np.array(
            [[np.int32(corner.loc[0]), np.int32(corner.loc[1])] for corner in slot]
        )
        print(pts)
        # for pt in pts:
        # cv2.circle(img, pt, 10, (255, 255, 255), 2)
        # cv2.fillConvexPoly(
        #     img, np.array([pts[0], pts[2], pts[3], pts[1]]), (255, 255, 255)
        # )
        cv2.polylines(
            img,
            [np.array([pts[0], pts[2], pts[3], pts[1]])],
            True,
            (255, 20, 255),
            3,
        )
    for pair in outer_pairs:
        cv2.line(
            img,
            np.int32(pair[0].loc),
            np.int32(pair[1].loc),
            (0, 128, 0),
            2,
        )
    cv2.imshow(f"detected slots: {len(slots)}, outer pairs: {len(outer_pairs)}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    image = np.array(cv2.imread("data/images/1/image_002.png"))
    label = []

    with open("data/labels/1/image_002.txt", "r") as file:
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
        # end = time.time()
        # print("걸린 시간:", end - start, "sec")
        draw_using_cv(
            image,
            visible_outer_corners,
            visible_inner_corners,
            auxiliary_inner_corners,
            outer_pairs,
            slots,
        )
        # goals = [infer_goal_pose(slot) for slot in slots]
        # publish_goal_pose_to_ros2(goals)
        # pass
        break  # 임시

    exit()
