from typing import List
from corner_vector import Corner
import numpy as np
from core.math import (
    is_opposite_direction,
    is_same_direction,
    get_area_from_four_points,
    get_distance_between_two_points,
)
from core.aux import estimate_inner_corner_from_other_pair, estimate_inner_corner

MIN_AREA = 2750
MAX_AREA = 8000


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
