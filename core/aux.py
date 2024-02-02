from corner_vector import Corner
import numpy as np


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
