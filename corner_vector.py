# from typing import List
# import numpy as np


class Corner:
    def __init__(self, type, loc, directions) -> None:
        """AI is creating summary for __init__

        Args:
            type (str): `out` | `in` | `aux_out`
            loc ((np.float32, np.float32)): cv2 좌표계 기준
            directions (List[(np.float32, np.float32)]): 크기가 1인 방향벡터들의 list

        Raises:
            ValueError: [description]
        """
        if type not in ["out", "in", "aux_in"]:
            raise ValueError("`type` of Corner must be `out` or `in` or `aux_in`.")
        self.type = type
        self.loc = loc
        self.directions = directions
