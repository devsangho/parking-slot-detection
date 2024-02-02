from corner_vector import Corner
import numpy as np

"""
샘플 데이터:

이상적인 상황을 가정합니다.
1. 코너 포인트가 잘 추출되었으며,
2. 이로부터 방향벡터 또한 잘 추출되었다.

resolution: 480 * 480 [px]
coordinate: opencv ----- x
                   |
                   |
                   y
"""

"""
ideal_parallel_situation_with_aux_corner
label: (index of out corner), [index of inner corner]

[0]----(0)  (4)----[4]
 |      |    |      |
[1]----(1)  (5)----[5]
 |      |    |      |
[2]----(2)  (6)----[6]
 |      |    |      |
[3]----(3)  (7)----[7]
"""
ideal_parallel_situation_with_aux_corner = {
    "out": [
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(120)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(180)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(300)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(120)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(180)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(300)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
    ],
    "in": [
        # Corner(
        #     type="in",
        #     loc=(np.float32(60), np.float32(300)),
        #     directions=[
        #         (np.float32(1), np.float32(0)),
        #         (np.float32(0), np.float32(-1)),
        #     ],
        # ),
        # Corner(
        #     type="in",
        #     loc=(np.float32(420), np.float32(120)),
        #     directions=[
        #         (np.float32(-1), np.float32(0)),
        #         (np.float32(0), np.float32(1)),
        #     ],
        # ),
        # Corner(
        #     type="in",
        #     loc=(np.float32(420), np.float32(180)),
        #     directions=[
        #         (np.float32(0), np.float32(-1)),
        #         (np.float32(-1), np.float32(0)),
        #         (np.float32(0), np.float32(1)),
        #     ],
        # ),
        # Corner(
        #     type="in",
        #     loc=(np.float32(420), np.float32(240)),
        #     directions=[
        #         (np.float32(0), np.float32(-1)),
        #         (np.float32(-1), np.float32(0)),
        #         (np.float32(0), np.float32(1)),
        #     ],
        # ),
        # Corner(
        #     type="in",
        #     loc=(np.float32(420), np.float32(300)),
        #     directions=[
        #         (np.float32(-1), np.float32(0)),
        #         (np.float32(0), np.float32(-1)),
        #     ],
        # ),
    ],
    "aux_in": [
        Corner(
            type="aux_in",
            loc=(np.float32(60), np.float32(120)),
            directions=[
                (np.float32(1), np.float32(0)),
            ],
        ),
        Corner(
            type="aux_in",
            loc=(np.float32(60), np.float32(180)),
            directions=[
                (np.float32(1), np.float32(0)),
            ],
        ),
        Corner(
            type="aux_in",
            loc=(np.float32(60), np.float32(240)),
            directions=[
                (np.float32(1), np.float32(0)),
            ],
        ),
    ],
}
