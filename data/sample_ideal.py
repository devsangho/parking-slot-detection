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
ideal_parallel_situation_without_aux_corner
label: (index of out corner), [index of inner corner]

[0]----(0)  (4)----[4]
 |      |    |      |
[1]----(1)  (5)----[5]
 |      |    |      |
[2]----(2)  (6)----[6]
 |      |    |      |
[3]----(3)  (7)----[7]
"""
ideal_parallel_situation_without_aux_corner = {
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
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(120)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(180)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(300)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(120)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(180)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(300)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
    ],
    "aux_in": [],
}

"""
ideal_perpendicular_situation_without_aux_corner
label: (index of out corner), [index of inner corner]

[0]--(0)  (3)--[3]
 |    |    |    |
 |    |    |    |
 |    |    |    |
[1]--(1)  (4)--[4]
 |    |    |    |
 |    |    |    |
 |    |    |    |
[2]--(2)  (5)--[5]
"""
ideal_perpendicular_situation_without_aux_corner = {
    "out": [
        Corner(
            type="out",
            loc=(np.float32(120), np.float32(120)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(120), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(120), np.float32(360)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(360), np.float32(120)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(360), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(360), np.float32(360)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
    ],
    "in": [
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(120)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(360)),
            directions=[
                (np.float32(1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(120)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(240)),
            directions=[
                (np.float32(0), np.float32(-1)),
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(360)),
            directions=[
                (np.float32(-1), np.float32(0)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
    ],
    "aux_in": [],
}

"""
ideal_opposite_situation_without_aux_corner
label: (index of out corner), [index of inner corner]
      (0)
      /  \
     /    \
  (1)      \
   \        \
    \        \
   (2)\     [0]
  /    \    /
 /      \  /
(3)     [1]
  \      \
   \      \
    \     /[2]
     \   /
      [3]
"""
ideal_opposite_situation_without_aux_corner = {
    "out": [
        Corner(
            type="out",
            loc=(np.float32(210), np.float32(210)),
            directions=[
                (np.float32(-np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(-np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(150), np.float32(270)),
            directions=[
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(-np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(210), np.float32(330)),
            directions=[
                (np.float32(-np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(-np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(150), np.float32(270)),
            directions=[
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(-np.cos(np.radians(45))), -np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(150), np.float32(390)),
            directions=[
                (np.float32(-np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(330), np.float32(210)),
            directions=[
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(390), np.float32(270)),
            directions=[
                (np.float32(-np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(-np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(330), np.float32(330)),
            directions=[
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(390), np.float32(390)),
            directions=[
                (np.float32(-np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
            ],
        ),
    ],
    "in": [
        Corner(
            type="in",
            loc=(np.float32(90), np.float32(90)),
            directions=[
                (np.float32(np.cos(np.radians(45))), np.sin(np.radians(45))),
                (np.float32(-np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(30), np.float32(150)),
            directions=[
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(90), np.float32(210)),
            directions=[
                (np.float32(-np.cos(np.radians(45))), np.sin(-np.radians(45))),
                (np.float32(-np.cos(np.radians(45))), np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(30), np.float32(270)),
            directions=[
                (np.float32(np.cos(np.radians(45))), -np.sin(np.radians(45))),
                (np.float32(np.cos(np.radians(45))), np.sin(np.radians(45))),
            ],
        ),
    ],
    "aux_in": [],
}

"""
ideal_other_situation_without_aux_corner
label: (index of out corner), [index of inner corner]
(0)                   (0)
|   \                /  |
|    \              /   |
|     \            /    |
(0)    \          /   (0)  
|   \    [0]    [0]   / |
|    \               /  |
|     \             /   |
(0)    \           /  (0)
|   \    [0]    [0]   / |
|    \               /  |
|     \             /   |
(0)    \           /  (0)
|    \   [0]    [0]   /  |
|     \              /  |
|      \            /   |
        \          /    |
         [0]    [0]
         
"""
theta = np.arctan(1 / 2)
ideal_other_situation_without_aux_corner = {
    "out": [
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(180)),
            directions=[
                (np.float32(-np.cos(theta)), -np.sin(theta)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(240)),
            directions=[
                (np.float32(-np.cos(theta)), -np.sin(theta)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(300)),
            directions=[
                (np.float32(-np.cos(theta)), -np.sin(theta)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(180), np.float32(360)),
            directions=[
                (np.float32(-np.cos(theta)), -np.sin(theta)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(180)),
            directions=[
                (np.float32(np.cos(theta)), -np.sin(theta)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(240)),
            directions=[
                (np.float32(np.cos(theta)), -np.sin(theta)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(300)),
            directions=[
                (np.float32(np.cos(theta)), -np.sin(theta)),
            ],
        ),
        Corner(
            type="out",
            loc=(np.float32(300), np.float32(360)),
            directions=[
                (np.float32(np.cos(theta)), -np.sin(theta)),
            ],
        ),
    ],
    "in": [
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(120)),
            directions=[
                (np.float32(np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(180)),
            directions=[
                (np.float32(np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(1)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(240)),
            directions=[
                (np.float32(np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(1)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(60), np.float32(300)),
            directions=[
                (np.float32(np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(120)),
            directions=[
                (np.float32(-np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(180)),
            directions=[
                (np.float32(-np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(1)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(240)),
            directions=[
                (np.float32(-np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(1)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
        Corner(
            type="in",
            loc=(np.float32(420), np.float32(300)),
            directions=[
                (np.float32(-np.cos(theta)), np.sin(theta)),
                (np.float32(0), np.float32(-1)),
            ],
        ),
    ],
    "aux_in": [],
}
