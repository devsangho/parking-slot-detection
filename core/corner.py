from corner_vector import Corner
import numpy as np
import cv2


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

    target_color = [157, 234, 50]
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
        center_x = 500 * cx
        center_y = 500 * cy
        size = 500 * w

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
        threshold = 2.5
        threshold_indices = np.where(histogram > threshold)[0]

        # 인덱스에 해당하는 각도와 빈도수 추출
        selected_angles = bins[:-1][threshold_indices]
        selected_frequencies = histogram[threshold_indices]

        # 빈도수를 기준으로 내림차순으로 정렬하여 상위 3개의 각도 선택
        # sorted_indices = np.argsort(selected_frequencies)[::-1][:4]
        sorted_indices = np.argsort(selected_frequencies)[::-1][:4]
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
