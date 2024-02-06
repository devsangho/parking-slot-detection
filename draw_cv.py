import cv2
import numpy as np


def draw_using_cv(
    img,
    visible_outer_corners,
    visible_inner_corners,
    auxiliary_inner_corners,
    outer_pairs,
    slots,
    goals,
):
    if img is None:
        img = np.zeros((500, 500, 3))
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
    for goal in goals:
        loc = goal['loc']
        direction = goal['direction']
        dx = direction[0] * 20
        dy = direction[1] * 20
        
        destination_x, destination_y = loc[0] + dx, loc[1] + dy
        cv2.arrowedLine(
            img,
            np.int32(loc),
            np.int32([destination_x, destination_y]),
            (0, 255, 255),
            3,
            tipLength=0.3,
        )
    cv2.imshow(f"detected slots: {len(slots)}, outer pairs: {len(outer_pairs)}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
