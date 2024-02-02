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
