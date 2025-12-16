import json
import math

def find_centre(corners):
    x = sum(pt[0] for pt in corners) / 4
    y = sum(pt[1] for pt in corners) / 4
    return [x, y]

def find_distance(A,B):
    return math.hypot((B[0]-A[0]),(B[1]-A[1]))

#die in order
def solve_M1(input_data):
    current_position = input_data["InitialPosition"]
    stage_velocity = input_data["StageVelocity"]
    time = 0
    path = [current_position]
    for die in input_data["Dies"]:
        centre = find_centre(die["Corners"])
        distance = find_distance(current_position,centre)
        t  = distance/stage_velocity
        time += t
        current_position = centre
        path.append(centre)
    return {
            "TotalTime" : time,
            "Path": path
        }
#in DP 
'''
def solve_M1(input_data):
    start_pos = input_data["InitialPosition"]
    start_angle = input_data["InitialAngle"] % 360
    v_stage = input_data["StageVelocity"]
    v_cam = input_data["CameraVelocity"]

    centers = [find_centre(d["Corners"]) for d in input_data["Dies"]]
    angles = [find_angle(d["Corners"]) for d in input_data["Dies"]]
    n = len(centers)

    @lru_cache(None)
    def dp(mask, last):
        if mask == (1 << n) - 1:
            return 0.0

        best = float("inf")
        for i in range(n):
            if not (mask & (1 << i)):
                if last == -1:
                    t_stage = find_distance(start_pos, centers[i]) / v_stage
                    t_cam = angle_diff(start_angle, angles[i]) / v_cam
                else:
                    t_stage = find_distance(centers[last], centers[i]) / v_stage
                    t_cam = angle_diff(angles[last], angles[i]) / v_cam

                step = t_stage
                best = min(best, step + dp(mask | (1 << i), i))
        return best

    mask = 0
    last = -1
    path = [start_pos]
    total_time = 0.0
    pos = start_pos[:]
    angle = start_angle

    while mask != (1 << n) - 1:
        best_i = None
        best_cost = float("inf")

        for i in range(n):
            if not (mask & (1 << i)):
                if last == -1:
                    t_stage = find_distance(start_pos, centers[i]) / v_stage
                    t_cam = angle_diff(start_angle, angles[i]) / v_cam
                else:
                    t_stage = find_distance(centers[last], centers[i]) / v_stage
                    t_cam = angle_diff(angles[last], angles[i]) / v_cam

                step = t_stage
                cost = step + dp(mask | (1 << i), i)

                if cost < best_cost:
                    best_cost = cost
                    best_i = i

        t_stage = find_distance(pos, centers[best_i]) / v_stage
        t_cam = angle_diff(angle, angles[best_i]) / v_cam
        step = t_stage

        total_time += step
        pos = centers[best_i]
        angle = angles[best_i]
        path.append(pos)

        mask |= (1 << best_i)
        last = best_i

    return {
        "TotalTime": total_time,
        "Path": path
    }'''


#IN ORDER NOT HIGH SCORE
'''def solve_M2(input_data):
    current_position = input_data["InitialPosition"]
    stage_velocity = input_data["StageVelocity"]
    current_angle = input_data["InitialAngle"]
    camera_velocity = input_data["CameraVelocity"]
    time = 0
    path = [current_position]
    for die in input_data["Dies"]:
        centre = find_centre(die["Corners"])
        target_die_angle = find_angle(die["Corners"])
        distance = find_distance(current_position,centre)
        t  = distance/stage_velocity
        delta_angle = abs(target_die_angle - current_angle)
        cam_time = delta_angle/camera_velocity
        time+=max(t,cam_time)
        path.append(centre)
        current_position = centre
        current_angle = target_die_angle
    return {
            "TotalTime" : time,
            "Path": path
        }'''




def find_angle(corners):
    max_len = -1
    best_angle = 0
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            best_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return best_angle % 360

def angle_diff(a, b):
    diff = abs(a - b) % 90
    return min(diff, 90 - diff)
#DP LOGIC TAKES TIME FOR LARGE INPUTS BUT GIVES 100% SCORE
from functools import lru_cache

def solve_M2(input_data):
    start_pos = input_data["InitialPosition"]
    start_angle = input_data["InitialAngle"] % 360
    v_stage = input_data["StageVelocity"]
    v_cam = input_data["CameraVelocity"]

    centers = [find_centre(d["Corners"]) for d in input_data["Dies"]]
    angles = [find_angle(d["Corners"]) for d in input_data["Dies"]]
    n = len(centers)
    

    @lru_cache(None)
    def dp(mask, last):
        if mask == (1 << n) - 1:
            return 0.0

        best = float("inf")
        for i in range(n):
            if not (mask & (1 << i)):
                if last == -1:
                    t_stage = find_distance(start_pos, centers[i]) / v_stage
                    t_cam = angle_diff(start_angle, angles[i]) / v_cam
                else:
                    t_stage = find_distance(centers[last], centers[i]) / v_stage
                    t_cam = angle_diff(angles[last], angles[i]) / v_cam

                step = max(t_stage, t_cam)
                best = min(best, step + dp(mask | (1 << i), i))
        return best
    
    mask = 0
    last = -1
    path = [start_pos]
    total_time = 0.0
    pos = start_pos[:]
    angle = start_angle

    while mask != (1 << n) - 1:
        best_i = None
        best_cost = float("inf")

        for i in range(n):
            if not (mask & (1 << i)):
                if last == -1:
                    t_stage = find_distance(start_pos, centers[i]) / v_stage
                    t_cam = angle_diff(start_angle, angles[i]) / v_cam
                else:
                    t_stage = find_distance(centers[last], centers[i]) / v_stage
                    t_cam = angle_diff(angles[last], angles[i]) / v_cam

                step = max(t_stage, t_cam)
                cost = step + dp(mask | (1 << i), i)

                if cost < best_cost:
                    best_cost = cost
                    best_i = i

        t_stage = find_distance(pos, centers[best_i]) / v_stage
        t_cam = angle_diff(angle, angles[best_i]) / v_cam
        step = max(t_stage, t_cam)

        total_time += step
        pos = centers[best_i]
        angle = angles[best_i]
        path.append(pos)

        mask |= (1 << best_i)
        last = best_i

    
    return {
        "TotalTime": total_time,
        "Path": path
    }

def angle_diff_T(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)
'''
def trapezoid_time(d, vmax, amax):
    if amax == 0:
        return d / vmax
    d_acc = (vmax ** 2) / (2 * amax)
    if d <= 2 * d_acc:
        return math.sqrt(2 * d / amax)
    return 2 * (vmax / amax) + (d - 2 * d_acc) / vmax

def solve_M3(input_data):
    pos = input_data["InitialPosition"]
    angle = input_data["InitialAngle"] % 360

    v_stage = input_data["StageVelocity"]
    a_stage = input_data.get("StageAcceleration", 0)
    v_cam = input_data["CameraVelocity"]
    a_cam = input_data.get("CameraAcceleration", 0)

    total = 0.0
    path = [pos]

    for die in input_data["Dies"]:
        center = find_centre(die["Corners"])
        target_angle = find_angle(die["Corners"])

        t_stage = trapezoid_time(find_distance(pos, center), v_stage, a_stage)
        t_cam = trapezoid_time(angle_diff_T(angle, target_angle), v_cam, a_cam)

        total += max(t_stage, t_cam)
        pos = center
        angle = target_angle
        path.append(pos)

    return {"TotalTime": round(total, 6), "Path": path}
'''
def angular_diff(a1, a2):
    diff = abs(a1 - a2) % 90
    return min(diff, 90 - diff)

def trap_time(d, vmax, amax):
    if d <= 0:
        return 0.0

    d_acc = (vmax * vmax) / (2 * amax)

    if d <= 2 * d_acc:
        return math.sqrt((2 * d) / amax)

    t_acc = vmax / amax
    d_cruise = d - 2 * d_acc
    t_cruise = d_cruise / vmax
    return 2 * t_acc + t_cruise

'''
def solve_M3(data):
    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    cam_vmax = data["CameraVelocity"]
    cam_amax = data["CameraAcceleration"]
    curr_pos = data["InitialPosition"][:]
    curr_angle = data["InitialAngle"]
    total_time = 0.0
    path = [curr_pos[:]]
    
    for die in data["Dies"]:
        wafer_radius = data["WaferDiameter"] / 2
        center = find_centre(die["Corners"])
        if find_distance([0,0], center) > wafer_radius:
            continue
        dist = find_distance(curr_pos, center)
        stage_time = trap_time(dist, vmax, amax)
        
        target_angle = find_angle(die["Corners"])
        angle_delta = angular_diff(curr_angle, target_angle)
        cam_time = trap_time(angle_delta, cam_vmax, cam_amax)
        
        step_time = max(stage_time, cam_time)
        total_time += step_time
        
        curr_pos = center[:]
        curr_angle = target_angle
        path.append(center[:])
    
    return {
        "TotalTime": round(total_time, 15),
        "Path": path
    }
'''
def solve_M3(data):
    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    cam_vmax = data["CameraVelocity"]
    cam_amax = data["CameraAcceleration"]
    start_pos = data["InitialPosition"][:]
    start_angle = data["InitialAngle"]
    wafer_radius = data["WaferDiameter"] / 2

    raw_dies = data["Dies"]
    dies = []
    centers = []
    angles = []

    for die in raw_dies:
        center = find_centre(die["Corners"])
        if find_distance([0, 0], center) <= wafer_radius:
            dies.append(die)
            centers.append(center)
            angles.append(find_angle(die["Corners"]))

    n = len(dies)
    if n == 0:
        return {
            "TotalTime": 0.0,
            "Path": [start_pos[:]]
        }

    stage_time = [[0.0] * n for _ in range(n)]
    cam_time   = [[0.0] * n for _ in range(n)]
    start_stage_time = [0.0] * n
    start_cam_time   = [0.0] * n

    for i in range(n):
        dist_start = find_distance(start_pos, centers[i])
        start_stage_time[i] = trap_time(dist_start, vmax, amax)

        ang_delta_start = angular_diff(start_angle, angles[i])
        start_cam_time[i] = trap_time(ang_delta_start, cam_vmax, cam_amax)

        for j in range(n):
            if i == j:
                continue
            d_ij = find_distance(centers[i], centers[j])
            stage_time[i][j] = trap_time(d_ij, vmax, amax)

            ang_delta_ij = angular_diff(angles[i], angles[j])
            cam_time[i][j] = trap_time(ang_delta_ij, cam_vmax, cam_amax)

    def step_time(prev_idx, next_idx, curr_angle=None):
        if prev_idx is None:
            t_stage = start_stage_time[next_idx]
            t_cam   = start_cam_time[next_idx]
        else:
            t_stage = stage_time[prev_idx][next_idx]
            t_cam   = cam_time[prev_idx][next_idx]
        return max(t_stage, t_cam)

    DP_THRESHOLD = 12

    if n <= DP_THRESHOLD:

        @lru_cache(None)
        def dp(mask, last):
            if mask == (1 << n) - 1:
                return 0.0
            best = float("inf")
            for i in range(n):
                if not (mask & (1 << i)):
                    prev_idx = None if last == -1 else last
                    st = step_time(prev_idx, i)
                    best = min(best, st + dp(mask | (1 << i), i))
            return best

        order = []
        mask, last = 0, -1
        while mask != (1 << n) - 1:
            best_i, best_cost = None, float("inf")
            for i in range(n):
                if not (mask & (1 << i)):
                    prev_idx = None if last == -1 else last
                    st = step_time(prev_idx, i)
                    cost = st + dp(mask | (1 << i), i)
                    if cost < best_cost:
                        best_cost, best_i = cost, i
            order.append(best_i)
            mask |= (1 << best_i)
            last = best_i

    else:
        unvisited = set(range(n))
        order = []
        prev = None
        while unvisited:
            best_i, best_c = None, float("inf")
            for i in unvisited:
                c = step_time(prev, i)
                if c < best_c:
                    best_c, best_i = c, i
            order.append(best_i)
            unvisited.remove(best_i)
            prev = best_i

        def total_time_order(ordr):
            tot = 0.0
            p = None
            for idx in ordr:
                tot += step_time(p, idx)
                p = idx
            return tot

        def delta_swap(ordr, i, k):
            n_ord = len(ordr)
            prev_left = ordr[i - 1] if i > 0 else None
            A = ordr[i]
            B = ordr[k]
            next_right = ordr[k + 1] if k + 1 < n_ord else None

            old = step_time(prev_left, A)
            if next_right is not None:
                old += step_time(B, next_right)

            new = step_time(prev_left, B)
            if next_right is not None:
                new += step_time(A, next_right)

            return new - old

        improved = True
        max_iters = 5
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            for i in range(0, len(order) - 2):
                for k in range(i + 1, len(order) - 1):
                    if delta_swap(order, i, k) < -1e-9:
                        order[i:k+1] = reversed(order[i:k+1])
                        improved = True

    total_time = 0.0
    curr_idx = None
    curr_pos = start_pos[:]
    path = [curr_pos[:]]

    for idx in order:
        st = step_time(curr_idx, idx)
        total_time += st
        curr_idx = idx
        curr_pos = centers[idx][:]
        path.append(curr_pos)

    return {
        "TotalTime": round(total_time, 15),
        "Path": path
    }

def main():
    while True:
        print("ENTER THE MILESTONE NUMBER: ")
        n = int(input())
        if n==1:
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone1\Input_Milestone1_Testcase3.json","r") as f:
                input_data = json.load(f)
            output_M1 = solve_M1(input_data)
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone1\TestCase_1_3.json", "w") as f:
                json.dump(output_M1, f, indent=2)
            print(output_M1)
        elif n == 2:
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone2\Input_Milestone2_Testcase4.json","r") as f:
                input_data = json.load(f)
            output_M2 = solve_M2(input_data)
            
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone2\TestCase_2_4.json", "w") as f:
                json.dump(output_M2, f, indent=2)
                
            print(output_M2)
        elif n == 3:
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone3\Input_Milestone3_Testcase2.json","r") as f:
                input_data = json.load(f)
            output_M3 = solve_M3(input_data)
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone3\TestCase_3_2.json", "w") as f:
                json.dump(output_M3, f, indent=2)
            print(output_M3)
        elif n == 0:
            break
        else:
            "ENTER 1 OR 2 OR 3"
            
if __name__ == "__main__":
    main()

