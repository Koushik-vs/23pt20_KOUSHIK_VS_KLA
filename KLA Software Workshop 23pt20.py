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

def trapezoid_time(distance, vmax, amax):
    if distance <= 0:
        return 0.0
    if vmax <= 0:
        raise ValueError("Velocity must be positive")
    if amax < 0:
        raise ValueError("Acceleration must be non-negative")

    if amax == 0:
        return distance / vmax

    d_acc = 0.5 * (vmax ** 2) / amax
    if distance <= 2 * d_acc:
        return math.sqrt(2 * distance / amax)
    t_acc = vmax / amax
    d_cruise = distance - 2 * d_acc
    return 2 * t_acc + d_cruise / vmax

def build_time_matrices(centers, angles, start_pos, start_angle, v_stage, a_stage, v_cam, a_cam):
    n = len(centers)
    stage_time = [[0.0] * n for _ in range(n)]
    cam_time = [[0.0] * n for _ in range(n)]
    start_stage = [0.0] * n
    start_cam = [0.0] * n

    for i in range(n):
        start_stage[i] = trapezoid_time(find_distance(start_pos, centers[i]), v_stage, a_stage)
        start_cam[i] = trapezoid_time(angle_diff_T(start_angle, angles[i]), v_cam, a_cam)
        for j in range(n):
            if i == j:
                continue
            stage_time[i][j] = trapezoid_time(find_distance(centers[i], centers[j]), v_stage, a_stage)
            cam_time[i][j] = trapezoid_time(angle_diff_T(angles[i], angles[j]), v_cam, a_cam)

    return stage_time, cam_time, start_stage, start_cam

def step_cost(prev_idx, next_idx, stage_time, cam_time, start_stage, start_cam):
    if prev_idx is None:
        return max(start_stage[next_idx], start_cam[next_idx])
    return max(stage_time[prev_idx][next_idx], cam_time[prev_idx][next_idx])

def path_time(order, stage_time, cam_time, start_stage, start_cam):
    total = 0.0
    prev = None
    for idx in order:
        total += step_cost(prev, idx, stage_time, cam_time, start_stage, start_cam)
        prev = idx
    return total


def solve_M3(input_data):
    start_pos = input_data["InitialPosition"]
    start_angle = input_data["InitialAngle"] % 360
    v_stage = input_data["StageVelocity"]
    a_stage = input_data.get("StageAcceleration", 0)
    v_cam = input_data["CameraVelocity"]
    a_cam = input_data.get("CameraAcceleration", 0)

    dies = input_data["Dies"]
    centers = [find_centre(d["Corners"]) for d in dies]
    angles = [find_angle(d["Corners"]) for d in dies]
    n = len(dies)

    stage_time, cam_time, start_stage, start_cam = build_time_matrices(
        centers, angles, start_pos, start_angle, v_stage, a_stage, v_cam, a_cam
    )

    @lru_cache(None)
    def dp(mask, last):
        if mask == (1 << n) - 1:
            return 0.0
        best = float("inf")
        for i in range(n):
            if not (mask & (1 << i)):
                step = step_cost(last if last != -1 else None, i, stage_time, cam_time, start_stage, start_cam)
                best = min(best, step + dp(mask | (1 << i), i))
        return best

    order = []
    mask, last = 0, -1
    while mask != (1 << n) - 1:
        best_i, best_cost = None, float("inf")
        for i in range(n):
            if not (mask & (1 << i)):
                step = step_cost(last if last != -1 else None, i, stage_time, cam_time, start_stage, start_cam)
                cost = step + dp(mask | (1 << i), i)
                if cost < best_cost:
                    best_cost, best_i = cost, i
        order.append(best_i)
        mask |= (1 << best_i)
        last = best_i

    total_time = path_time(order, stage_time, cam_time, start_stage, start_cam)
    path = [start_pos[:]] + [centers[i][:] for i in order]

    return {
        "TotalTime": round(total_time, 6),
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
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone3\Input_Milestone3_Testcase1.json","r") as f:
                input_data = json.load(f)
            output_M3 = solve_M3(input_data)
            with open(r"Software_Workshop_Day1\Day1\TestCases\Milestone3\TestCase_3_1.json", "w") as f:
                json.dump(output_M3, f, indent=2)
            print(output_M3)
        elif n == 0:
            break
        else:
            "ENTER 1 OR 2 OR 3"
            
if __name__ == "__main__":
    main()


