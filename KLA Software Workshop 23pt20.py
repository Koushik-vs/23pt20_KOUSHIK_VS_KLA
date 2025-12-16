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
'''
def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def segment_intersects_rect(p1, p2, rect):
    x1, y1 = p1
    x2, y2 = p2
    rx1, ry1, rx2, ry2 = rect

    if x1 == x2: 
        if rx1 < x1 < rx2:
            return not (max(y1, y2) <= ry1 or min(y1, y2) >= ry2)

    if y1 == y2: 
        if ry1 < y1 < ry2:
            return not (max(x1, x2) <= rx1 or min(x1, x2) >= rx2)

    return False

def path_clear(path, forbidden):
    for i in range(len(path) - 1):
        for rect in forbidden:
            if segment_intersects_rect(path[i], path[i + 1], rect):
                return False
    return True

def manhattan_paths(A, B):
    x0, y0 = A
    x1, y1 = B
    return [
        [A, [x1, y0], B],
        [A, [x0, y1], B]
    ]

def detour_paths(A, B, rect, eps=1e-6):
    rx1, ry1, rx2, ry2 = rect
    x0, y0 = A
    x1, y1 = B

    paths = []
    xs = [rx1 - eps, rx2 + eps]
    ys = [ry1 - eps, ry2 + eps]

    for x in xs:
        paths.append([A, [x, y0], [x, y1], B])
    for y in ys:
        paths.append([A, [x0, y], [x1, y], B])

    return paths

def path_length(path):
    return sum(manhattan_dist(path[i], path[i + 1]) for i in range(len(path) - 1))

def find_best_path(A, B, forbidden):
    candidates = []

    for p in manhattan_paths(A, B):
        if path_clear(p, forbidden):
            candidates.append(p)

    if candidates:
        return min(candidates, key=path_length)

    for rect in forbidden:
        for p in detour_paths(A, B, rect):
            if path_clear(p, forbidden):
                candidates.append(p)

    if not candidates:
        raise RuntimeError("No valid path found")

    return min(candidates, key=path_length)


def solve_M4(data):
    pos = data["InitialPosition"][:]
    angle = data["InitialAngle"]

    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    cam_v = data["CameraVelocity"]
    cam_a = data["CameraAcceleration"]

    forbidden = [
        (
            z["BottomLeft"][0],
            z["BottomLeft"][1],
            z["TopRight"][0],
            z["TopRight"][1]
        )
        for z in data["ForbiddenZones"]
    ]

    total_time = 0.0
    full_path = [pos[:]]

    for die in data["Dies"]:
        target = find_centre(die["Corners"])
        target_angle = find_angle(die["Corners"])

        polyline = find_best_path(pos, target, forbidden)

        for i in range(1, len(polyline)):
            d = manhattan_dist(polyline[i - 1], polyline[i])
            t_stage = trap_time(d, vmax, amax)
            t_cam = trap_time(angle_diff(angle, target_angle), cam_v, cam_a)
            total_time += max(t_stage, t_cam)
            full_path.append(polyline[i])

        pos = target[:]
        angle = target_angle

    return {
        "TotalTime": round(total_time, 6),
        "Path": full_path
    }
'''
import heapq

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def inside_wafer(p, R):
    return p[0]*p[0] + p[1]*p[1] <= R*R + 1e-9

def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

'''def point_on_segment(P, A, B, eps=1e-9):
    x, y = P
    x1, y1 = A
    x2, y2 = B

    # Collinear check
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False

    # Within bounding box
    if min(x1, x2) - eps <= x <= max(x1, x2) + eps and min(y1, y2) - eps <= y <= max(y1, y2) + eps:
        return True

    return False
'''
def point_on_segment(p, a, b, eps=1e-9):
    # Check if p lies on segment a-b
    cross = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if abs(cross) > eps:
        return False

    dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    if dot < 0:
        return False

    sq_len = (b[0] - a[0])**2 + (b[1] - a[1])**2
    if dot > sq_len:
        return False

    return True
def get_forbidden_corners(forbidden):
    corners = []
    for x1, y1, x2, y2 in forbidden:
        corners.extend([
            [x1, y1],
            [x1, y2],
            [x2, y1],
            [x2, y2]
        ])
    return corners
def insert_corner_points(path, forbidden, eps=1e-9):
    refined = []
    corners = get_forbidden_corners(forbidden)

    def same(p, q):
        return abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) < eps

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]

        # collect corners lying strictly inside segment a-b
        on_seg = []
        for c in corners:
            if point_on_segment(c, a, b):
                on_seg.append(c)

        # sort by distance from a
        on_seg.sort(key=lambda p: (p[0] - a[0])**2 + (p[1] - a[1])**2)

        if not refined:
            refined.append(a)

        for p in on_seg:
            if not same(refined[-1], p):
                refined.append(p)

        if not same(refined[-1], b):
            refined.append(b)

    return refined
def segment_intersects_rect(p1, p2, rect):
    x1, y1, x2, y2 = rect

    # Rectangle corners
    corners = [
        [x1, y1], [x1, y2],
        [x2, y1], [x2, y2]
    ]

    # ✅ If segment touches ONLY corners → allow
    for c in corners:
        if point_on_segment(c, p1, p2):
            # touching a corner is allowed
            pass

    # ✅ If either endpoint is strictly inside → block
    if (x1 < p1[0] < x2 and y1 < p1[1] < y2) or (x1 < p2[0] < x2 and y1 < p2[1] < y2):
        return True

    # Rectangle edges
    edges = [
        ([x1, y1], [x2, y1]),
        ([x2, y1], [x2, y2]),
        ([x2, y2], [x1, y2]),
        ([x1, y2], [x1, y1]),
    ]

    # ✅ If segment lies exactly on an edge → allow
    for e1, e2 in edges:
        if point_on_segment(p1, e1, e2) and point_on_segment(p2, e1, e2):
            return False

    # ✅ If segment intersects edge at a non‑corner point → block
    for e1, e2 in edges:
        if segments_intersect(p1, p2, e1, e2):
            # Check if intersection is a corner
            for c in corners:
                if point_on_segment(c, p1, p2) and point_on_segment(c, e1, e2):
                    # touching corner → allowed
                    break
            else:
                return True  # real intersection

    return False


def segment_valid(p1, p2, forbidden, wafer_radius):
    if not inside_wafer(p1, wafer_radius) or not inside_wafer(p2, wafer_radius):
        return False
    for rect in forbidden:
        if segment_intersects_rect(p1, p2, rect):
            return False
    return True

def inside_forbidden(p, forbidden):
    x, y = p
    for rx1, ry1, rx2, ry2 in forbidden:
        if rx1 < x < rx2 and ry1 < y < ry2:
            return True
    return False

import random

def shortest_path(start, goal, forbidden, wafer_radius):
    nodes = [start, goal]

    # 1) Rectangle corners
    for r in forbidden:
        x1, y1, x2, y2 = r
        corners = [
            [x1, y1],
            [x1, y2],
            [x2, y1],
            [x2, y2]
        ]
        nodes.extend(corners)

    # 2) Rectangle edge midpoints (still on edges, not inside)
    for r in forbidden:
        x1, y1, x2, y2 = r
        candidates = [
            [(x1 + x2) / 2.0, y1],
            [(x1 + x2) / 2.0, y2],
            [x1, (y1 + y2) / 2.0],
            [x2, (y1 + y2) / 2.0],
        ]
        nodes.extend(candidates)

    # 3) Free-space random samples inside wafer, outside forbidden
    N_SAMPLES = 200          # you can tune this (100–400)
    for _ in range(N_SAMPLES):
        # Sample in bounding box of wafer
        R = wafer_radius
        x = random.uniform(-R, R)
        y = random.uniform(-R, R)
        p = [x, y]
        if not inside_wafer(p, wafer_radius):
            continue
        if inside_forbidden(p, forbidden):  # strict interior check
            continue
        nodes.append(p)

    # Deduplicate and filter again
    unique_nodes = []
    seen = set()
    for p in nodes:
        tp = (round(p[0], 6), round(p[1], 6))
        if tp in seen:
            continue
        seen.add(tp)
        if not inside_wafer(p, wafer_radius):
            continue
        if inside_forbidden(p, forbidden):
            continue
        unique_nodes.append(p)

    nodes = unique_nodes

    # Build visibility graph: edges between any pair with a valid segment
    adj = {tuple(p): [] for p in nodes}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            p = nodes[i]
            q = nodes[j]
            if segment_valid(p, q, forbidden, wafer_radius):
                d = euclid(p, q)
                adj[tuple(p)].append((tuple(q), d))
                adj[tuple(q)].append((tuple(p), d))

    start_t = tuple(start)
    goal_t  = tuple(goal)

    # Dijkstra
    pq = [(0.0, start_t, None)]
    dist = {start_t: 0.0}
    parent = {}

    while pq:
        cd, u, par = heapq.heappop(pq)
        if u in parent:
            continue
        parent[u] = par
        if u == goal_t:
            break
        for v, w in adj[u]:
            nd = cd + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v, u))

    # If no path found in graph, fall back to direct segment (or handle as unreachable)
    if goal_t not in parent:
        return [start, goal]

    # Reconstruct path
    path = []
    cur = goal_t
    while cur is not None:
        path.append(list(cur))
        cur = parent.get(cur)
    path.reverse()
    return path

def solve_M4(data):
    pos = data["InitialPosition"][:]
    angle = data["InitialAngle"]

    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    cam_v = data["CameraVelocity"]
    cam_a = data["CameraAcceleration"]

    wafer_radius = data["WaferDiameter"] / 2

    forbidden = [
        (
            z["BottomLeft"][0],
            z["BottomLeft"][1],
            z["TopRight"][0],
            z["TopRight"][1]
        )
        for z in data["ForbiddenZones"]
    ]

    dies = data["Dies"]
    centers = [find_centre(d["Corners"]) for d in dies]
    angles  = [find_angle(d["Corners"]) for d in dies]

    n = len(dies)
    unvisited = set(range(n))

    total_time = 0.0
    full_path = [pos[:]]

    while unvisited:
        best_i = None
        best_cost = float("inf")
        best_polyline = None

       
        for i in unvisited:
            target = centers[i]

            polyline = shortest_path(pos, target, forbidden, wafer_radius)
            polyline = insert_corner_points(polyline, forbidden)

            if len(polyline) == 2 and not segment_valid(pos, target, forbidden, wafer_radius):
                continue  

            stage_time = 0.0
            for k in range(1, len(polyline)):
                d = euclid(polyline[k-1], polyline[k])
                stage_time = max(stage_time,trap_time(d, vmax, amax))

            cam_time = trap_time(angle_diff(angle, angles[i]), cam_v, cam_a)

            step_cost = max(stage_time, cam_time)

            if step_cost < best_cost:
                best_cost = step_cost
                best_i = i
                best_polyline = polyline

        if best_i is None:
            break

        for k in range(1, len(best_polyline)):
            full_path.append(best_polyline[k])

        total_time += best_cost
        pos = centers[best_i][:]
        angle = angles[best_i]

        unvisited.remove(best_i)

    return {
        "TotalTime": round(total_time, 6),
        "Path": full_path
    }

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

def plot_dies(input_data, title="Wafer Dies Plot"):
    dies = input_data["Dies"]
    wafer_radius = input_data["WaferDiameter"] / 2
    initial_pos = input_data["InitialPosition"]

    fig, ax = plt.subplots(figsize=(10,10))

    wafer = Circle((0,0), wafer_radius, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(wafer)

    for die in dies:
        corners = die["Corners"]
        poly = Polygon(corners, closed=True, fill=True, alpha=0.3, edgecolor='blue')
        ax.add_patch(poly)

        cx = sum(c[0] for c in corners) / 4
        cy = sum(c[1] for c in corners) / 4
        ax.plot(cx, cy, 'ro', markersize=4)

    if "ForbiddenZones" in input_data:
        for zone in input_data["ForbiddenZones"]:
            lb = zone["BottomLeft"]   
            tr = zone["TopRight"]     

            width  = tr[0] - lb[0]
            height = tr[1] - lb[1]

            rect = Rectangle(
                (lb[0], lb[1]),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='red',
                alpha=0.3,
                label="Forbidden Zone"
            )
            ax.add_patch(rect)

    ax.plot(initial_pos[0], initial_pos[1], 'kx', markersize=10, label="Initial Position")

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.show()



def main():
    while True:
        print("\nENTER MILESTONE NUMBER (1–4) OR 0 TO EXIT:")
        milestone = int(input())

        if milestone == 0:
            break

        print("ENTER TESTCASE NUMBER:")
        testcase = int(input())

        input_path  = f"Software_Workshop_Day1\Day1\TestCases\Milestone{milestone}\Input_Milestone{milestone}_Testcase{testcase}.json"
        output_path = f"Software_Workshop_Day1\Day1\TestCases\Milestone{milestone}\TestCase_{milestone}_{testcase}.json"

        with open(input_path, "r") as f:
            input_data = json.load(f)
        plot_dies(input_data, title=f"Milestone {milestone} - Testcase {testcase}")

        if milestone == 1:
            result = solve_M1(input_data)
        elif milestone == 2:
            result = solve_M2(input_data)
        elif milestone == 3:
            result = solve_M3(input_data)
        elif milestone == 4:
            result = solve_M4(input_data)
        else:
            print("Invalid milestone")
            continue

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ Output saved to {output_path}")
        print(result)



if __name__ == "__main__":
    main()
