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
    d_acc = (vmax ** 2) / (2 * amax)  # distance to reach vmax
    if d <= 2 * d_acc:  # triangular profile
        return math.sqrt(2 * d / amax)
    else:  # trapezoidal
        return 2 * (vmax / amax) + (d - 2 * d_acc) / vmax

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

import heapq
def line_intersects_rect(p1, p2, bl, tr):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return False
    t_near = float('-inf')
    t_far = float('inf')
    for i, dir_, minb, maxb in [(0, dx, bl[0], tr[0]), (1, dy, bl[1], tr[1])]:
        if dir_ == 0:
            if p1[i] < minb or p1[i] > maxb:#check if the line is outside the bl and tr
                return False
        else:#here it is computed,if the infinite lie crosses the 2 parallel edges of the rectan
            t1 = (minb - p1[i]) / dir_
            t2 = (maxb - p1[i]) / dir_
            t_near = max(t_near, min(t1, t2))
            t_far = min(t_far, max(t1, t2))
    if t_near > t_far:
        return False
    if t_far < 0 or t_near > 1:
        return False
    return t_near < t_far

def intersect_forbidden(p1, p2, zones, eps=1e-6):
    if find_distance(p1, p2) < eps:
        return False
    for zone in zones:
        bl = zone["BottomLeft"]
        tr = zone["TopRight"]
        bl_eps = [bl[0] + eps, bl[1] + eps]
        tr_eps = [tr[0] - eps, tr[1] - eps]
        if line_intersects_rect(p1, p2, bl_eps, tr_eps):
            return True
    return False

def dijkstra(adj, src, n):
    dist = [float('inf')] * n
    dist[src] = 0
    prev = [-1] * n
    pq = [(0, src)]
    while pq:
        dd, u = heapq.heappop(pq)
        if dd > dist[u]: continue
        for v, w in adj[u]:
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    return dist, prev

def get_path(src, tgt, prev):
    path = []
    cur = tgt
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path if path and path[0] == src else []


def inside_wafer(p, R):
    return p[0]*p[0] + p[1]*p[1] <= R*R + 1e-9

def ang_diff(a1, a2, sym):
    diff = abs(a1 - a2) % sym
    return min(diff, sym - diff)

def angular_time(a, wmax, alphamax):
    if a <= 0:
        return 0.0
    a_switch = (wmax * wmax) / alphamax
    if a >= a_switch:
        return (2 * wmax / alphamax) + (a - a_switch) / wmax
    return 2 * math.sqrt(a / alphamax)

def move_cost(idx1, a1, idx2, a2, dists, wmax, alphamax, sym):
    return max(
        dists[idx1][idx2],
        angular_time(ang_diff(a1, a2, sym), wmax, alphamax)
    )

def total_time(path, dists, wmax, alphamax, sym):
    return sum(
        move_cost(path[i][0], path[i][1], path[i+1][0], path[i+1][1],
                  dists, wmax, alphamax, sym)
        for i in range(len(path) - 1)
    )


def solve_M4(data):
    start_pos = data["InitialPosition"][:]
    start_angle = data["InitialAngle"]

    vmax = data["StageVelocity"]
    amax = data["StageAcceleration"]
    cam_v = data["CameraVelocity"]
    cam_a = data["CameraAcceleration"]

    wafer_radius = data["WaferDiameter"] / 2

    zones = data["ForbiddenZones"]
    dies = data["Dies"]

    centers = [find_centre(d["Corners"]) for d in dies]
    angles  = [find_angle(d["Corners"]) for d in dies]

    n = len(dies)
    unvisited = set(range(n))

    total_time_acc = 0.0
    full_path = [start_pos[:]]

    cur_pos = start_pos[:]
    cur_angle = start_angle

    while unvisited:
        best_i = None
        best_cost = float("inf")
        best_geom_path = None

        for i in unvisited:
            tgt = centers[i]

            nodes = [cur_pos, tgt]
            for z in zones:
                bl = z["BottomLeft"]
                tr = z["TopRight"]
                nodes.extend([
                    [bl[0], bl[1]], [bl[0], tr[1]],
                    [tr[0], bl[1]], [tr[0], tr[1]],
                ])

            uniq = []
            seen = set()
            for p in nodes:
                key = (round(p[0], 6), round(p[1], 6))
                if key in seen:
                    continue
                seen.add(key)
                if inside_wafer(p, wafer_radius):
                    uniq.append(p)

            nodes = uniq
            idx = {tuple(p): k for k, p in enumerate(nodes)}

            adj = [[] for _ in range(len(nodes))]
            for i1 in range(len(nodes)):
                for i2 in range(i1 + 1, len(nodes)):
                    p1 = nodes[i1]
                    p2 = nodes[i2]
                    if intersect_forbidden(p1, p2, zones):
                        continue
                    d = find_distance(p1, p2)
                    adj[i1].append((i2, d))
                    adj[i2].append((i1, d))

            src   = idx[tuple(cur_pos)]
            tgt_i = idx[tuple(tgt)]

            dist, prev = dijkstra(adj, src, len(nodes))
            path_idx = get_path(src, tgt_i, prev)
            if not path_idx:
                continue

            geom_path = [nodes[k] for k in path_idx]

            total_stage_dist = 0.0
            for k in range(len(geom_path) - 1):
                total_stage_dist += find_distance(geom_path[k], geom_path[k+1])
            stage_time = trap_time(total_stage_dist, vmax, amax)
            stage_time = trap_time(total_stage_dist, vmax, amax)
            cam_time = trap_time(angle_diff(cur_angle, angles[i]), cam_v, cam_a)
            step_cost = max(stage_time, cam_time)

            if step_cost < best_cost:
                best_cost = step_cost
                best_i = i
                best_geom_path = geom_path

        if best_i is None:
            break

        for p in best_geom_path[1:]:
            full_path.append(p)

        total_time_acc += best_cost
        cur_pos = centers[best_i][:]
        cur_angle = angles[best_i]

        unvisited.remove(best_i)

    return {
        "TotalTime": round(total_time_acc, 6),
        "Path": full_path
    }



def dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def center(corners):
    return (
        sum(p[0] for p in corners) / 4.0,
        sum(p[1] for p in corners) / 4.0
    )

def base_angle(corners):
    # Take edge from corners[0] to corners[1]
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    return math.degrees(math.atan2(dy, dx)) % 360.0



def linear_time(d, vmax, amax):
    """
    Trapezoidal motion time for linear distance d.
    """
    if d <= 0:
        return 0.0
    d_switch = (vmax * vmax) / amax
    if d >= d_switch:
        return (2.0 * vmax / amax) + (d - d_switch) / vmax
    return 2.0 * math.sqrt(d / amax)


def move_cost_idx(idx1, a1, idx2, a2, dists, wmax, alphamax, sym=90.0):
    """
    Cost of moving from node idx1 (angle a1) to idx2 (angle a2),
    given stage times in dists and camera limits.
    """
    stage_t = dists[idx1][idx2]
    cam_t   = angular_time(ang_diff(a1, a2, sym), wmax, alphamax)
    return max(stage_t, cam_t)

def total_time_m4(path, dists, wmax, alphamax, sym=90.0):
    """
    Total time for a sequence of (idx, angle) states.
    """
    return sum(
        move_cost_idx(path[i][0], path[i][1],
                      path[i+1][0], path[i+1][1],
                      dists, wmax, alphamax, sym)
        for i in range(len(path) - 1)
    )



def does_intersect_forbidden(p1, p2, zones, eps=1e-6):
    """
    True if segment p1–p2 passes through the interior of any forbidden zone.
    Corners/edges are avoided via epsilon shrink.
    """
    if dist(p1, p2) < eps:
        return False
    for zone in zones:
        bl = zone["BottomLeft"]
        tr = zone["TopRight"]
        bl_eps = [bl[0] + eps, bl[1] + eps]
        tr_eps = [tr[0] - eps, tr[1] - eps]
        if line_intersects_rect(p1, p2, bl_eps, tr_eps):
            return True
    return False


def find_detour(p1, p2, zones, offset=5.0, max_attempts=10):
    """
    If segment p1–p2 grazes/crosses forbidden, try perpendicular detour near mid.
    """
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return None

    perp_dx, perp_dy = -dy / length, dx / length

    for i in range(1, max_attempts + 1):
        for sign in (1, -1):
            detour = (mx + sign * offset * i * perp_dx,
                      my + sign * offset * i * perp_dy)
            if (not does_intersect_forbidden(p1, detour, zones) and
                not does_intersect_forbidden(detour, p2, zones)):
                return detour

    return None

# ---------- Ordering & optimization for M4 ----------

def greedy_nn_m4(start_idx, start_ang, dies, sym, dists, wmax, alphamax, node_list, pos_to_idx, angle_map):
    """
    Greedy nearest-neighbour on die indices, using dists[][] and camera timing.
    path is list of (idx, angle).
    """
    used = [False] * len(dies)
    current_idx = start_idx
    current_ang = start_ang
    path = [(current_idx, current_ang)]

    for _ in range(len(dies)):
        best_die_idx = -1
        best_ang = 0.0
        best_cost = float("inf")

        for ii, (p, base) in enumerate(dies):
            if used[ii]:
                continue
            idx = pos_to_idx[p]

            # 90° symmetry: 4 possible orientations
            angles = [(base + k * 90.0) % 360.0 for k in range(4 if sym else 1)]

            for a in angles:
                c = move_cost_idx(current_idx, current_ang, idx, a,
                                  dists, wmax, alphamax, sym=90.0)
                if c < best_cost:
                    best_cost = c
                    best_die_idx = ii
                    best_ang = a
                    best_idx = idx

        if best_die_idx == -1:
            break

        used[best_die_idx] = True
        current_idx = best_idx
        current_ang = best_ang
        path.append((current_idx, current_ang))

    return path

def two_opt_m4(path, dists, wmax, alphamax, sym=90.0):
    """
    2-opt improvement on ordering (path is (idx, angle)), angles re-optimized later.
    """
    best = path
    best_t = total_time_m4(best, dists, wmax, alphamax, sym)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                t = total_time_m4(cand, dists, wmax, alphamax, sym)
                if t < best_t - 1e-9:
                    best, best_t = cand, t
                    improved = True
    return best

def optimize_angles_m4(path, angle_map, sym, node_list):
    """
    Re-assign best angles for each node in path, given base angles & symmetry.
    """
    res = [path[0]]
    for i in range(1, len(path)):
        idx = path[i][0]
        base = angle_map[node_list[idx]]
        prev_ang = res[-1][1]

        angles = [(base + k * 90.0) % 360.0 for k in range(4 if sym else 1)]
        best_ang = min(angles, key=lambda a: ang_diff(prev_ang, a, sym))
        res.append((idx, best_ang))
    return res

# ---------- Final Milestone 4 solver ----------

def solve_M4(data):
    sym = True  # 90-degree symmetry

    vmax     = data["StageVelocity"]
    amax     = data["StageAcceleration"]
    wmax     = data["CameraVelocity"]
    alphamax = data["CameraAcceleration"]

    start_pos = tuple(data["InitialPosition"])
    start_ang = data["InitialAngle"]
    wafer_diameter  = data.get("WaferDiameter", None)
    forbidden_zones = data.get("ForbiddenZones", [])
    dies_raw        = data["Dies"]

    # --- Build die centers and base angles ---
    dies_centers = []
    angle_map_pos = {}
    for d in dies_raw:
        c = center(d["Corners"])
        a = base_angle(d["Corners"])
        dies_centers.append(c)
        angle_map_pos[c] = a

    # Filter dies inside wafer (if wafer_diameter is given)
    def inside_wafer(c):
        if wafer_diameter is None:
            return True
        R = wafer_diameter / 2.0
        return (c[0] * c[0] + c[1] * c[1]) <= R * R + 1e-9

    dies = []
    for c in dies_centers:
        if inside_wafer(c):
            dies.append((c, angle_map_pos[c]))

    if not dies:
        return {"TotalTime": 0.0, "Path": [list(start_pos)]}

    # --- Build node list: start + all die centers + all forbidden corners ---
    points = [start_pos] + [c for (c, _) in dies]

    corners = []
    for zone in forbidden_zones:
        bl = tuple(zone["BottomLeft"])
        tr = tuple(zone["TopRight"])
        tl = (bl[0], tr[1])
        br = (tr[0], bl[1])
        corners.extend([bl, tl, tr, br])

    all_nodes_set = set(points + corners)
    node_list = list(all_nodes_set)
    n_nodes = len(node_list)

    pos_to_idx = {pos: idx for idx, pos in enumerate(node_list)}
    start_idx  = pos_to_idx[start_pos]

    # Map position -> base angle (only for die centers)
    angle_map = {}
    for c, base in dies:
        angle_map[c] = base
    # Start node: keep its actual initial angle (but we don't store in angle_map)

    # --- Build visibility graph for stage, edges weighted by linear_time ---
    adj = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p1 = node_list[i]
            p2 = node_list[j]
            if not does_intersect_forbidden(p1, p2, forbidden_zones):
                d = dist(p1, p2)
                t = linear_time(d, vmax, amax)
                adj[i].append((j, t))
                adj[j].append((i, t))

    # --- All-pairs (or at least from each die/start) shortest stage times ---
    # Build dists matrix and prev list for path reconstruction.
    dists = [[float('inf')] * n_nodes for _ in range(n_nodes)]
    prevs = []
    for src in range(n_nodes):
        dist_src, prev_src = dijkstra(adj, src)
        prevs.append(prev_src)
        for j in range(n_nodes):
            dists[src][j] = dist_src[j]

    # --- Prepare dies in terms of indices ---
    dies_as_nodes = []
    for c, base in dies:
        idx = pos_to_idx[c]
        dies_as_nodes.append((c, base))  # keep (pos, base); index via pos_to_idx when needed

    # --- Greedy nearest-neighbour order in node space ---
    path_idx_ang = greedy_nn_m4(
        start_idx, start_ang,
        dies_as_nodes, sym, dists, wmax, alphamax,
        node_list, pos_to_idx, angle_map
    )

    # 2-opt on node order
    path_idx_ang = two_opt_m4(path_idx_ang, dists, wmax, alphamax, sym=90.0)

    # Re-assign best angles with symmetry
    path_idx_ang = optimize_angles_m4(path_idx_ang, angle_map, sym=90.0, node_list=node_list)

    # --- Reconstruct full geometric path via prevs (node indices) ---
    full_idx_path = []
    for k in range(len(path_idx_ang) - 1):
        a_idx = path_idx_ang[k][0]
        b_idx = path_idx_ang[k+1][0]
        sub_path = get_path(a_idx, b_idx, prevs[a_idx])
        if not sub_path:
            continue
        if k == 0:
            full_idx_path = sub_path
        else:
            full_idx_path += sub_path[1:]

    full_pos_path = [list(node_list[idx]) for idx in full_idx_path]

    # --- Optional: inject small detours if any segment still grazes forbidden ---
    adjusted_path = []
    if full_pos_path:
        for i in range(len(full_pos_path) - 1):
            p1, p2 = full_pos_path[i], full_pos_path[i+1]
            if does_intersect_forbidden(p1, p2, forbidden_zones):
                detour = find_detour(p1, p2, forbidden_zones)
                if detour:
                    adjusted_path.append(p1)
                    adjusted_path.append(list(detour))
                else:
                    adjusted_path.append(p1)
            else:
                adjusted_path.append(p1)
        adjusted_path.append(full_pos_path[-1])
    else:
        adjusted_path = [list(start_pos)]

    # --- Final total time using node path (idx, angle) ---
    total_T = total_time_m4(path_idx_ang, dists, wmax, alphamax, sym=90.0)

    return {
        "TotalTime": round(total_T, 6),
        "Path": adjusted_path
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
