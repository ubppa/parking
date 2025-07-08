import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import random
import math
import time
from heapq import nsmallest
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Patch
import numpy as np

# ======================================================
# GA シミュレーション用パラメータ・関数
# ======================================================

# 車両サイズ（普通乗用車の比率 60×24）
CAR_LENGTH = 60.0
CAR_WIDTH  = 24.0

# シミュレーション領域（キャンバスサイズ＝ウィンドウサイズ）
ENV_WIDTH = 800
ENV_HEIGHT = 600

# GA の各パラメータ
POPULATION_SIZE = 40
GENERATIONS = 10000
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.3
ADDITION_RATE = 0.3
DELETION_RATE = 0.3
MAX_GENES = 150
ELITE_COUNT = 5

# 動作パターン（前進・後退は短・中・長、旋回は複数角度）
MOVEMENTS_LIST = [
    (20.0, 0.0, 0.0),   # 短距離前進
    (27.0, 0.0, 0.0),   # 中距離前進
    (35.0, 0.0, 0.0),   # 長距離前進
    (27.0, 5.0, 15.0),     # 右旋回：軽
    (28.2, 8.9, 30.5),     # 右旋回：中
    (30.0, 12.0, 45.0),    # 右旋回：強
    (32.0, 15.0, 60.0),    # 右旋回：非常に強
    (27.0, -5.0, -15.0),   # 左旋回：軽
    (28.7, -8.9, -30.5),   # 左旋回：中
    (30.0, -12.0, -45.0),  # 左旋回：強
    (32.0, -15.0, -60.0),  # 左旋回：非常に強
    (-20.0, 0.0, 0.0),   # 短距離後退
    (-27.0, 0.0, 0.0),   # 中距離後退
    (-35.0, 0.0, 0.0)    # 長距離後退
]

# --- 状態の扱い ---
# スタート／ゴール状態は (x, y, theta, gear) の4要素タプルとして保持します。
# gear が "backward" の場合、描画・シミュレーションでは theta に 180° を加えたものとして扱います。
def effective_state(state):
    if len(state) == 4:
        x, y, theta, gear = state
        if gear.lower() == "backward":
            theta = (theta + 180) % 360
        return (x, y, theta)
    return state

def get_car_corners(state, car_length=CAR_LENGTH, car_width=CAR_WIDTH):
    x, y, theta = effective_state(state)
    theta_rad = math.radians(theta)
    hl = car_length / 2.0
    hw = car_width / 2.0
    offsets = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    corners = []
    for dx, dy in offsets:
        cx = x + dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
        cy = y + dx * math.sin(theta_rad) + dy * math.cos(theta_rad)
        corners.append((cx, cy))
    return corners

def get_bounding_box(corners):
    xs, ys = zip(*corners)
    return (min(xs), min(ys), max(xs), max(ys))

def rects_overlap(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

def apply_movement(state, movement_idx):
    dx, dy, dtheta = MOVEMENTS_LIST[movement_idx]
    x, y, theta = effective_state(state)
    theta_rad = math.radians(theta)
    new_x = x + dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
    new_y = y + dx * math.sin(theta_rad) + dy * math.cos(theta_rad)
    new_theta = (theta + dtheta) % 360
    return (new_x, new_y, new_theta)

def calculate_fitness(state, target):
    x, y, theta = effective_state(state)
    tx, ty, ttheta = effective_state(target)
    dist_error = math.hypot(x - tx, y - ty)
    angle_error = abs((theta - ttheta) % 360)
    angle_error = min(angle_error, 360 - angle_error)
    return dist_error + 10 * angle_error

def check_car_collision(state, obstacles):
    # まず、各コーナーがキャンバス内にあるかチェック
    corners = get_car_corners(state)
    for cx, cy in corners:
        if cx < 0 or cx > ENV_WIDTH or cy < 0 or cy > ENV_HEIGHT:
            return True
    # 次に障害物との衝突チェック
    for cx, cy in corners:
        for obs in obstacles:
            x_min, y_min, x_max, y_max = obs
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                return True
    return False

def initialize_population():
    return [
        [random.randint(0, len(MOVEMENTS_LIST) - 1) for _ in range(random.randint(1, MAX_GENES))]
        for _ in range(POPULATION_SIZE)
    ]

def evaluate_population(population, obstacles, START_STATE, TARGET_STATE):
    fitness_scores = []
    for genome in population:
        state = START_STATE
        collision = False
        for move in genome:
            state = apply_movement(state, move)
            if check_car_collision(state, obstacles):
                collision = True
                break
        if collision:
            fitness_scores.append(1e6)
        else:
            fitness_scores.append(calculate_fitness(state, TARGET_STATE))
    return fitness_scores

def select_parents(population, fitness_scores):
    top_indices = nsmallest(ELITE_COUNT, range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    return [population[i] for i in top_indices]

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE and len(parent1) > 1 and len(parent2) > 1:
        cut = random.randint(1, min(len(parent1), len(parent2)) - 1)
        return parent1[:cut] + parent2[cut:]
    return parent1[:] if random.random() < 0.5 else parent2[:]

def mutate(genome):
    if genome and random.random() < MUTATION_RATE:
        idx = random.randint(0, len(genome) - 1)
        genome[idx] = random.randint(0, len(MOVEMENTS_LIST) - 1)
    return genome

def add_gene(genome):
    if random.random() < ADDITION_RATE and len(genome) < MAX_GENES:
        genome.append(random.randint(0, len(MOVEMENTS_LIST) - 1))
    return genome

def delete_gene(genome):
    if len(genome) > 1 and random.random() < DELETION_RATE:
        del genome[random.randint(0, len(genome) - 1)]
    return genome

def genetic_operator(parent1, parent2):
    offspring = crossover(parent1, parent2)
    offspring = mutate(offspring)
    offspring = add_gene(offspring)
    offspring = delete_gene(offspring)
    return offspring

def run_genetic_algorithm(obstacles, START_STATE, TARGET_STATE):
    population = initialize_population()
    best_fitness = float('inf')
    best_genome = None
    start_time = time.time()
    for generation in range(GENERATIONS):
        fitness_scores = evaluate_population(population, obstacles, START_STATE, TARGET_STATE)
        gen_best_fitness = min(fitness_scores)
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_genome = population[fitness_scores.index(gen_best_fitness)]
        if best_fitness < 10:
            print(f"Early stopping at generation {generation} with fitness {best_fitness:.2f}")
            break
        parents = select_parents(population, fitness_scores)
        new_population = parents[:]  # エリート個体保持
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            offspring = genetic_operator(parent1, parent2)
            new_population.append(offspring)
        population = new_population
        if generation % 500 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
    end_time = time.time()
    print(f"Genetic algorithm completed in {end_time - start_time:.2f} seconds.")
    return best_genome, best_fitness

def compute_trajectory(genome, START_STATE):
    state = START_STATE
    trajectory = [state]
    moves = []
    for move in genome:
        state = apply_movement(state, move)
        trajectory.append(state)
        moves.append(move)
    return trajectory, moves

# ======================================================
# Tkinter Canvas UI（ペイント風）＋直感的な角度・ギア設定パネル
# ======================================================

# まずルートウィンドウを作成
root = tk.Tk()
root.title("駐車場シミュレーション設定 (ペイント風)")

# --- 右側設定パネル ---
frame_config = ttk.Frame(root)
frame_config.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

# スタート設定用パネル
frame_start = ttk.LabelFrame(frame_config, text="スタート設定", padding=5)
frame_start.pack(fill=tk.X, pady=5)
ttk.Label(frame_start, text="キャンバス上でスタート位置をクリック後、以下で調整").pack(anchor="w")
ttk.Label(frame_start, text="角度:").pack(anchor="w")
start_angle_var = tk.IntVar(value=0)
scale_start = ttk.Scale(frame_start, from_=0, to=359, orient=tk.HORIZONTAL, variable=start_angle_var)
scale_start.pack(fill=tk.X)
ttk.Label(frame_start, text="ギア:").pack(anchor="w")
start_gear_var = tk.StringVar(value="forward")
option_start = ttk.Combobox(frame_start, textvariable=start_gear_var, values=["forward", "backward"], state="readonly")
option_start.pack(fill=tk.X)

# ゴール設定用パネル
frame_goal = ttk.LabelFrame(frame_config, text="ゴール設定", padding=5)
frame_goal.pack(fill=tk.X, pady=5)
ttk.Label(frame_goal, text="キャンバス上でゴール位置をクリック後、以下で調整").pack(anchor="w")
ttk.Label(frame_goal, text="角度:").pack(anchor="w")
goal_angle_var = tk.IntVar(value=0)
scale_goal = ttk.Scale(frame_goal, from_=0, to=359, orient=tk.HORIZONTAL, variable=goal_angle_var)
scale_goal.pack(fill=tk.X)
ttk.Label(frame_goal, text="ギア:").pack(anchor="w")
goal_gear_var = tk.StringVar(value="forward")
option_goal = ttk.Combobox(frame_goal, textvariable=goal_gear_var, values=["forward", "backward"], state="readonly")
option_goal.pack(fill=tk.X)

# --- キャンバスエリア ---
frame_canvas = ttk.Frame(root)
frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(frame_canvas, width=ENV_WIDTH, height=ENV_HEIGHT, bg="white")
canvas.pack()

# グローバル変数：スタート、ゴール、障害物
start_pos = None    # 格納形式: (x, y, theta, gear)
goal_pos = None     # 格納形式: (x, y, theta, gear)
obstacles = []      # 各障害物は (x_min, y_min, x_max, y_max)

# キャンバス上の描画オブジェクトID
start_id = None
goal_id = None
start_arrow_id = None
goal_arrow_id = None
obstacle_ids = []

# 現在のモード（"start", "goal", "obstacle"）
current_mode = tk.StringVar(master=root, value="none")

# 障害物描画用変数
drag_start = None
current_rect = None

def set_mode(mode):
    current_mode.set(mode)
    status_label.config(text=f"Mode: {mode}")

def update_start_polygon():
    global start_pos, start_id, start_arrow_id
    if start_pos is not None:
        s = (start_pos[0], start_pos[1], start_angle_var.get(), start_gear_var.get())
        start_pos = s
        if start_id is not None:
            canvas.delete(start_id)
        if start_arrow_id is not None:
            canvas.delete(start_arrow_id)
        corners = get_car_corners(start_pos)
        start_id = canvas.create_polygon(corners, fill="green", outline="black")
        cx, cy, _ = effective_state(start_pos)
        front_mid = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        start_arrow_id = canvas.create_line(cx, cy, front_mid[0], front_mid[1], arrow=tk.LAST, fill="white", width=2)

def update_goal_polygon():
    global goal_pos, goal_id, goal_arrow_id
    if goal_pos is not None:
        s = (goal_pos[0], goal_pos[1], goal_angle_var.get(), goal_gear_var.get())
        goal_pos = s
        if goal_id is not None:
            canvas.delete(goal_id)
        if goal_arrow_id is not None:
            canvas.delete(goal_arrow_id)
        corners = get_car_corners(goal_pos)
        goal_id = canvas.create_polygon(corners, fill="red", outline="black")
        cx, cy, _ = effective_state(goal_pos)
        front_mid = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        goal_arrow_id = canvas.create_line(cx, cy, front_mid[0], front_mid[1], arrow=tk.LAST, fill="white", width=2)

def canvas_click(event):
    global start_pos, goal_pos, start_id, goal_id, drag_start, start_arrow_id, goal_arrow_id
    mode = current_mode.get()
    if mode == "start":
        x, y = event.x, event.y
        start_pos = (x, y, start_angle_var.get(), start_gear_var.get())
        if start_id is not None:
            canvas.delete(start_id)
        if start_arrow_id is not None:
            canvas.delete(start_arrow_id)
        corners = get_car_corners(start_pos)
        start_id = canvas.create_polygon(corners, fill="green", outline="black")
        cx, cy, _ = effective_state(start_pos)
        front_mid = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        start_arrow_id = canvas.create_line(cx, cy, front_mid[0], front_mid[1], arrow=tk.LAST, fill="white", width=2)
    elif mode == "goal":
        x, y = event.x, event.y
        goal_pos = (x, y, goal_angle_var.get(), goal_gear_var.get())
        if goal_id is not None:
            canvas.delete(goal_id)
        if goal_arrow_id is not None:
            canvas.delete(goal_arrow_id)
        corners = get_car_corners(goal_pos)
        goal_id = canvas.create_polygon(corners, fill="red", outline="black")
        cx, cy, _ = effective_state(goal_pos)
        front_mid = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        goal_arrow_id = canvas.create_line(cx, cy, front_mid[0], front_mid[1], arrow=tk.LAST, fill="white", width=2)
    elif mode == "obstacle":
        global drag_start
        drag_start = (event.x, event.y)

def canvas_drag(event):
    global current_rect, drag_start
    if current_mode.get() == "obstacle" and drag_start is not None:
        x0, y0 = drag_start
        x1, y1 = event.x, event.y
        if current_rect is not None:
            canvas.delete(current_rect)
        current_rect = canvas.create_rectangle(x0, y0, x1, y1, outline="purple", dash=(2,2))

def canvas_release(event):
    global current_rect, drag_start, obstacles, obstacle_ids
    if current_mode.get() == "obstacle" and drag_start is not None:
        x0, y0 = drag_start
        x1, y1 = event.x, event.y
        x_min, y_min = min(x0, x1), min(y0, y1)
        x_max, y_max = max(x0, x1), max(y0, y1)
        obstacles.append((x_min, y_min, x_max, y_max))
        if current_rect is not None:
            obstacle_ids.append(current_rect)
            current_rect = None
        drag_start = None

# 追加機能：障害物クリアボタンの実装
def clear_obstacles():
    global obstacles, obstacle_ids
    for oid in obstacle_ids:
        canvas.delete(oid)
    obstacle_ids.clear()
    obstacles = []

# 下部ボタンフレーム（モード切替用）
frame_bottom = ttk.Frame(frame_canvas)
frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)

ttk.Button(frame_bottom, text="スタート設定", command=lambda: set_mode("start")).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_bottom, text="ゴール設定", command=lambda: set_mode("goal")).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_bottom, text="障害物描画", command=lambda: set_mode("obstacle")).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_bottom, text="障害物クリア", command=clear_obstacles).pack(side=tk.LEFT, padx=5, pady=5)
status_label = ttk.Label(frame_bottom, text="Mode: none")
status_label.pack(side=tk.LEFT, padx=10)
ttk.Button(frame_bottom, text="シミュレーション開始", command=lambda: run_simulation()).pack(side=tk.RIGHT, padx=5, pady=5)

canvas.bind("<Button-1>", canvas_click)
canvas.bind("<B1-Motion>", canvas_drag)
canvas.bind("<ButtonRelease-1>", canvas_release)

# trace_add を利用して設定パネルの変更を反映する
start_angle_var.trace_add("write", lambda *args: update_start_polygon())
goal_angle_var.trace_add("write", lambda *args: update_goal_polygon())
start_gear_var.trace_add("write", lambda *args: update_start_polygon())
goal_gear_var.trace_add("write", lambda *args: update_goal_polygon())

def run_simulation():
    global start_pos, goal_pos, obstacles
    # 先頭でグローバル変数を宣言
    global start_pos, goal_pos, obstacles
    if start_pos is None or goal_pos is None:
        messagebox.showerror("エラー", "スタート位置とゴール位置をキャンバス上で設定してください。")
        return
    if not obstacles:
        messagebox.showerror("エラー", "少なくとも1つの障害物を設定してください。")
        return

    # 画面の端を障害物として追加（厚さ5）
    boundary_thickness = 5
    boundaries = [
        (0, 0, ENV_WIDTH, boundary_thickness),                          # 上端
        (0, ENV_HEIGHT - boundary_thickness, ENV_WIDTH, ENV_HEIGHT),       # 下端
        (0, 0, boundary_thickness, ENV_HEIGHT),                          # 左端
        (ENV_WIDTH - boundary_thickness, 0, ENV_WIDTH, ENV_HEIGHT)         # 右端
    ]
    sim_obstacles = obstacles.copy()
    sim_obstacles.extend(boundaries)
    
    # 有効なスタート／ゴール状態（プルダウンとスライダーで設定された値を利用）
    effective_start = effective_state(start_pos)
    effective_goal = effective_state(goal_pos)
    
    best_genome, best_fitness = run_genetic_algorithm(sim_obstacles, effective_start, effective_goal)
    print("Best Fitness:", best_fitness)
    trajectory, moves = compute_trajectory(best_genome, effective_start)
    final_state = trajectory[-1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, move_idx in enumerate(moves):
        start_pt = trajectory[i]
        end_pt = trajectory[i+1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        color = 'red' if move_idx >= 11 else 'blue'
        ax.arrow(start_pt[0], start_pt[1], dx, dy, head_width=3, head_length=3,
                 fc=color, ec=color, length_includes_head=True)
    
    start_corners = get_car_corners(start_pos)
    goal_corners = get_car_corners(goal_pos)
    start_polygon = Polygon(start_corners, closed=True, facecolor='green', alpha=0.5, label='Start')
    goal_polygon = Polygon(goal_corners, closed=True, facecolor='red', alpha=0.5, label='Goal')
    ax.add_patch(start_polygon)
    ax.add_patch(goal_polygon)
    
    for obs in sim_obstacles:
        x_min, y_min, x_max, y_max = obs
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1,
                                 edgecolor='purple', facecolor='none')
        ax.add_patch(rect)
    
    traj_xy = np.array([[p[0], p[1]] for p in trajectory])
    ax.plot(traj_xy[:,0], traj_xy[:,1], linestyle='--', color='gray', alpha=0.5)
    forward_patch = Patch(color='blue', label='Forward/Turning')
    backward_patch = Patch(color='red', label='Backward')
    ax.legend(handles=[forward_patch, backward_patch])
    ax.set_title(f"シミュレーション結果 (Best Fitness = {best_fitness:.2f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    plt.show()

root.mainloop()
