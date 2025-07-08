import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import random
import math
import time
from heapq import nsmallest
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Patch
import numpy as np

# ======================================================
# 定数・初期パラメータ
# ======================================================

# 車両サイズ（普通乗用車の比率 60×24）
CAR_LENGTH = 60.0
CAR_WIDTH  = 24.0

# シミュレーション領域（キャンバスサイズ＝ウィンドウサイズ）
ENV_WIDTH = 800
ENV_HEIGHT = 600

# GA の各パラメータ（初期値、後でUIから上書き）
POPULATION_SIZE = 40
GENERATIONS = 10000
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.3
ADDITION_RATE = 0.3
DELETION_RATE = 0.3
MAX_GENES = 150
ELITE_COUNT = 5

# 経路長ペナルティの重み
ROUTE_LENGTH_WEIGHT = 0.1

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

# ======================================================
# 衝突判定のためのポリゴン交差判定（SATを用いる）
# ======================================================
def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def normalize(v):
    norm = math.hypot(v[0], v[1])
    if norm == 0:
        return (0,0)
    return (v[0]/norm, v[1]/norm)

def get_axes(poly):
    axes = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1)%n]
        edge = (p2[0]-p1[0], p2[1]-p1[1])
        normal = (-edge[1], edge[0])
        axes.append(normalize(normal))
    return axes

def project(poly, axis):
    dots = [dot(p, axis) for p in poly]
    return min(dots), max(dots)

def polygons_intersect(poly1, poly2):
    axes = get_axes(poly1) + get_axes(poly2)
    for axis in axes:
        proj1 = project(poly1, axis)
        proj2 = project(poly2, axis)
        if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
            return False
    return True

# ======================================================
# 状態管理・描画関連
# ======================================================
# スタート／ゴール状態は (x, y, theta, gear) の4要素タプルとして保持
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

# ======================================================
# GA 用関数
# ======================================================
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
    car_poly = get_car_corners(state)
    # キャンバス外チェック
    for cx, cy in car_poly:
        if cx < 0 or cx > ENV_WIDTH or cy < 0 or cy > ENV_HEIGHT:
            return True
    # 障害物との衝突チェック：角だけでなく、側面との交差もSATで判定
    for obs in obstacles:
        obs_poly = [(obs[0], obs[1]), (obs[2], obs[1]), (obs[2], obs[3]), (obs[0], obs[3])]
        if polygons_intersect(car_poly, obs_poly):
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
        total_distance = 0.0
        prev = effective_state(START_STATE)
        for move in genome:
            state = apply_movement(state, move)
            curr = effective_state(state)
            total_distance += math.hypot(curr[0]-prev[0], curr[1]-prev[1])
            prev = curr
            if check_car_collision(state, obstacles):
                collision = True
                break
        if collision:
            fitness_scores.append(1e6)
        else:
            base_fit = calculate_fitness(state, TARGET_STATE)
            fitness_scores.append(base_fit + ROUTE_LENGTH_WEIGHT * total_distance)
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
# Tkinter UI 部分
# ======================================================

# ルートウィンドウ作成
root = tk.Tk()
root.title("駐車場シミュレーション設定 (ペイント風)")

# --- 右側設定パネル（スタート／ゴール＋GAパラメータ設定） ---
frame_config = ttk.Frame(root)
frame_config.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

# スタート設定パネル
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

# ゴール設定パネル
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

# GAパラメータ設定パネル（重複せず1セットのみ）
frame_ga = ttk.LabelFrame(frame_config, text="GAパラメータ設定", padding=5)
frame_ga.pack(fill=tk.X, pady=5)
def add_ga_param(label_text, default_value):
    f = ttk.Frame(frame_ga)
    f.pack(fill=tk.X, pady=2)
    ttk.Label(f, text=label_text, width=15).pack(side=tk.LEFT)
    var = tk.StringVar(value=str(default_value))
    ttk.Entry(f, textvariable=var, width=10).pack(side=tk.LEFT)
    return var

pop_var = add_ga_param("個体数", POPULATION_SIZE)
gen_var = add_ga_param("世代数", GENERATIONS)
mut_var = add_ga_param("突然変異率", MUTATION_RATE)
cross_var = add_ga_param("交叉率", CROSSOVER_RATE)
add_var = add_ga_param("追加率", ADDITION_RATE)
del_var = add_ga_param("削除率", DELETION_RATE)
max_gene_var = add_ga_param("最大遺伝子数", MAX_GENES)
elite_var = add_ga_param("エリート数", ELITE_COUNT)

# --- キャンバスエリア ---
frame_canvas = ttk.Frame(root)
frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas = tk.Canvas(frame_canvas, width=ENV_WIDTH, height=ENV_HEIGHT, bg="white")
canvas.pack()

# グローバル変数：スタート、ゴール、障害物
start_pos = None    # (x, y, theta, gear)
goal_pos = None     # (x, y, theta, gear)
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

def clear_obstacles():
    global obstacles, obstacle_ids
    for oid in obstacle_ids:
        canvas.delete(oid)
    obstacle_ids.clear()
    obstacles = []

# 下部ボタンフレーム
frame_bottom = ttk.Frame(frame_canvas)
frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)
ttk.Button(frame_bottom, text="スタート設定", command=lambda: set_mode("start")).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_bottom, text="ゴール設定", command=lambda: set_mode("goal")).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_bottom, text="障害物描画", command=lambda: set_mode("obstacle")).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_bottom, text="障害物クリア", command=clear_obstacles).pack(side=tk.LEFT, padx=5, pady=5)
status_label = ttk.Label(frame_bottom, text="Mode: none")
status_label.pack(side=tk.LEFT, padx=10)
ttk.Button(frame_bottom, text="シミュレーション開始", command=lambda: start_simulation_thread()).pack(side=tk.RIGHT, padx=5, pady=5)

canvas.bind("<Button-1>", canvas_click)
canvas.bind("<B1-Motion>", canvas_drag)
canvas.bind("<ButtonRelease-1>", canvas_release)

# trace_add を利用して設定パネルの変更を反映
start_angle_var.trace_add("write", lambda *args: update_start_polygon())
goal_angle_var.trace_add("write", lambda *args: update_goal_polygon())
start_gear_var.trace_add("write", lambda *args: update_start_polygon())
goal_gear_var.trace_add("write", lambda *args: update_goal_polygon())

# --- シミュレーション実行部 ---
def simulation_thread():
    global start_pos, goal_pos, obstacles
    # --- ここから自動初期値セット ---
    if start_pos is None:
        # デフォルト値 (画面左上)
        start_pos = (100, 100, 0, "forward")
        root.after(0, update_start_polygon)
    if goal_pos is None:
        # デフォルト値 (画面右下)
        goal_pos = (700, 500, 0, "forward")
        root.after(0, update_goal_polygon)
    if not obstacles:
        # デフォルト障害物（中央に1つ）
        obstacles.append((300, 200, 500, 400))
        oid = canvas.create_rectangle(300, 200, 500, 400, outline="purple", dash=(2,2))
        obstacle_ids.append(oid)
    # --- ここまで自動初期値セット ---

    try:
        global POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ADDITION_RATE, DELETION_RATE, MAX_GENES, ELITE_COUNT
        POPULATION_SIZE = int(pop_var.get())
        GENERATIONS = int(gen_var.get())
        MUTATION_RATE = float(mut_var.get())
        CROSSOVER_RATE = float(cross_var.get())
        ADDITION_RATE = float(add_var.get())
        DELETION_RATE = float(del_var.get())
        MAX_GENES = int(max_gene_var.get())
        ELITE_COUNT = int(elite_var.get())
    except Exception as e:
        messagebox.showerror("エラー", f"GAパラメータの入力値に誤りがあります。\n{e}")
        run_popup.destroy()
        return

    effective_start = effective_state(start_pos)
    effective_goal = effective_state(goal_pos)
    boundary_thickness = 5
    boundaries = [
        (0, 0, ENV_WIDTH, boundary_thickness),
        (0, ENV_HEIGHT - boundary_thickness, ENV_WIDTH, ENV_HEIGHT),
        (0, 0, boundary_thickness, ENV_HEIGHT),
        (ENV_WIDTH - boundary_thickness, 0, ENV_WIDTH, ENV_HEIGHT)
    ]
    sim_obstacles = obstacles.copy()
    sim_obstacles.extend(boundaries)
    
    best_genome, best_fitness = run_genetic_algorithm(sim_obstacles, effective_start, effective_goal)
    print("Best Fitness:", best_fitness)
    trajectory, moves = compute_trajectory(best_genome, effective_start)
    root.after(0, lambda: show_animation(trajectory, moves, best_fitness, sim_obstacles))
    root.after(0, run_popup.destroy)

def start_simulation_thread():
    global run_popup
    run_popup = tk.Toplevel(root)
    run_popup.title("実行中")
    ttk.Label(run_popup, text="シミュレーション実行中...").pack(padx=20, pady=20)
    run_popup.geometry("200x100")
    t = threading.Thread(target=simulation_thread)
    t.start()

def show_animation(trajectory, moves, best_fitness, sim_obstacles):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(0, ENV_HEIGHT)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"シミュレーション結果 (Best Fitness = {best_fitness:.2f})")
    ax.grid(True)
    start_poly = Polygon(get_car_corners(start_pos), closed=True, facecolor='green', alpha=0.5, label='Start')
    goal_poly = Polygon(get_car_corners(goal_pos), closed=True, facecolor='red', alpha=0.5, label='Goal')
    ax.add_patch(start_poly)
    ax.add_patch(goal_poly)
    for obs in sim_obstacles:
        x_min, y_min, x_max, y_max = obs
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='purple', facecolor='none')
        ax.add_patch(rect)
    vehicle_patch = Polygon(get_car_corners(trajectory[0]), closed=True, facecolor='blue', alpha=0.8)
    ax.add_patch(vehicle_patch)
    def update(frame):
        state = trajectory[frame]
        new_corners = get_car_corners(state)
        vehicle_patch.set_xy(new_corners)
        return vehicle_patch,
    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=200, blit=True)
    try:
        Writer = animation.FFMpegWriter
        writer = Writer(fps=5, metadata=dict(artist='Simulation'), bitrate=1800)
        ani.save("simulation_result.mp4", writer=writer)
        print("アニメーションを simulation_result.mp4 として保存しました。")
    except Exception as e:
        print("アニメーション保存に失敗しました:", e)
    plt.legend()
    plt.show()

root.mainloop()

