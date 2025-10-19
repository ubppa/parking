import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
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
import csv
import os
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

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


@dataclass
class VehicleConfig:
    length: float = CAR_LENGTH
    width: float = CAR_WIDTH
    wheel_base: float = 40.0
    steering_angles: List[float] = field(default_factory=lambda: [15.0, 30.0, 45.0, 60.0])
    forward_steps: List[float] = field(default_factory=lambda: [20.0, 27.0, 35.0])
    backward_steps: List[float] = field(default_factory=lambda: [20.0, 27.0, 35.0])
    min_turn_radius: float = 80.0

    @classmethod
    def from_dict(cls, data: Dict) -> "VehicleConfig":
        if data is None:
            return cls()
        return cls(
            length=float(data.get("length", CAR_LENGTH)),
            width=float(data.get("width", CAR_WIDTH)),
            wheel_base=float(data.get("wheel_base", 40.0)),
            steering_angles=[float(v) for v in data.get("steering_angles", [15.0, 30.0, 45.0, 60.0])],
            forward_steps=[float(v) for v in data.get("forward_steps", [20.0, 27.0, 35.0])],
            backward_steps=[float(v) for v in data.get("backward_steps", [20.0, 27.0, 35.0])],
            min_turn_radius=float(data.get("min_turn_radius", 80.0)),
        )


@dataclass
class EnvironmentConfig:
    width: float = ENV_WIDTH
    height: float = ENV_HEIGHT
    obstacles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    start: Optional[Tuple[float, float, float, str]] = None
    goal: Optional[Tuple[float, float, float, str]] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "EnvironmentConfig":
        if data is None:
            return cls()
        obstacles = []
        for item in data.get("obstacles", []):
            obstacles.append(
                (
                    float(item.get("x_min", 0.0)),
                    float(item.get("y_min", 0.0)),
                    float(item.get("x_max", 0.0)),
                    float(item.get("y_max", 0.0)),
                )
            )
        start_cfg = data.get("start")
        goal_cfg = data.get("goal")
        start = None
        goal = None
        if start_cfg:
            start = (
                float(start_cfg.get("x", 0.0)),
                float(start_cfg.get("y", 0.0)),
                float(start_cfg.get("theta", 0.0)),
                str(start_cfg.get("gear", "forward")),
            )
        if goal_cfg:
            goal = (
                float(goal_cfg.get("x", 0.0)),
                float(goal_cfg.get("y", 0.0)),
                float(goal_cfg.get("theta", 0.0)),
                str(goal_cfg.get("gear", "forward")),
            )
        return cls(
            width=float(data.get("width", ENV_WIDTH)),
            height=float(data.get("height", ENV_HEIGHT)),
            obstacles=obstacles,
            start=start,
            goal=goal,
        )


@dataclass
class GAConfig:
    population_size: int = POPULATION_SIZE
    generations: int = GENERATIONS
    mutation_rate: float = MUTATION_RATE
    crossover_rate: float = CROSSOVER_RATE
    addition_rate: float = ADDITION_RATE
    deletion_rate: float = DELETION_RATE
    max_genes: int = MAX_GENES
    elite_count: int = ELITE_COUNT

    @classmethod
    def from_dict(cls, data: Dict) -> "GAConfig":
        if data is None:
            return cls()
        return cls(
            population_size=int(data.get("population_size", POPULATION_SIZE)),
            generations=int(data.get("generations", GENERATIONS)),
            mutation_rate=float(data.get("mutation_rate", MUTATION_RATE)),
            crossover_rate=float(data.get("crossover_rate", CROSSOVER_RATE)),
            addition_rate=float(data.get("addition_rate", ADDITION_RATE)),
            deletion_rate=float(data.get("deletion_rate", DELETION_RATE)),
            max_genes=int(data.get("max_genes", MAX_GENES)),
            elite_count=int(data.get("elite_count", ELITE_COUNT)),
        )


LOG_DIR = Path("logs")
vehicle_config = VehicleConfig()
environment_config = EnvironmentConfig()
ga_config = GAConfig()

# 経路長ペナルティの重み
ROUTE_LENGTH_WEIGHT = 0.1

# 動作パターン（前進・後退は短・中・長、旋回は複数角度）
DEFAULT_MOVEMENTS = [
    (20.0, 0.0, 0.0),
    (27.0, 0.0, 0.0),
    (35.0, 0.0, 0.0),
    (27.0, 5.0, 15.0),
    (28.2, 8.9, 30.5),
    (30.0, 12.0, 45.0),
    (32.0, 15.0, 60.0),
    (27.0, -5.0, -15.0),
    (28.7, -8.9, -30.5),
    (30.0, -12.0, -45.0),
    (32.0, -15.0, -60.0),
    (-20.0, 0.0, 0.0),
    (-27.0, 0.0, 0.0),
    (-35.0, 0.0, 0.0),
]
MOVEMENTS_LIST = []


def generate_movements_from_vehicle(config: VehicleConfig) -> List[Tuple[float, float, float]]:
    movements: List[Tuple[float, float, float]] = []

    for dist in config.forward_steps:
        if dist > 0:
            movements.append((float(dist), 0.0, 0.0))
    for dist in config.backward_steps:
        if dist > 0:
            movements.append((-float(dist), 0.0, 0.0))

    steering_angles = [angle for angle in config.steering_angles if abs(angle) > 1e-3]

    def add_arc(step_length: float, heading_deg: float):
        step_abs = abs(step_length)
        if step_abs < 1e-3:
            return
        desired_delta_rad = math.radians(abs(heading_deg))
        if desired_delta_rad < 1e-4:
            return
        radius = step_abs / desired_delta_rad
        radius = max(radius, config.min_turn_radius)
        actual_delta_rad = step_abs / radius
        travel_sign = 1.0 if step_length >= 0 else -1.0
        base_dx = travel_sign * radius * math.sin(actual_delta_rad)
        base_dy = radius * (1 - math.cos(actual_delta_rad))
        base_delta_deg = math.degrees(actual_delta_rad) * travel_sign
        movements.append((base_dx, base_dy, base_delta_deg))
        movements.append((base_dx, -base_dy, -base_delta_deg))

    for dist in config.forward_steps:
        for angle in steering_angles:
            add_arc(float(dist), float(angle))
    for dist in config.backward_steps:
        for angle in steering_angles:
            add_arc(-float(dist), float(angle))

    unique: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    for move in movements:
        key = (round(move[0], 3), round(move[1], 3), round(move[2], 3))
        if key not in unique:
            unique[key] = move

    if not unique:
        return DEFAULT_MOVEMENTS[:]
    return list(unique.values())


def refresh_movement_library(config: Optional[VehicleConfig] = None) -> None:
    global MOVEMENTS_LIST
    cfg = config or vehicle_config
    MOVEMENTS_LIST = generate_movements_from_vehicle(cfg)


refresh_movement_library()


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    parts = [segment.strip() for segment in value.split(",")]
    return [float(part) for part in parts if part]


def redraw_obstacles_on_canvas() -> None:
    if canvas is None:
        return
    for oid in obstacle_ids:
        canvas.delete(oid)
    obstacle_ids.clear()
    for obs in obstacles:
        x_min, y_min, x_max, y_max = obs
        oid = canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="purple", dash=(2, 2))
        obstacle_ids.append(oid)


def apply_vehicle_config(config: VehicleConfig, update_ui: bool = True) -> None:
    global vehicle_config
    vehicle_config = config
    refresh_movement_library(vehicle_config)
    if update_ui and all(var is not None for var in (vehicle_length_var, vehicle_width_var, wheel_base_var, steering_angles_var, forward_steps_var, backward_steps_var, min_turn_radius_var)):
        vehicle_length_var.set(f"{vehicle_config.length:.2f}")
        vehicle_width_var.set(f"{vehicle_config.width:.2f}")
        wheel_base_var.set(f"{vehicle_config.wheel_base:.2f}")
        steering_angles_var.set(", ".join(f"{angle:.1f}" for angle in vehicle_config.steering_angles))
        forward_steps_var.set(", ".join(f"{step:.1f}" for step in vehicle_config.forward_steps))
        backward_steps_var.set(", ".join(f"{step:.1f}" for step in vehicle_config.backward_steps))
        min_turn_radius_var.set(f"{vehicle_config.min_turn_radius:.2f}")
    update_start_polygon()
    update_goal_polygon()
    if status_label is not None:
        mode_text = current_mode.get() if current_mode is not None else "none"
        status_label.config(text=f"Mode: {mode_text} | 動作 {len(MOVEMENTS_LIST)}件")


def apply_environment_config(config: EnvironmentConfig, update_ui: bool = True) -> None:
    global environment_config, ENV_WIDTH, ENV_HEIGHT, obstacles, start_pos, goal_pos
    environment_config = config
    ENV_WIDTH = config.width
    ENV_HEIGHT = config.height
    if canvas is not None:
        canvas.config(width=ENV_WIDTH, height=ENV_HEIGHT)
    obstacles = config.obstacles.copy()
    redraw_obstacles_on_canvas()

    if config.start:
        start_pos = config.start
        if update_ui and start_angle_var is not None:
            start_angle_var.set(int(round(config.start[2])) % 360)
        if update_ui and start_gear_var is not None:
            start_gear_var.set(config.start[3])
    if config.goal:
        goal_pos = config.goal
        if update_ui and goal_angle_var is not None:
            goal_angle_var.set(int(round(config.goal[2])) % 360)
        if update_ui and goal_gear_var is not None:
            goal_gear_var.set(config.goal[3])
    update_start_polygon()
    update_goal_polygon()


def apply_ga_config(config: GAConfig, update_ui: bool = True) -> None:
    global ga_config, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ADDITION_RATE, DELETION_RATE, MAX_GENES, ELITE_COUNT
    ga_config = config
    POPULATION_SIZE = config.population_size
    GENERATIONS = config.generations
    MUTATION_RATE = config.mutation_rate
    CROSSOVER_RATE = config.crossover_rate
    ADDITION_RATE = config.addition_rate
    DELETION_RATE = config.deletion_rate
    MAX_GENES = config.max_genes
    ELITE_COUNT = config.elite_count
    if update_ui and all(var is not None for var in (pop_var, gen_var, mut_var, cross_var, add_var, del_var, max_gene_var, elite_var)):
        pop_var.set(str(config.population_size))
        gen_var.set(str(config.generations))
        mut_var.set(str(config.mutation_rate))
        cross_var.set(str(config.crossover_rate))
        add_var.set(str(config.addition_rate))
        del_var.set(str(config.deletion_rate))
        max_gene_var.set(str(config.max_genes))
        elite_var.set(str(config.elite_count))


def load_configuration_from_file(path: str) -> Tuple[Optional[VehicleConfig], Optional[EnvironmentConfig], Optional[GAConfig]]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    vehicle_data = payload.get("vehicle")
    environment_data = payload.get("environment")
    ga_data = payload.get("ga")
    vehicle_conf = VehicleConfig.from_dict(vehicle_data) if vehicle_data else None
    environment_conf = EnvironmentConfig.from_dict(environment_data) if environment_data else None
    ga_conf = GAConfig.from_dict(ga_data) if ga_data else None
    return vehicle_conf, environment_conf, ga_conf


def load_configuration_dialog():
    if root is None:
        return
    file_path = filedialog.askopenfilename(title="設定ファイルを選択", filetypes=[("JSON", "*.json"), ("All Files", "*.*")])
    if not file_path:
        return
    try:
        vehicle_conf, environment_conf, ga_conf = load_configuration_from_file(file_path)
    except Exception as exc:
        messagebox.showerror("読込エラー", f"設定ファイルの読み込みに失敗しました。\n{exc}")
        return
    if vehicle_conf:
        apply_vehicle_config(vehicle_conf)
    if environment_conf:
        apply_environment_config(environment_conf)
    if ga_conf:
        apply_ga_config(ga_conf)
    if config_path_var is not None:
        config_path_var.set(Path(file_path).name)


def apply_vehicle_settings_from_ui():
    if any(var is None for var in (vehicle_length_var, vehicle_width_var, wheel_base_var, steering_angles_var, forward_steps_var, backward_steps_var, min_turn_radius_var)):
        return
    try:
        length = float(vehicle_length_var.get())
        width = float(vehicle_width_var.get())
        wheel_base = float(wheel_base_var.get())
        steering_angles = parse_float_list(steering_angles_var.get())
        forward_steps = parse_float_list(forward_steps_var.get())
        backward_steps = parse_float_list(backward_steps_var.get())
        min_turn_radius = float(min_turn_radius_var.get())
        if not steering_angles:
            raise ValueError("旋回角を1つ以上入力してください。")
        if not forward_steps:
            raise ValueError("前進距離を1つ以上入力してください。")
        if not backward_steps:
            raise ValueError("後退距離を1つ以上入力してください。")
    except ValueError as exc:
        messagebox.showerror("入力エラー", f"車両設定の値に誤りがあります。\n{exc}")
        return

    new_config = VehicleConfig(
        length=length,
        width=width,
        wheel_base=wheel_base,
        steering_angles=steering_angles,
        forward_steps=forward_steps,
        backward_steps=backward_steps,
        min_turn_radius=min_turn_radius,
    )
    apply_vehicle_config(new_config)
    if status_label is not None:
        mode_text = current_mode.get() if current_mode is not None else "none"
        status_label.config(text=f"Mode: {mode_text} | 動作 {len(MOVEMENTS_LIST)}件 (車両設定更新)")


def append_status_message(message: str) -> None:
    if not message:
        return

    def _append():
        if progress_log is None or not progress_log.winfo_exists():
            return
        progress_log.configure(state="normal")
        progress_log.insert(tk.END, f"{datetime.now():%H:%M:%S} {message}\n")
        progress_log.see(tk.END)
        progress_log.configure(state="disabled")

    if root is not None:
        root.after(0, _append)
    else:
        _append()


def update_progress(value: Optional[float] = None, message: Optional[str] = None) -> None:
    def _update():
        if run_popup is None or not run_popup.winfo_exists():
            return
        if value is not None and progress_var is not None:
            progress_var.set(max(0.0, min(100.0, value)))
        if message is not None and progress_message_var is not None:
            progress_message_var.set(message)

    if root is not None:
        root.after(0, _update)
    else:
        _update()

# --- UI グローバル参照 (initialize_ui 内で設定) ---
root = None
canvas = None
start_angle_var = None
start_gear_var = None
goal_angle_var = None
goal_gear_var = None
pop_var = None
gen_var = None
mut_var = None
cross_var = None
add_var = None
del_var = None
max_gene_var = None
elite_var = None
status_label = None
current_mode = None
run_popup = None
start_pos = None
goal_pos = None
obstacles = []
start_id = None
goal_id = None
start_arrow_id = None
goal_arrow_id = None
obstacle_ids = []
drag_start = None
current_rect = None
vehicle_length_var = None
vehicle_width_var = None
wheel_base_var = None
steering_angles_var = None
forward_steps_var = None
backward_steps_var = None
min_turn_radius_var = None
config_path_var = None
progress_var = None
progress_message_var = None
progress_bar = None
progress_log = None

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

def get_car_corners(state, car_length=None, car_width=None):
    cfg = vehicle_config
    car_length = car_length if car_length is not None else cfg.length
    car_width = car_width if car_width is not None else cfg.width
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
def apply_movement(state, movement_idx, movements=None):
    movements = movements or MOVEMENTS_LIST
    dx, dy, dtheta = movements[movement_idx]
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

def initialize_population(movement_count=None):
    movement_count = movement_count or len(MOVEMENTS_LIST)
    return [
        [random.randint(0, movement_count - 1) for _ in range(random.randint(1, MAX_GENES))]
        for _ in range(POPULATION_SIZE)
    ]

def evaluate_population(population, obstacles, START_STATE, TARGET_STATE, movements=None, executor=None):
    movements = movements or MOVEMENTS_LIST
    fitness_scores = [0.0] * len(population)
    evaluations = [None] * len(population)
    if executor is not None and len(population) > 1:
        tasks = [
            (idx, population[idx], obstacles, START_STATE, TARGET_STATE, movements)
            for idx in range(len(population))
        ]
        for idx, fitness, collision, total_distance in executor.map(_evaluate_genome_task, tasks):
            fitness_scores[idx] = fitness
            evaluations[idx] = {
                "index": idx,
                "fitness": fitness,
                "collision": collision,
                "total_distance": total_distance,
                "genome_length": len(population[idx]),
            }
    else:
        for idx, genome in enumerate(population):
            fitness, collision, total_distance = _evaluate_genome_core(genome, obstacles, START_STATE, TARGET_STATE, movements)
            fitness_scores[idx] = fitness
            evaluations[idx] = {
                "index": idx,
                "fitness": fitness,
                "collision": collision,
                "total_distance": total_distance,
                "genome_length": len(genome),
            }
    return fitness_scores, evaluations

def select_parents(population, fitness_scores):
    top_indices = nsmallest(ELITE_COUNT, range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    return [population[i] for i in top_indices]

def crossover(parent1, parent2, crossover_rate=None):
    rate = CROSSOVER_RATE if crossover_rate is None else crossover_rate
    if random.random() < rate and len(parent1) > 1 and len(parent2) > 1:
        cut = random.randint(1, min(len(parent1), len(parent2)) - 1)
        return parent1[:cut] + parent2[cut:]
    return parent1[:] if random.random() < 0.5 else parent2[:]

def mutate(genome, mutation_rate=None, movement_count=None):
    rate = MUTATION_RATE if mutation_rate is None else mutation_rate
    if genome and random.random() < rate:
        idx = random.randint(0, len(genome) - 1)
        upper = movement_count if movement_count is not None else len(MOVEMENTS_LIST)
        genome[idx] = random.randint(0, upper - 1)
    return genome

def add_gene(genome, addition_rate=None, movement_count=None):
    rate = ADDITION_RATE if addition_rate is None else addition_rate
    if random.random() < rate and len(genome) < MAX_GENES:
        upper = movement_count if movement_count is not None else len(MOVEMENTS_LIST)
        genome.append(random.randint(0, upper - 1))
    return genome

def delete_gene(genome, deletion_rate=None):
    rate = DELETION_RATE if deletion_rate is None else deletion_rate
    if len(genome) > 1 and random.random() < rate:
        del genome[random.randint(0, len(genome) - 1)]
    return genome

def genetic_operator(parent1, parent2, movement_count=None, params=None):
    params = params or {}
    offspring = crossover(parent1, parent2, crossover_rate=params.get("crossover_rate"))
    offspring = mutate(offspring, mutation_rate=params.get("mutation_rate"), movement_count=movement_count)
    offspring = add_gene(offspring, addition_rate=params.get("addition_rate"), movement_count=movement_count)
    offspring = delete_gene(offspring, deletion_rate=params.get("deletion_rate"))
    return offspring

def run_genetic_algorithm(
    obstacles,
    START_STATE,
    TARGET_STATE,
    movements=None,
    enable_parallel=True,
    log_dir=LOG_DIR,
    status_callback=None,
    progress_callback=None,
):
    movements = movements or MOVEMENTS_LIST
    movement_count = len(movements)
    population = initialize_population(movement_count=movement_count)
    best_fitness = float("inf")
    best_genome = None
    start_time = time.time()
    stagnation = 0

    mutation_rate = MUTATION_RATE
    crossover_rate = CROSSOVER_RATE
    addition_rate = ADDITION_RATE
    deletion_rate = DELETION_RATE

    log_writer = None
    log_file = None
    log_path = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"ga_run_{datetime.now():%Y%m%d_%H%M%S}.csv"
        log_file = log_path.open("w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "generation",
            "generation_best",
            "overall_best",
            "mean_fitness",
            "collision_rate",
            "avg_distance",
            "diversity",
            "mutation_rate",
            "addition_rate",
            "deletion_rate",
            "best_length",
        ])

    executor = None
    if enable_parallel and POPULATION_SIZE > 1:
        max_workers = min(os.cpu_count() or 2, POPULATION_SIZE)
        executor = ProcessPoolExecutor(max_workers=max_workers)

    if status_callback:
        status_callback("GA探索を開始しました。")
    if progress_callback:
        progress_callback(0.0)
    try:
        for generation in range(GENERATIONS):
            fitness_scores, eval_details = evaluate_population(
                population,
                obstacles,
                START_STATE,
                TARGET_STATE,
                movements=movements,
                executor=executor,
            )
            if progress_callback:
                progress_callback(100.0 * (generation + 1) / max(1, GENERATIONS))
            gen_best_index = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            gen_best_fitness = fitness_scores[gen_best_index]

            if gen_best_fitness + 1e-6 < best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_index][:]
                stagnation = 0
                if status_callback:
                    status_callback(f"世代 {generation}: 新しい最良個体 (フィットネス {best_fitness:.2f})")
            else:
                stagnation += 1

            if best_fitness < 10:
                print(f"Early stopping at generation {generation} with fitness {best_fitness:.2f}")
                if status_callback:
                    status_callback("目標精度に到達したため早期終了しました。")
                break

            if stagnation and stagnation % 50 == 0:
                mutation_rate = min(mutation_rate * 1.2, 0.6)
                addition_rate = min(addition_rate + 0.05, 0.8)
                deletion_rate = min(deletion_rate + 0.03, 0.7)
                if status_callback:
                    status_callback(f"停滞検出: 操作率を調整します (mutation={mutation_rate:.2f})")
            elif stagnation == 0:
                mutation_rate = max(mutation_rate * 0.9, 0.01)
                addition_rate = max(addition_rate * 0.95, 0.05)
                deletion_rate = max(deletion_rate * 0.95, 0.05)

            params = {
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate,
                "addition_rate": addition_rate,
                "deletion_rate": deletion_rate,
            }

            parents = select_parents(population, fitness_scores)
            if len(parents) < 2:
                parents = [genome[:] for genome in population[:2]]
            new_population = [p[:] for p in parents]
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = random.sample(parents, 2)
                offspring = genetic_operator(parent1, parent2, movement_count=movement_count, params=params)
                new_population.append(offspring)
            population = new_population

            mean_fitness = float(np.mean(fitness_scores)) if fitness_scores else float("inf")
            collision_rate = (
                sum(1 for d in eval_details if d["collision"]) / len(eval_details)
                if eval_details else 0.0
            )
            avg_distance = (
                float(np.mean([d["total_distance"] for d in eval_details]))
                if eval_details else 0.0
            )

            if log_writer:
                diversity = len({tuple(genome) for genome in population}) / len(population)
                log_writer.writerow([
                    generation,
                    round(gen_best_fitness, 4),
                    round(best_fitness, 4),
                    round(mean_fitness, 4),
                    round(collision_rate, 4),
                    round(avg_distance, 4),
                    round(diversity, 4),
                    round(mutation_rate, 4),
                    round(addition_rate, 4),
                    round(deletion_rate, 4),
                    len(best_genome) if best_genome else 0,
                ])

            if generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
                if status_callback:
                    status_callback(f"世代 {generation}: 最良 {best_fitness:.2f} / 平均 {mean_fitness:.2f}")
    finally:
        if executor:
            executor.shutdown()
        if log_file:
            log_file.close()

    end_time = time.time()
    print(f"Genetic algorithm completed in {end_time - start_time:.2f} seconds.")
    if status_callback:
        status_callback(f"GA完了 (所要時間 {end_time - start_time:.1f} 秒)")
    return best_genome, best_fitness, log_path

def compute_trajectory(genome, START_STATE, movements=None):
    if not genome:
        return [START_STATE], []
    state = START_STATE
    trajectory = [state]
    moves = []
    for move in genome:
        state = apply_movement(state, move, movements=movements)
        trajectory.append(state)
        moves.append(move)
    return trajectory, moves


def _evaluate_genome_core(genome, obstacles, start_state, target_state, movements):
    state = start_state
    collision = False
    total_distance = 0.0
    prev = effective_state(start_state)
    for move in genome:
        state = apply_movement(state, move, movements=movements)
        curr = effective_state(state)
        total_distance += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        prev = curr
        if check_car_collision(state, obstacles):
            collision = True
            break
    if collision:
        fitness = 1e6
    else:
        base_fit = calculate_fitness(state, target_state)
        fitness = base_fit + ROUTE_LENGTH_WEIGHT * total_distance
    return fitness, collision, total_distance


def _evaluate_genome_task(args):
    index, genome, obstacles, start_state, target_state, movements = args
    fitness, collision, total_distance = _evaluate_genome_core(genome, obstacles, start_state, target_state, movements)
    return index, fitness, collision, total_distance

def initialize_ui():
    global root, canvas, start_angle_var, start_gear_var, goal_angle_var, goal_gear_var
    global pop_var, gen_var, mut_var, cross_var, add_var, del_var, max_gene_var, elite_var
    global current_mode, status_label
    global vehicle_length_var, vehicle_width_var, wheel_base_var, steering_angles_var
    global forward_steps_var, backward_steps_var, min_turn_radius_var, config_path_var

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
        frame = ttk.Frame(frame_ga)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default_value))
        ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT)
        return var

    pop_var = add_ga_param("個体数", POPULATION_SIZE)
    gen_var = add_ga_param("世代数", GENERATIONS)
    mut_var = add_ga_param("突然変異率", MUTATION_RATE)
    cross_var = add_ga_param("交叉率", CROSSOVER_RATE)
    add_var = add_ga_param("追加率", ADDITION_RATE)
    del_var = add_ga_param("削除率", DELETION_RATE)
    max_gene_var = add_ga_param("最大遺伝子数", MAX_GENES)
    elite_var = add_ga_param("エリート数", ELITE_COUNT)

    frame_vehicle = ttk.LabelFrame(frame_config, text="車両設定", padding=5)
    frame_vehicle.pack(fill=tk.X, pady=5)

    def add_vehicle_param(label_text, variable):
        frame = ttk.Frame(frame_vehicle)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=18).pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=variable, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)

    vehicle_length_var = tk.StringVar(value=f"{vehicle_config.length:.2f}")
    vehicle_width_var = tk.StringVar(value=f"{vehicle_config.width:.2f}")
    wheel_base_var = tk.StringVar(value=f"{vehicle_config.wheel_base:.2f}")
    steering_angles_var = tk.StringVar(value=", ".join(f"{v:.1f}" for v in vehicle_config.steering_angles))
    forward_steps_var = tk.StringVar(value=", ".join(f"{v:.1f}" for v in vehicle_config.forward_steps))
    backward_steps_var = tk.StringVar(value=", ".join(f"{v:.1f}" for v in vehicle_config.backward_steps))
    min_turn_radius_var = tk.StringVar(value=f"{vehicle_config.min_turn_radius:.2f}")
    config_path_var = tk.StringVar(value="(未読込)")

    add_vehicle_param("全長", vehicle_length_var)
    add_vehicle_param("全幅", vehicle_width_var)
    add_vehicle_param("ホイールベース", wheel_base_var)
    add_vehicle_param("旋回角(度,カンマ)", steering_angles_var)
    add_vehicle_param("前進距離", forward_steps_var)
    add_vehicle_param("後退距離", backward_steps_var)
    add_vehicle_param("最小旋回半径", min_turn_radius_var)

    control_frame = ttk.Frame(frame_vehicle)
    control_frame.pack(fill=tk.X, pady=4)
    ttk.Button(control_frame, text="車両設定を適用", command=apply_vehicle_settings_from_ui).pack(side=tk.LEFT, padx=2)
    ttk.Button(control_frame, text="設定ファイル読込", command=load_configuration_dialog).pack(side=tk.LEFT, padx=2)
    ttk.Label(frame_vehicle, textvariable=config_path_var, foreground="gray").pack(anchor="w", pady=(4, 0))

    # --- キャンバスエリア ---
    frame_canvas = ttk.Frame(root)
    frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(frame_canvas, width=ENV_WIDTH, height=ENV_HEIGHT, bg="white")
    canvas.pack()

    # 状態初期化
    obstacles.clear()
    obstacle_ids.clear()
    global start_pos, goal_pos, start_id, goal_id, start_arrow_id, goal_arrow_id
    global drag_start, current_rect
    start_pos = None
    goal_pos = None
    start_id = None
    goal_id = None
    start_arrow_id = None
    goal_arrow_id = None
    drag_start = None
    current_rect = None

    # 現在のモード（"start", "goal", "obstacle"）
    current_mode = tk.StringVar(master=root, value="none")

    # 下部ボタンフレーム
    frame_bottom = ttk.Frame(frame_canvas)
    frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)
    ttk.Button(frame_bottom, text="スタート設定", command=lambda: set_mode("start")).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Button(frame_bottom, text="ゴール設定", command=lambda: set_mode("goal")).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Button(frame_bottom, text="障害物描画", command=lambda: set_mode("obstacle")).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Button(frame_bottom, text="障害物クリア", command=clear_obstacles).pack(side=tk.LEFT, padx=5, pady=5)
    status_label = ttk.Label(frame_bottom, text="Mode: none")
    status_label.pack(side=tk.LEFT, padx=10)
    ttk.Button(frame_bottom, text="シミュレーション開始", command=start_simulation_thread).pack(side=tk.RIGHT, padx=5, pady=5)

    canvas.bind("<Button-1>", canvas_click)
    canvas.bind("<B1-Motion>", canvas_drag)
    canvas.bind("<ButtonRelease-1>", canvas_release)

    # trace_add を利用して設定パネルの変更を反映
    start_angle_var.trace_add("write", lambda *args: update_start_polygon())
    goal_angle_var.trace_add("write", lambda *args: update_goal_polygon())
    start_gear_var.trace_add("write", lambda *args: update_start_polygon())
    goal_gear_var.trace_add("write", lambda *args: update_goal_polygon())

def set_mode(mode):
    if current_mode is None or status_label is None:
        return
    current_mode.set(mode)
    status_label.config(text=f"Mode: {mode}")

def update_start_polygon():
    global start_pos, start_id, start_arrow_id
    if canvas is None or start_angle_var is None or start_gear_var is None:
        return
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
    if canvas is None or goal_angle_var is None or goal_gear_var is None:
        return
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
    if current_mode is None or canvas is None:
        return
    mode = current_mode.get()
    if mode == "start":
        x, y = event.x, event.y
        if start_angle_var is None or start_gear_var is None:
            return
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
        if goal_angle_var is None or goal_gear_var is None:
            return
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
    if current_mode is None or canvas is None:
        return
    if current_mode.get() == "obstacle" and drag_start is not None:
        x0, y0 = drag_start
        x1, y1 = event.x, event.y
        if current_rect is not None:
            canvas.delete(current_rect)
        current_rect = canvas.create_rectangle(x0, y0, x1, y1, outline="purple", dash=(2,2))

def canvas_release(event):
    global current_rect, drag_start, obstacles, obstacle_ids
    if current_mode is None or canvas is None:
        return
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
    if canvas is None:
        return
    obstacles = []
    redraw_obstacles_on_canvas()

# --- シミュレーション実行部 ---
def simulation_thread():
    global start_pos, goal_pos, obstacles
    append_status_message("シミュレーション初期化を開始しました。")
    update_progress(0.0, "初期設定を確認しています...")

    # --- ここから自動初期値セット ---
    if start_pos is None:
        start_pos = (100, 100, 0, "forward")
        root.after(0, update_start_polygon)
        append_status_message("スタート位置をデフォルト値に設定しました。")
    if goal_pos is None:
        goal_pos = (700, 500, 0, "forward")
        root.after(0, update_goal_polygon)
        append_status_message("ゴール位置をデフォルト値に設定しました。")
    if not obstacles:
        obstacles.append((300, 200, 500, 400))
        oid = canvas.create_rectangle(300, 200, 500, 400, outline="purple", dash=(2,2))
        obstacle_ids.append(oid)
        append_status_message("デフォルト障害物を追加しました。")
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
        append_status_message("GAパラメータの読み込みに失敗しました。")
        if root is not None:
            root.after(0, lambda: messagebox.showerror("エラー", f"GAパラメータの入力値に誤りがあります。\n{e}"))
        if run_popup is not None and run_popup.winfo_exists():
            if root is not None:
                root.after(0, run_popup.destroy)
            else:
                run_popup.destroy()
        return

    append_status_message("GAパラメータを適用しました。探索を準備しています。")

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

    update_progress(0.0, "GA探索を開始しています...")
    last_progress = {"value": 0.0}

    def progress_callback(percent: float):
        delta = percent - last_progress["value"]
        last_progress["value"] = percent
        message = None
        if delta >= 5.0 or percent >= 100.0 or percent <= 0.1:
            message = f"GA探索進行中... {percent:.1f}%"
        update_progress(percent, message)

    best_genome, best_fitness, log_path = run_genetic_algorithm(
        sim_obstacles,
        effective_start,
        effective_goal,
        movements=MOVEMENTS_LIST,
        enable_parallel=True,
        status_callback=append_status_message,
        progress_callback=progress_callback,
    )
    append_status_message(f"GA探索が完了しました (最良フィットネス {best_fitness:.2f})")
    update_progress(100.0, "シミュレーション結果を描画しています...")
    print("Best Fitness:", best_fitness)
    if log_path:
        append_status_message(f"世代別ログを {log_path} に保存しました。")
        print(f"世代別ログを {log_path} に保存しました。")
    if not best_genome:
        append_status_message("経路が見つかりませんでした。パラメータを再調整してください。")
        if root is not None:
            root.after(0, lambda: messagebox.showwarning("結果なし", "有効な経路が見つかりませんでした。パラメータを調整してください。"))
        if run_popup is not None and run_popup.winfo_exists():
            if root is not None:
                root.after(0, run_popup.destroy)
            else:
                run_popup.destroy()
        return
    trajectory, moves = compute_trajectory(best_genome, effective_start, movements=MOVEMENTS_LIST)
    root.after(0, lambda: show_animation(trajectory, moves, best_fitness, sim_obstacles))
    root.after(0, lambda: run_popup.destroy() if run_popup is not None and run_popup.winfo_exists() else None)

def start_simulation_thread():
    global run_popup, progress_var, progress_message_var, progress_bar, progress_log
    if root is None:
        return
    if run_popup is not None and run_popup.winfo_exists():
        append_status_message("別のシミュレーションが進行中です。完了をお待ちください。")
        return
    run_popup = tk.Toplevel(root)
    run_popup.title("実行中")
    run_popup.geometry("340x220")
    ttk.Label(run_popup, text="シミュレーション実行中...").pack(padx=12, pady=(12, 6), anchor="w")
    progress_message_var = tk.StringVar(value="初期化中...")
    ttk.Label(run_popup, textvariable=progress_message_var, wraplength=300, justify="left").pack(fill=tk.X, padx=12)
    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = ttk.Progressbar(run_popup, mode="determinate", maximum=100, variable=progress_var)
    progress_bar.pack(fill=tk.X, padx=12, pady=6)
    progress_log = tk.Text(run_popup, height=6, width=40, state="disabled", wrap="word")
    progress_log.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
    t = threading.Thread(target=simulation_thread)
    t.daemon = True
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

def run_app():
    initialize_ui()
    root.mainloop()


if __name__ == "__main__":
    run_app()
