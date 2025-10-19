"""Streamlit UI prototype for the GA parking simulator."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st

import main


def parse_float_list(text: str, fallback: Optional[Tuple[float, ...]] = None) -> list[float]:
    if not text:
        return list(fallback or [])
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values or list(fallback or [])


def load_json_payload(uploaded_file) -> dict:
    if uploaded_file is None:
        return {}
    data = uploaded_file.read()
    return json.loads(data.decode("utf-8"))


def prepare_configs(payload: dict) -> tuple[main.VehicleConfig, main.EnvironmentConfig, main.GAConfig]:
    vehicle_conf = main.VehicleConfig.from_dict(payload.get("vehicle", {}))
    environment_conf = main.EnvironmentConfig.from_dict(payload.get("environment", {}))
    ga_conf = main.GAConfig.from_dict(payload.get("ga", {}))
    return vehicle_conf, environment_conf, ga_conf


def sidebar_controls(vehicle_conf: main.VehicleConfig, environment_conf: main.EnvironmentConfig, ga_conf: main.GAConfig):
    st.sidebar.header("車両パラメータ")
    length = st.sidebar.number_input("全長", value=float(vehicle_conf.length), min_value=10.0, max_value=120.0, step=1.0)
    width = st.sidebar.number_input("全幅", value=float(vehicle_conf.width), min_value=5.0, max_value=60.0, step=1.0)
    wheel_base = st.sidebar.number_input("ホイールベース", value=float(vehicle_conf.wheel_base), min_value=10.0, max_value=100.0, step=1.0)
    steering_angles = parse_float_list(
        st.sidebar.text_input("旋回角 (deg, カンマ区切り)", ", ".join(f"{v:.1f}" for v in vehicle_conf.steering_angles)),
        tuple(vehicle_conf.steering_angles),
    )
    forward_steps = parse_float_list(
        st.sidebar.text_input("前進距離", ", ".join(f"{v:.1f}" for v in vehicle_conf.forward_steps)),
        tuple(vehicle_conf.forward_steps),
    )
    backward_steps = parse_float_list(
        st.sidebar.text_input("後退距離", ", ".join(f"{v:.1f}" for v in vehicle_conf.backward_steps)),
        tuple(vehicle_conf.backward_steps),
    )
    min_turn_radius = st.sidebar.number_input("最小旋回半径", value=float(vehicle_conf.min_turn_radius), min_value=20.0, max_value=400.0, step=5.0)

    st.sidebar.header("GA パラメータ")
    population_size = st.sidebar.number_input("個体数", value=int(ga_conf.population_size), min_value=10, max_value=200, step=5)
    generations = st.sidebar.number_input("世代数", value=int(ga_conf.generations), min_value=100, max_value=20000, step=100)
    mutation_rate = st.sidebar.number_input("突然変異率", value=float(ga_conf.mutation_rate), min_value=0.0, max_value=1.0, step=0.01)
    crossover_rate = st.sidebar.number_input("交叉率", value=float(ga_conf.crossover_rate), min_value=0.0, max_value=1.0, step=0.01)
    addition_rate = st.sidebar.number_input("追加率", value=float(ga_conf.addition_rate), min_value=0.0, max_value=1.0, step=0.01)
    deletion_rate = st.sidebar.number_input("削除率", value=float(ga_conf.deletion_rate), min_value=0.0, max_value=1.0, step=0.01)
    max_genes = st.sidebar.number_input("最大遺伝子数", value=int(ga_conf.max_genes), min_value=10, max_value=500, step=5)
    elite_count = st.sidebar.number_input("エリート数", value=int(ga_conf.elite_count), min_value=1, max_value=20, step=1)

    updated_vehicle = main.VehicleConfig(
        length=length,
        width=width,
        wheel_base=wheel_base,
        steering_angles=steering_angles,
        forward_steps=forward_steps,
        backward_steps=backward_steps,
        min_turn_radius=min_turn_radius,
    )
    updated_ga = main.GAConfig(
        population_size=int(population_size),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        crossover_rate=float(crossover_rate),
        addition_rate=float(addition_rate),
        deletion_rate=float(deletion_rate),
        max_genes=int(max_genes),
        elite_count=int(elite_count),
    )

    st.sidebar.header("環境設定")
    width_env = st.sidebar.number_input("フィールド幅", value=float(environment_conf.width), min_value=100.0, max_value=2000.0, step=10.0)
    height_env = st.sidebar.number_input("フィールド高さ", value=float(environment_conf.height), min_value=100.0, max_value=2000.0, step=10.0)
    obstacles_text = st.sidebar.text_area(
        "障害物 (JSON 配列)",
        json.dumps(
            [
                {"x_min": obs[0], "y_min": obs[1], "x_max": obs[2], "y_max": obs[3]}
                for obs in environment_conf.obstacles
            ],
            ensure_ascii=False,
            indent=2,
        ),
    )

    try:
        obstacles_payload = json.loads(obstacles_text) if obstacles_text else []
        obstacles = [
            (float(item["x_min"]), float(item["y_min"]), float(item["x_max"]), float(item["y_max"]))
            for item in obstacles_payload
        ]
    except Exception:
        st.sidebar.warning("障害物の JSON が正しくありません。初期値を使用します。")
        obstacles = environment_conf.obstacles

    start_pose = environment_conf.start or (100.0, 100.0, 0.0, "forward")
    goal_pose = environment_conf.goal or (width_env - 100.0, height_env - 100.0, 0.0, "forward")

    updated_environment = main.EnvironmentConfig(
        width=width_env,
        height=height_env,
        obstacles=obstacles,
        start=start_pose,
        goal=goal_pose,
    )

    return updated_vehicle, updated_environment, updated_ga


def run_simulation(vehicle_conf: main.VehicleConfig, environment_conf: main.EnvironmentConfig, ga_conf: main.GAConfig):
    movements = main.generate_movements_from_vehicle(vehicle_conf)
    main.apply_ga_config(ga_conf, update_ui=False)

    start_state = environment_conf.start or (100.0, 100.0, 0.0, "forward")
    goal_state = environment_conf.goal or (environment_conf.width - 100.0, environment_conf.height - 100.0, 0.0, "forward")

    boundaries = [
        (0, 0, environment_conf.width, 5),
        (0, environment_conf.height - 5, environment_conf.width, environment_conf.height),
        (0, 0, 5, environment_conf.height),
        (environment_conf.width - 5, 0, environment_conf.width, environment_conf.height),
    ]

    obstacles = list(environment_conf.obstacles) + boundaries

    best_genome, best_fitness, _ = main.run_genetic_algorithm(
        obstacles,
        main.effective_state(start_state),
        main.effective_state(goal_state),
        movements=movements,
        enable_parallel=False,
        log_dir=None,
    )

    trajectory, _ = main.compute_trajectory(best_genome, main.effective_state(start_state), movements=movements)
    coords = np.array([[state[0], state[1], state[2]] for state in trajectory])
    return best_fitness, coords


def main_app():
    st.title("GA 自動駐車シミュレーション (Web プロトタイプ)")
    st.markdown("外部設定ファイルと GUI 以外の操作フローを検証するための試験的 UI です。")

    uploaded = st.sidebar.file_uploader("設定ファイル (JSON)", type="json")
    payload = load_json_payload(uploaded) if uploaded else {}
    vehicle_conf, environment_conf, ga_conf = prepare_configs(payload)
    vehicle_conf, environment_conf, ga_conf = sidebar_controls(vehicle_conf, environment_conf, ga_conf)

    if st.button("シミュレーション実行"):
        with st.spinner("GA を実行中..."):
            best_fitness, coords = run_simulation(vehicle_conf, environment_conf, ga_conf)
        st.success(f"最良フィットネス: {best_fitness:.2f}")
        st.write("最適経路の各ステップ (X, Y, θ):")
        st.dataframe(
            {
                "x": np.round(coords[:, 0], 2),
                "y": np.round(coords[:, 1], 2),
                "theta": np.round(coords[:, 2], 2),
            }
        )
        path_chart = st.line_chart(coords[:, :2], height=300)
        st.caption("※ プロットは X-Y のみを表示しています")

    st.sidebar.markdown("---")
    st.sidebar.markdown("[設定ファイルサンプル](https://gist.github.com/) を参考に JSON を作成してください。")


if __name__ == "__main__":
    main_app()
