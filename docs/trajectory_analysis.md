# Trajectory Dump & Analysis Tools

학습된 V-Max TD3 trajectory 모델의 actor 출력, baseline(path_target), 실제 ego 궤적, LQR 추적 오차를 정량적으로 분석하기 위한 스크립트.

mp4 영상으로는 보이지 않는 수치적 진단(예: actor 출력 saturation, LQR 추적 오차, lane 추종 여부)을 가능하게 한다.

설계는 nuPlan `SimulationLog` / `SerializationCallback` 구조를 참고했다.

---

## 디렉터리 구조

```
vmax/scripts/dump_trajectory/
├── dump_trajectory.py    # 모델 + 시나리오 → 한 시나리오 = 한 pickle
└── analyze_dump.py       # pickle 여러 개 → LQR 추적 오차 + PNG 시각화
```

---

## 1. dump_trajectory.py — per-step 데이터 캡처

학습된 모델을 특정 시나리오에서 step-by-step으로 굴리면서 매 step의 actor 입력/출력, baseline, ego 상태를 pickle로 저장한다.

### 실행

```bash
cd /home/jovyan/workspace/V-Max

/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/dump_trajectory/dump_trajectory.py \
  --path_model "ritp_phase1_v6_ttc" \
  --path_dataset /home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  --scenario_indexes 0 1 2 \
  --out_dir /home/jovyan/workspace/dump_results
```

### 인수

| 인수 | 설명 |
|------|------|
| `--path_model` | `runs/` 아래 run 폴더명 (e.g. `ritp_phase1_v6_ttc`) |
| `--path_dataset` | 평가용 TFRecord 경로 |
| `--scenario_indexes` | dump할 시나리오 index 목록 (e.g. `0 1 2`) |
| `--out_dir` | pickle 저장 디렉터리 |
| `--src_dir` | run 폴더 위치 (기본: `runs`) |
| `--max_num_objects` | 시뮬레이션 최대 객체 수 (기본 64) |
| `--seed` | 난수 시드 (기본 0) |

### 출력 스키마 (한 시나리오 = 한 pickle)

```python
{
    "scenario_idx": int,
    "dt": 0.1,
    "num_waypoints": 16,
    "num_path_points": 10,
    "roadgraph": {                          # 시나리오 시작 시 1번
        "xy":     ndarray (N_rg, 2),         # world frame
        "dir_xy": ndarray (N_rg, 2),
        "types":  ndarray (N_rg,),
        "ids":    ndarray (N_rg,),
        "valid":  ndarray (N_rg,) bool,
    },
    "steps": [
        {
            "step": int,
            "timestamp_s": float,
            "ego_xy":         ndarray (2,),      # world frame [m]
            "ego_yaw":        float,             # [rad]
            "ego_vel_xy":     ndarray (2,),      # world frame [m/s]
            "ego_speed":      float,             # [m/s]
            "obs":            ndarray (obs_size,),
            "actor_raw":      ndarray (32,),     # actor tanh 출력
            "predicted_traj": ndarray (16, 2),   # ego-rel [m] (decode_trajectory 결과)
            "path_target":    ndarray (10, 2),   # ego-rel [m] (역정규화 적용됨)
            "baseline_interp":ndarray (16, 2),   # 10pt → 16pt 보간 [m]
            "residual":       ndarray (16, 2),   # predicted_traj − baseline_interp [m]
        }, ...
    ],
}
```

### 단위 주의

- `obs[:, -20:]`에 저장된 path_target은 **정규화 (÷ 50 m)** 되어 있다.
- dump 스크립트는 이를 **`_PATH_NORM_METERS=50` 곱해 미터 단위로 저장**한다.
- 이렇게 해야 `decode_trajectory()` 내부의 `_interp_path_target()`이 사용하는 baseline 단위와 일치한다.

---

## 2. analyze_dump.py — LQR 추적 오차 + 시각화

dump pickle을 읽어서 두 가지 분석을 수행한다:

### (A) LQR 추적 오차

```
error[t] = || ego_xy[t+1] − world_frame(predicted_traj[t][0]) ||
```

각 step에서 actor가 예측한 "0.1초 뒤 ego 위치"와 **실제 0.1초 뒤 ego 위치**의 거리. LQR + bicycle dynamics가 actor 의도대로 차량을 움직였는지 측정.

정상치 예시 (ego 2.67 m/s):
- 0.1초 이동 거리: ~0.27 m
- 추적 오차가 이보다 훨씬 크면 → actor 의도 ≠ 실제 거동

### (B) 시각화 (PNG)

매 N step마다 다음을 한 figure에 겹쳐 그린다:

| 색상/마커 | 의미 |
|-----------|------|
| 연한 회색 점 | roadgraph (lane geometry) |
| 빨간 선 | ego 실제 trail (full episode) |
| 빨간 ● / 어두운 빨간 ■ | ego 시작 / 끝 |
| **초록 점선** | path_target baseline (10 points, ego-rel → world) |
| **시안 점선** | predicted_traj (16 waypoints) |
| 주황 × | predicted next ego (predicted_traj 첫 waypoint) |

### 실행

```bash
cd /home/jovyan/workspace/V-Max

/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/dump_trajectory/analyze_dump.py \
  --dump_dir /home/jovyan/workspace/dump_results/ritp_phase1_v6_ttc \
  --overlay_every 2
```

### 인수

| 인수 | 설명 |
|------|------|
| `--dump_dir` | `scenario_*.pkl` 들이 있는 디렉터리 |
| `--overlay_every` | 몇 step마다 baseline/predicted_traj 오버레이 (기본 2) |
| `--out_subdir` | dump_dir 하위 출력 폴더명 (기본 `analysis`) |

### 출력

```
<dump_dir>/analysis/
├── scenario_0000.png
├── scenario_0001.png
├── ...
└── lqr_tracking_summary.txt
```

---

## 3. 진단 워크플로

학습된 모델의 lane 추종 / actor saturation / LQR 동작을 진단할 때:

### Step 1. mp4로 정성적 관찰
`evaluate_traj/evaluate.py --render true --show_trajectory true` 로 mp4를 보고 어느 시나리오에서 문제가 두드러지는지 확인.

### Step 2. 문제 시나리오 dump
```bash
python vmax/scripts/dump_trajectory/dump_trajectory.py \
  --path_model "<run_name>" \
  --path_dataset <tfrecord_path> \
  --scenario_indexes <문제_시나리오> \
  --out_dir /home/jovyan/workspace/dump_results
```

### Step 3. 분석 실행
```bash
python vmax/scripts/dump_trajectory/analyze_dump.py \
  --dump_dir /home/jovyan/workspace/dump_results/<run_name>
```

### Step 4. 수치 진단 포인트

pickle을 직접 열거나 PNG를 보면서 확인:

| 진단 항목 | 확인 방법 | 정상 / 비정상 |
|----------|----------|---------------|
| **Actor saturation** | `actor_raw` 의 절댓값 분포 | ±1에 박혀있으면 saturation (scale 부족) |
| **Residual 크기** | `residual` 의 평균/최대 | scale 한계(`_TRAJ_SCALE_X/Y`)에 가까우면 actor가 더 흔들고 싶어함 |
| **Baseline 정확성** | PNG의 초록 baseline vs lane 중심 | path_target이 lane center에서 벗어나면 route extractor 문제 |
| **Actor lane 추종** | PNG의 시안 predicted_traj vs lane | predicted_traj가 lane을 벗어나면 actor 학습 부족 |
| **LQR 추적** | `lqr_tracking_summary.txt` | 0.1초 이동 거리(`ego_speed × 0.1`)보다 크면 LQR/dynamics 문제 |
| **속도 패턴** | `ego_speed` 시계열 | 일정하게 0 또는 saturation이면 reward/scale 문제 |

---

## 4. 실측 예시 (v6 모델, train_boston_test scenario 0~2)

| 시나리오 | steps | LQR 추적 오차 평균 [m] | residual_y max [m] | residual_x max [m] |
|---------|------:|----------------------:|-------------------:|-------------------:|
| 0 | 11 | 4.14 | 1.00 | 2.00 |
| 1 | 8 | 2.32 | 1.00 | 2.00 |
| 2 | 11 | 1.66 | 1.00 | 2.00 |

**관찰**:
- residual_x/y 가 정확히 `_TRAJ_SCALE_X=2.0`, `_TRAJ_SCALE_Y=1.0` 한계에 박혀있음 → **actor가 양축 모두 saturate**
- LQR 추적 오차가 0.27 m (예상 이동 거리) 대비 **6~20배 큼**
- 결론: actor가 물리적으로 도달 불가능한 trajectory를 출력 중. residual scale 또는 baseline 자체에 대한 검토 필요.
