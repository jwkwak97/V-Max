# V-Max 시뮬레이션 파이프라인

## 1. SimulatorState

**SimulatorState**는 Waymax 시뮬레이터의 현재 상태를 담은 데이터 컨테이너입니다.  
시뮬레이션의 현재 스냅샷을 나타내며, 모든 데이터는 **전역 좌표계(global coordinates)** 기준입니다.

### 필드 구조

```
SimulatorState                                     shape
├── sim_trajectory  (Trajectory)                   (batch, num_objects=64, num_timesteps=91)
│   │  AI 에이전트가 실제로 주행한 궤적.
│   │  시뮬레이션 시작 시 log_trajectory로 초기화되고,
│   │  매 스텝 env.step(action) 이후 AI의 새 위치로 덮어씌워진다.
│   │  미래 스텝은 valid=False로 표시됨.
│   ├── x, y, z         float32                    # 위치 (전역 좌표)
│   ├── vel_x, vel_y    float32                    # 속도 벡터
│   ├── yaw             float32                    # 방향각 (반시계, rad)
│   ├── length, width, height  float32             # 차량 크기
│   └── valid           bool                       # 유효 여부
│
├── log_trajectory  (Trajectory)                   (batch, num_objects=64, num_timesteps=91)
│   │  nuPlan DB에서 읽어온 실제 사람이 운전한 궤적 (expert trajectory).
│   │  시뮬레이션 내내 변하지 않는 고정값.
│   │  reward 계산 시 log_div metric에서 sim_trajectory와 비교하는 기준값으로 사용.
│   └── (sim_trajectory와 동일한 필드 구조)
│
├── log_traffic_light  (TrafficLights)             (batch, num_traffic_lights, num_timesteps=91)
│   ├── x, y, z         float32                    # 신호등 위치
│   ├── state           int32                      # 신호 상태 (RED/YELLOW/GREEN 등)
│   ├── lane_ids        int32                      # 제어 차선 ID
│   └── valid           bool
│
├── object_metadata  (ObjectMetadata)              (batch, num_objects=64)
│   └── is_valid        bool                       # 해당 객체가 유효한지 여부
│
├── timestep            int                        # 현재 스텝 인덱스 (0~90, 0.1초 단위)
│
├── sdc_paths  (Paths)                             (batch, num_paths, num_points_per_path)
│   └── SDC가 주행 가능한 경로 후보들
│
└── roadgraph_points  (RoadgraphPoints)            (batch, num_points=20000)
    ├── x, y, z         float32                    # 맵 포인트 위치
    ├── dir_x, dir_y, dir_z  float32               # 방향 벡터 (단위 벡터)
    ├── types           int32                      # 도로 요소 타입 (차선/경계/정지선 등)
    ├── ids             int32                      # 맵 피처 고유 ID
    └── valid           bool
```

### RoadgraphPoints types 값

| 값 | 의미 |
|----|------|
| 1 | LaneCenter_FreewayLane |
| 2 | LaneCenter_SurfaceStreet |
| 3 | LaneCenter_BikeLane |
| 6 | RoadLine_BrokenSingleWhite |
| 7 | RoadLine_SolidSingleWhite |
| 13 | RoadEdge_Boundary |
| 15 | RoadEdge_Median |
| 17 | StopSign |
| 18 | Crosswalk |
| 19 | SpeedBump |

---

## 2. Observation 추출

SimulatorState에서 신경망 입력을 위한 flat 벡터를 추출하는 과정입니다.  
모든 피처는 **ego(SDC) 기준 상대 좌표계**로 변환됩니다.

### 추출되는 피처 구성

```
Observation (flat JAX 배열)
├── sdc_traj_features      (1, obs_past_steps, object_features)    # ego 차량 과거 궤적
├── other_traj_features    (num_objects-1, obs_past_steps, object_features)  # 주변 차량 궤적
├── roadgraph_features     (roadgraph_top_k=1000, rg_features)     # 맵 포인트 top-1000
├── traffic_light_features (num_closest_tl=16, tl_features)        # 가장 가까운 신호등 16개
└── path_target_features   (num_paths, path_features)              # 주행 경로 목표
```

### object_features 구성 (차량 피처)

```python
# 차량 1개당 추출되는 피처 (ego-relative 좌표)
[x, y,          # 위치 (m)
 vel_x, vel_y,  # 속도 (m/s)
 yaw,           # 방향각 (rad)
 length, width, # 차량 크기 (m)
 valid]         # 유효 마스크
```

### roadgraph_features 구성 (맵 피처)

```python
# 맵 포인트 1개당 추출되는 피처 (ego-relative 좌표, top-1000 선택)
[x, y,          # 위치 (m)
 dir_x, dir_y,  # 방향 벡터
 types,         # 도로 요소 타입 (int)
 valid]         # 유효 마스크
```

### traffic_light_features 구성

```python
# 신호등 1개당 추출되는 피처 (ego-relative 좌표, 가장 가까운 16개)
[x, y,          # 위치 (m)
 state,         # 신호 상태 (int: RED/YELLOW/GREEN 등)
 valid]         # 유효 마스크
```

---

## 3. 학습 루프

```
TFRecord (nuPlan DB)
    ↓
SimulatorState 초기화       ← 시나리오 시작 상태
    ↓
Observation 추출
  ├── ego 궤적 (1, steps, features)
  ├── 주변 차량 (63, steps, features)
  ├── 맵 포인트 top-1000 (1000, features)     ← jax.lax.top_k로 선택
  ├── 신호등 top-16 (16, features)
  └── 경로 목표
    ↓ flatten → (batch, total_obs_dim)
Wayformer Encoder
  ├── 각 피처 MLP 임베딩 → (batch, N, dk=64)
  └── Self-Attention → (batch, dk)
    ↓
TD3 Actor → action [accel_norm, kappa_norm]
    ↓
JAX LQR (trajectory 모드일 경우)
    ↓
env.step(action)
    ↓
SimulatorState 업데이트     ← 다음 timestep 상태
    ↓
reward 계산 (여러 metric의 가중합)
  ├── log_div    : sim_trajectory vs log_trajectory 이탈 정도 (expert 비교)
  ├── overlap    : 다른 차량과 충돌 여부
  ├── offroad    : 도로 이탈 여부
  ├── off_route  : 계획 경로 이탈 여부
  ├── below_ttc  : 전방 충돌 시간 < 1.5초
  ├── red_light  : 신호 위반 여부
  ├── comfort    : 급가속/급감속 페널티
  ├── overspeed  : 제한속도 초과 여부
  └── progression: 경로 전진 여부
    ↓
반복 (total_timesteps까지)
```

---

## 4. 맵 피처 처리 흐름

### 전체 흐름

```
SimulatorState.roadgraph_points (전체 맵 포인트)
    ↓ jax.lax.top_k (ego 기준 거리 상위 1000개 선택)
RoadgraphPoints (xy, dir_xy, types, valid)
    ↓ normalize + stack
(batch, 1000, num_features) JAX 배열
    ↓ MLP 임베딩
(batch, 1000, 64)
    ↓ Self-Attention
(batch, num_latents, 64)
    ↓ agent / traffic_light 피처와 fusion
(batch, dk) → TD3 Actor/Critic 입력
```

### 단계별 코드

#### Step 1. SimulatorState → Observation 변환
`vmax/simulator/overrides/datatypes/observation.py`

ego 기준 좌표계로 변환하고 top-k 맵 포인트를 필터링합니다.

```python
sdc_observation_from_state(state, roadgraph_top_k=1000)
```

#### Step 2. 맵 포인트 필터링 (거리 기준 상위 k개)
`vmax/simulator/overrides/datatypes/roadgraph.py`

```python
def filter_topk_roadgraph_points(roadgraph, reference_points, topk):
    distances = jnp.linalg.norm(reference_points - roadgraph.xy, axis=-1)
    _, top_idx = jax.lax.top_k(-distances, topk)  # 가장 가까운 k개
    return jax.tree.map(lambda x: x[top_idx], roadgraph)
```

#### Step 3. 피처 추출 → flat JAX 배열
`vmax/simulator/features/extractor/vec_extractor.py`

xy, dir_xy, types 등을 하나의 배열로 합칩니다.

```python
# Shape: (batch, roadgraph_top_k, num_features)
roadgraph_features.stack_fields()
```

#### Step 4. Wayformer 인코더 처리
`vmax/agents/networks/encoders/wayformer.py`

```python
# MLP 임베딩
rg_encoding = build_mlp_embedding(rg_features, dk=64)
# → (batch, roadgraph_top_k, 64)

# Self-Attention
output_rg = self_attn(rg_encoding, rg_valid_mask)
# → (batch, num_latents, 64)
```

---

## 5. JAX 컴파일

맵 피처 처리에 사용되는 `jax.lax.top_k`, `jnp.linalg.norm` 등 모든 연산이 JAX로 처리됩니다.  
따라서 **첫 실행 시 XLA 컴파일**이 필요합니다 (약 10분 소요).

| 컴파일 | 소요 시간 | 발생 시점 |
|--------|-----------|-----------|
| TF PTX | ~30분 | 첫 실행 시 1회 |
| JAX XLA | ~10분 | 첫 실행 또는 코드 변경 후 |

---

## 6. 데이터 로딩과 JAX의 역할 구분

| 단계 | 처리 주체 | JAX 영향 |
|------|-----------|----------|
| TFRecord에서 맵 데이터 읽기 | TensorFlow | 없음 |
| 맵 피처를 Observation으로 변환 | JAX (Waymax) | XLA 컴파일 대상 |
| Wayformer 인코더로 맵 처리 | JAX | XLA 컴파일 대상 |

`NUPLAN_MAPS_ROOT`는 **ScenarioMax 변환 시에만** 필요하며,  
학습 시에는 맵 데이터가 이미 TFRecord 안에 포함되어 있습니다.
