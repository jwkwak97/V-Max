# V-Max 아키텍처 설계 문서 (CLAUDE.md)

## 0. 서버 사양

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA B200 × 8 (compute capability 10.0) |
| GPU 메모리 | 183 GB / GPU |
| RAM | 2.2 TB |
| OS | Linux |

**주의사항**:
- GPU는 여러 사용자가 공유. 타인이 GPU 점유 중이면 OOM으로 `Killed` 발생
- TF CUDA PTX 컴파일: B200용 바이너리 없어서 첫 실행 시 JIT 컴파일 (~30분)
- JAX XLA 컴파일: 코드 변경 후 첫 실행 시 추가 ~10분
- OOM 방지: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` 환경변수로 JAX 메모리 제한 가능
- GPU 점유 확인: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`

**파일 권한 주의사항**:
- `/home/jovyan/workspace/eval_results` 디렉토리는 jhlee2 소유 → jovyan 사용자로 쓸 수 없음
- JupyterHub 터미널(jovyan)에서 evaluate.py 실행 시 `PermissionError` 발생
- **해결책**: `--eval_name /home/jovyan/workspace/eval_results` 사용 (workspace는 모든 사용자 쓰기 가능)
  ```bash
  --eval_name /home/jovyan/workspace/eval_results
  ```
- **해결책 2**: jhlee2 SSH 세션에서 실행 (기존 `/home/jovyan/workspace/eval_results` 그대로 사용 가능)
  ```bash
  ssh -p 31842 jhlee2@ssh-nipagpu.kakaocloud.com
  ```

---

## 0-1. JAX B200 GPU 호환성 현황 및 모델 지원 범위 (JAX 0.5.3 + CUDA 12 환경)

> 실측 기준 (2026-04-24). JAX가 공식 B200(cc10.0) 지원 버전을 출시하면 재검증 필요.

### ✅ GPU에서 정상 동작하는 연산

| 카테고리 | 연산 |
|---|---|
| 기본 행렬 | `matmul`, `dot`, `einsum`, `tensordot`, `outer`, `norm` |
| FFT (순방향) | `fft`, `rfft`, `fft2` |
| Convolution | `lax.conv_general_dilated` (CNN 레이어) |
| 고유값 (비대칭) | `linalg.eig`, `linalg.eigvals` |
| 난수 | `random.normal`, `random.uniform` |
| 인덱싱 | `take`, `scatter` (at.set) |
| 변환 | `jit`, `vmap`, `grad`, `pmap` |

### ❌ GPU에서 실패하는 연산 및 원인

| 원인 | 실패 연산 |
|---|---|
| **cuSolverDn 핸들 생성 실패** | `linalg.{solve, inv, pinv, qr, eigh, eigvalsh, det, slogdet, lstsq, cond, matrix_rank}` |
| **cuSolverDn 핸들 생성 실패** | `scipy.linalg.{solve, lu, expm}` |
| **cuSolver internal error** | `linalg.cholesky`, `scipy.linalg.cho_factor` |
| **BLAS 미지원** | `fft.ifft`, `fft.irfft`, `fft.ifft2`, `scipy.linalg.solve_triangular` |
| **CUDA MLIR 변환 규칙 없음** | `scipy.linalg.{schur, hessenberg}` |

**근본 원인**: JAX 0.5.3 bundled CUDA 12의 `cuSolverDn` 라이브러리가 B200(compute capability 10.0)에서 핸들 초기화에 실패. JAX 0.6.x는 `cusolver_getrf_ffi` 미구현으로 실패하여 현재 사용 불가.

### V-Max 학습에서의 영향

| 기능 | 상태 | 이유 |
|---|---|---|
| JIT + vmap 시뮬레이션 | ✅ 정상 | cuSolverDn 불필요 |
| TD3 학습 (`algorithm=td3`) | ✅ 정상 | MLP만 사용 |
| TD3 + LQR (`algorithm=td3_trajectory`) | ✅ 정상 | JAX LQR은 closed-form 스칼라 연산 |
| LQR 역전파 (actor loss) | ✅ 정상 | `jax.grad` 완전 통과 |
| `jnp.linalg.solve/inv/svd` 직접 사용 | ❌ 실패 | cuSolverDn 불가 |

**우회 방법**: `jnp.linalg.solve` 등 실패 연산이 꼭 필요한 경우 CPU로 강제 실행:
```python
with jax.default_device(jax.devices('cpu')[0]):
    x = jnp.linalg.solve(A, b)
```

### JAX + B200에서 모델 아키텍처별 GPU 실행 가능 여부

| 모델 / 아키텍처 | GPU 실행 | 핵심 연산 | 비고 |
|---|---|---|---|
| **MLP** | ✅ | matmul + activation | 제한 없음 |
| **CNN** | ✅ | conv_general_dilated | 제한 없음 |
| **Transformer / ViT** | ✅ | matmul + softmax | self-attention 전부 가능 |
| **LSTM / GRU** | ✅ | element-wise + matmul | 순환 레이어 모두 가능 |
| **Wayformer** | ✅ | cross-attention (matmul) | V-Max 기본 인코더 |
| **Gaussian Process (GP)** | ❌ | `K⁻¹ = linalg.solve` | cuSolverDn 실패 |
| **Kalman Filter** | ❌ | `P update = solve/inv` | cuSolverDn 실패 |
| **PCA / ICA** | ❌ | SVD / eigh | cuSolverDn 실패 |
| **Spectral 방법** | ❌ | 대칭 고유값분해 (eigh) | cuSolverDn 실패 |

**결론**: matmul·activation·softmax 기반 딥러닝 모델(MLP/CNN/Transformer/LSTM/Wayformer)은 B200 + JAX 0.5.3에서 모두 GPU로 정상 학습 가능. 행렬 분해(solve/inv/svd/eigh)에 의존하는 통계적 모델(GP, Kalman, PCA 등)은 CPU 우회 필요.

### V-Max 전체 기능 동작 검증 결과 (2026-04-24, GPU 실측)

| 기능 | 상태 | 소요 시간 | 비고 |
|---|---|---|---|
| `algorithm=td3` | ✅ | ~47s | MLP 인코더 |
| `algorithm=td3_trajectory` | ✅ | ~47s | LQR 포함 |
| `algorithm=sac` | ✅ | ~44s | |
| `algorithm=ppo` | ✅ | ~70s | |
| `algorithm=bc` | ✅ | ~40s | |
| `network/encoder=wayformer` | ✅ | ~327s | 첫 실행 XLA 컴파일 포함 |
| `evaluate.py` | ✅ | ~34s | 20 에피소드, V-Max Score 출력 |

**evaluate.py 사용법** (`--path_model`은 `runs/` 아래 run 이름만):
```bash
cd /home/jovyan/workspace/V-Max
/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/evaluate/evaluate.py \
  --sdc_actor ai \
  --path_model "BC_VEC_24-04_04:13:51" \
  --path_dataset /home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  --batch_size 2
```

---

## 1. 프로젝트 목표

| 단계 | 내용 |
|------|------|
| 최종 목표 | Apollo 자율주행 플랫폼에 배포 가능한 ONNX 정책 모델 |
| 학습 프레임워크 | V-Max (JAX + Waymax 기반 RL) |
| 정책 출력 | Trajectory (궤적) — 직접 제어값(acceleration, steering)이 아님 |
| 제어 변환 | JAX LQR: trajectory → [acceleration, steering_curvature] |
| 시뮬레이터 | Waymax `InvertibleBicycleModel(normalize_actions=True)` |

---

## 2. 전체 파이프라인

```
nuPlan DB
   ↓ (ScenarioMax, 2-step 변환)
TFRecord
   ↓
V-Max 환경 (Waymax)
   ↓
Observation → WayformerEncoder → TD3 Actor → Trajectory(16×2)
                                                    ↓
                                              JAX LQR
                                                    ↓
                                         [accel_norm, kappa_norm] ∈ [-1,1]
                                                    ↓
                                   InvertibleBicycleModel → 다음 상태
                                                    ↓
                                              Reward 계산
                                                    ↓
                                         TD3 Critic 학습
   ↓ (학습 완료)
ONNX Export (policy only: obs → trajectory)
   ↓
Apollo: trajectory → 별도 LQR → 차량 제어
```

---

## 3. 데이터 파이프라인

### Step 1: nuPlan DB → Pickle
- **환경**: `nuplan_ritp` conda 환경
- **필요 이유**: ScenarioMax가 nuPlan SDK에 의존하지만, nuPlan SDK와 TensorFlow가 같은 환경에 공존 불가
- **주의사항**: `LD_PRELOAD=/home/jovyan/.conda/envs/nuplan_ritp/lib/libstdc++.so.6` 필요 (CXXABI 버전 충돌)

```bash
LD_PRELOAD=/home/jovyan/.conda/envs/nuplan_ritp/lib/libstdc++.so.6 \
  /home/jovyan/.conda/envs/nuplan_ritp/bin/python convert_nuplan_to_pickle.py \
  --data_path /path/to/nuplan/data \
  --maps_path /home/jovyan/aitc-plan-team-1/nuplan-maps-v1.0
```

### Step 2: Pickle → TFRecord
- **환경**: `vmax` conda 환경 (TensorFlow 포함)
- **필요 이유**: TFRecord 생성에 TF가 필요한데 nuplan_ritp 환경에는 TF 없음

```bash
/home/jovyan/.conda/envs/vmax/bin/python convert_pickle_to_tfrecord.py
```

### 변환된 데이터 위치
```
/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord
```

---

## 4. 원본 V-Max에서 추가된 코드

### 4-1. WandB 연동

**파일**: `vmax/scripts/training/train_utils.py`, `vmax/scripts/training/train.py`, `vmax/config/base_config.yaml`

**추가 이유**: TensorBoard만으로는 학습 중 실시간 모니터링이 불편. WandB를 통해 원격에서 reward, loss 추이를 확인하기 위함.

**추가 내용**:

```python
# train_utils.py에 추가
def setup_wandb(config: dict, run_path: str) -> None:
    wandb.init(
        project=config.get("wandb_project", "vmax"),
        name=config.get("name_run") or run_path.split("/")[-1],
        config=config,
        dir=run_path,
    )

# log_metrics()에 추가
if _WANDB_AVAILABLE and wandb.run is not None:
    wandb.log(wandb_metrics, step=num_steps)
```

```yaml
# base_config.yaml에 추가
use_wandb: false
wandb_project: vmax
```

**사용법**: `python train.py ... use_wandb=true`

---

### 4-2. TD3 알고리즘 추가

**파일**: `vmax/agents/learning/reinforcement/td3/` (신규 디렉터리)  
`vmax/config/algorithm/td3.yaml` (신규)  
`vmax/agents/learning/__init__.py` (수정)

**추가 이유**: 기존 V-Max는 SAC(Soft Actor-Critic)만 지원. RITP 시스템에서 영감을 받은 결정론적(deterministic) 정책이 필요했고, TD3가 결정론적 actor + twin critics 구조를 제공함.

**TD3 vs SAC 차이점**:
| 항목 | SAC | TD3 |
|------|-----|-----|
| 정책 | Stochastic (확률적) | Deterministic (결정론적) |
| Actor 출력 | 확률분포 파라미터 | 직접 action 값 |
| Exploration | 엔트로피 보너스 | 탐색 노이즈 직접 추가 |
| Target smoothing | 없음 | 있음 (noise_std, noise_clip) |
| Actor 업데이트 | 매 step | policy_freq step마다 (기본 2) |

**핵심 파일 구조**:
```
vmax/agents/learning/reinforcement/td3/
├── __init__.py          # initialize, make_sgd_step, make_inference_fn 노출
├── td3_factory.py       # 네트워크 생성, 추론 함수, loss 함수
└── td3_trainer.py       # 학습 루프 (replay buffer, pmap 분산학습)
```

**evaluate.py에도 TD3 지원 추가** (`vmax/scripts/evaluate/utils.py`):
- `get_algorithm_modules()`: TD3 case 추가
- `load_model()`: "td3" 조건 추가
- `action_distribution` KeyError 방지: `.get()` 사용

---

### 4-3. JAX LQR 모듈 (신규)

**파일**: `vmax/agents/networks/lqr/jax_lqr.py` (신규)  
`vmax/agents/networks/lqr/__init__.py` (신규)  
`vmax/agents/networks/__init__.py` (수정 — `jax_lqr` export 추가)

**추가 이유**:  
RITP(Reference-Integrated Trajectory Planning)는 trajectory를 출력하고 LQR로 차량을 제어하는 구조. 이를 V-Max에 이식하기 위해 RITP의 `BatchLQRTracker`(numpy 기반)를 순수 JAX로 포팅.

**핵심 설계 원칙**:
- **CPU 병목 없음**: 모든 연산이 `jnp` 연산 → GPU에서 실행, `jax.jit` 범위 내 유지
- **자동 미분 가능**: 순수 JAX 행렬 연산이므로 `jax.grad`를 통해 actor loss backprop 가능
- **Waymax 호환**: 출력이 `InvertibleBicycleModel(normalize_actions=True)` 형식에 맞음

**LQR 입출력**:
```
입력: trajectory (batch, 16, 2)  — ego-relative (x_forward, y_left) [m]
      ego_speed  (batch,)        — 현재 종방향 속도 [m/s]

출력: action (batch, 2)          — [accel_norm, kappa_norm] ∈ [-1, 1]
      accel_norm  = acceleration / 6.0   (max_accel = 6.0 m/s²)
      kappa_norm  = curvature   / 0.3    (max_kappa = 0.3 rad/m)
```

**종방향 LQR (속도 추적)**:
```
상태: v (현재 속도)
입력: a (가속도)
동역학: v_next = v + a * dt
비용: Q_lon*(v_next - v_ref)^2 + R_lon*a^2
닫힌형 해: a = -Q_lon*dt / (Q_lon*dt^2 + R_lon) * (v - v_ref)
파라미터: Q_lon=10, R_lon=1, dt=0.1s
```

**횡방향 LQR (경로 추적)**:
```
상태: [lateral_error, heading_error]
입력: dκ = κ - κ_ref  (기준 곡률 대비 편차)
동역학:
  lat_err_next  = lat_err + v * heading_err * dt
  head_err_next = heading_err + dκ * v * dt
비용: Q_head*head_err_next^2 + R_lat*dκ^2
닫힌형 해: dκ = -Q_head*v*dt / (Q_head*(v*dt)^2 + R_lat) * heading_err
           + k_lat/v * lateral_err  (Stanley 게인)
파라미터: Q_head=10, R_lat=1, k_lat=0.5
```

**RITP BatchLQRTracker와의 차이**:
| 항목 | RITP (원본) | V-Max JAX LQR (이식) |
|------|------------|----------------------|
| 구현 | numpy | jax.numpy |
| 실행 환경 | CPU | GPU (jit) |
| 미분 | 불가 | jax.grad 가능 |
| 제어 출력 | [accel, steering_rate] | [accel_norm, kappa_norm] |
| 동역학 모델 | KinematicBicycle (RITP) | InvertibleBicycleModel (Waymax) |
| 지평선 | 10 step (multi-step) | 1 step (closed-form) |

---

### 4-4. TD3 Trajectory 모드 (신규)

**파일**: `vmax/agents/learning/reinforcement/td3/td3_factory.py` (수정)  
`vmax/agents/learning/reinforcement/td3/td3_trainer.py` (수정)  
`vmax/config/algorithm/td3_trajectory.yaml` (신규)

**추가 이유**:  
Apollo 배포 시 정책이 trajectory를 출력하고, 차량 내부에서 LQR로 제어값을 계산하는 구조가 필요. 기존 TD3는 직접 [accel, steering]을 출력하므로 이 구조를 지원하도록 확장.

**`trajectory_size` 파라미터**:
- `trajectory_size=0` (기본값): 기존 TD3와 동일, actor가 2-dim action 직접 출력
- `trajectory_size=32` (16 waypoints × 2): actor가 32-dim trajectory 출력, LQR 변환

**수정된 함수들**:

```python
# td3_factory.py

# 1. initialize() — trajectory_size 파라미터 추가
def initialize(..., trajectory_size: int = 0):
    # actor는 trajectory_size차원 출력, critic은 2-dim action 평가

# 2. make_networks() — actor/critic 출력 크기 분리
def make_networks(..., trajectory_size: int = 0):
    actor_out_size = trajectory_size if trajectory_size > 0 else action_size
    policy_network = make_policy_network(..., actor_out_size, ...)  # 32-dim
    value_network  = make_value_network(..., action_size, ...)       # 2-dim 유지

# 3. make_inference_fn() — LQR 변환 포함
def make_inference_fn(td3_network, trajectory_size=0):
    def policy(obs, key=None):
        output = actor.apply(params, obs)           # (batch, 32)
        if use_lqr:
            traj = output.reshape(-1, 16, 2)        # (batch, 16, 2)
            ego_speed = ||traj[:,0,:]|| / 0.1       # 첫 waypoint로 속도 추정
            action = jax_lqr(traj, ego_speed)       # (batch, 2)
        return action, {}  # 항상 2-dim control 반환 → env.step에 전달

# 4. _make_loss_fn() — policy gradient가 LQR 통과
def _actor_to_action(params, obs):
    traj = actor.apply(params, obs)                 # (batch, 32)
    control = jax_lqr(traj.reshape(-1,16,2), speed) # (batch, 2)
    return control                                   # gradient 흐름 유지

# policy loss: obs → actor → LQR → control → Q (end-to-end backprop)
def compute_policy_loss(policy_params, value_params, transitions):
    action = _actor_to_action(policy_params, obs)
    q = value_network.apply(value_params, obs, action)
    return -mean(q)
```

**Replay Buffer 저장 구조**:
```
trajectory 모드에서도 replay buffer에는 2-dim control action 저장
(actor → LQR → control → env.step → reward → buffer에 control 저장)
→ critic은 standard TD3와 동일하게 Q(obs, control) 학습
```

---

## 5. Waymax 동역학 모델

```python
from waymax import dynamics
dynamics.InvertibleBicycleModel(normalize_actions=True)
```

| 파라미터 | 범위 | 설명 |
|---------|------|------|
| `action[0]` (acceleration) | [-1, 1] → ×6.0 = [-6, 6] m/s² | 종방향 가속도 |
| `action[1]` (steering_curvature) | [-1, 1] → ×0.3 = [-0.3, 0.3] rad/m | 1/회전반경 |
| `dt` | 0.1 s | 시뮬레이션 타임스텝 |

**dynamics 수식**:
```
new_x   = x + vel_x*t + 0.5*accel*cos(yaw)*t²
new_y   = y + vel_y*t + 0.5*accel*sin(yaw)*t²
new_yaw = yaw + kappa*(speed*t + 0.5*accel*t²)
new_vel = speed + accel*t
```

---

## 6. 학습 설정

### 학습 명령어

```bash
cd /home/jovyan/workspace/V-Max
conda activate vmax_test

# 표준 TD3
python vmax/scripts/training/train.py \
  algorithm=td3 \
  path_dataset=/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  use_wandb=true

# TD3 + Trajectory + LQR (신규)
python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  path_dataset=/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  use_wandb=true

# TD3 + Trajectory + LQR + WayformerEncoder (full 아키텍처)
python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  network=wayformer \
  path_dataset=/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  use_wandb=true
```

### 주요 하이퍼파라미터 (td3_trajectory.yaml)

```yaml
trajectory_size: 32      # 16 waypoints × 2 (x, y)
learning_rate: 3e-4
discount: 0.99
tau: 0.005               # soft target update
policy_freq: 2           # actor는 critic의 절반 빈도로 업데이트
noise_std: 0.2           # target policy smoothing 노이즈
noise_clip: 0.5
batch_size: 64
buffer_size: 100_000
learning_start: 100      # 이 step 이후부터 학습 시작
```

### 학습 모니터링 지표

| 지표 | 정상 패턴 | 설명 |
|------|-----------|------|
| `value_loss` | 점진적 감소 | Critic이 Q값을 정확히 추정하는지 |
| `policy_loss` | 처음 N step 0 → 이후 음수 방향 감소 | N = policy_freq × learning_start |
| `ep_rew_mean` | 점진적 증가 | 에피소드 평균 보상 |
| `ep_len_mean` | 증가 (max=scenario_length) | 에피소드 길이 (조기 종료 없을수록 길어짐) |

---

## 7. 평가 및 시각화

```bash
# 학습된 모델로 mp4 생성
PATH="/home/jovyan/.conda/envs/vmax_test/bin:$PATH" \
python vmax/scripts/evaluate/evaluate.py \
  --algo td3_trajectory \
  --model_path runs/TD3_VEC_.../model/model_final.pkl \
  --data_path /home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  --output_dir ./eval_videos
```

**시각화 색상 코드**: Waymax 기본 렌더러 기준
- 파란색: Ego 차량 (SDC)
- 회색: 다른 에이전트 (logged trajectory)
- 빨간색: 충돌 또는 오프로드

---

## 8. 전체 실행 방법 (Step by Step)

### Step 0: 환경 활성화

```bash
conda activate vmax_test
cd /home/jovyan/workspace/V-Max
```

### Step 1: 데이터 변환 (이미 완료된 경우 생략)

```bash
# nuPlan DB → Pickle (nuplan_ritp 환경에서)
conda activate nuplan_ritp
LD_PRELOAD=/home/jovyan/.conda/envs/nuplan_ritp/lib/libstdc++.so.6 \
  python convert_to_pickle.py \
  --data_path /path/to/nuplan/train \
  --maps_path /home/jovyan/aitc-plan-team-1/nuplan-maps-v1.0 \
  --output_dir /tmp/nuplan_pickle

# Pickle → TFRecord (vmax_test 환경에서)
conda activate vmax_test
python convert_to_tfrecord.py \
  --input_dir /tmp/nuplan_pickle \
  --output_path /home/jovyan/workspace/vmax_data/nuplan_tfrecord/train.tfrecord
```

### Step 2: 학습 실행

**A. 표준 TD3 (baseline 검증용)**
```bash
python vmax/scripts/training/train.py \
  algorithm=td3 \
  path_dataset=/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  use_wandb=true \
  total_timesteps=1_000_000
```

**B. TD3 + Trajectory + LQR (신규 아키텍처)**
```bash
python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  path_dataset=/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  use_wandb=true \
  total_timesteps=1_000_000
```

**C. TD3 + Trajectory + LQR + WayformerEncoder (full 아키텍처)**
```bash
python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  network=wayformer \
  path_dataset=/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  use_wandb=true \
  total_timesteps=20_000_000
```

**주요 override 옵션**:
```bash
# 디버그 (빠른 확인)
python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  path_dataset=... \
  total_timesteps=10_000 \
  num_envs=4 \
  log_freq=1

# 체크포인트 저장 빈도 조정
  save_freq=50     # 50 iter마다 저장

# WandB 프로젝트/실험명 지정
  use_wandb=true \
  wandb_project=vmax \
  name_run=td3_traj_test1 \
  name_exp=my_experiment
```

### Step 3: WandB 모니터링

```bash
# 학습 시작 후 브라우저에서 확인
# https://wandb.ai/jewoo2963/vmax
# 계정: jewoo2963@naver.com
```

모니터링 지표:
- `train/value_loss` → 감소해야 정상
- `train/policy_loss` → 음수 방향으로 감소해야 정상 (처음 몇백 step은 0)
- `ep_rew_mean` → 증가해야 정상
- `ep_len_mean` → 증가해야 정상 (충돌·이탈이 줄어들수록)

### Step 4: 학습 결과 시각화 (mp4)

```bash
# ffmpeg PATH 설정 필수
PATH="/home/jovyan/.conda/envs/vmax_test/bin:$PATH" \
python vmax/scripts/evaluate/evaluate.py \
  --algo td3_trajectory \
  --model_path runs/<실험명>/model/model_final.pkl \
  --data_path /home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \
  --output_dir ./eval_videos \
  --num_scenarios 20
```

mp4 파일이 `./eval_videos/` 에 생성됨.  
- 파란색 차량 = Ego (SDC), 회색 = 다른 에이전트

### Step 5: 체크포인트 경로 확인

```bash
# 학습 완료 후 모델 파일 위치
ls runs/<실험명>/model/
# model_<step>.pkl  (중간 저장)
# model_final.pkl   (최종)
```

---

## 9. ONNX Export 계획 (미구현)

```python
# 내보낼 부분: policy network만 (LQR은 Apollo 내부에서 실행)
# obs → trajectory (16×2)

import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
import tensorflow as tf

def policy_fn(obs):
    return actor_network.apply(params, obs)  # (batch, 32)

# JAX → TF → ONNX 변환
tf_fn = jax2tf.convert(policy_fn)
```

Apollo 배포 구조:
```
[Camera/LiDAR] → Feature Extraction → obs
obs → ONNX 정책 (trajectory) → Apollo LQR → 차량 제어
```

---

## 9. 환경 설정

| conda 환경 | 용도 |
|------------|------|
| `nuplan_ritp` | nuPlan DB 파싱, pickle 변환 |
| `vmax_test` | TFRecord 변환, V-Max 학습, 평가 |

**ffmpeg 경로 이슈**: 평가 시 `PATH="/home/jovyan/.conda/envs/vmax_test/bin:$PATH"` 필요

**데이터 경로**:
```
/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/  # 변환된 TFRecord
/home/jovyan/aitc-plan-team-1/nuplan-maps-v1.0/         # nuPlan 지도
```

---

## 10. 명령어 관리 규칙

프로젝트에서 사용하는 모든 실행 명령어는 아래 파일에 기록·관리한다.

```
/home/jovyan/workspace/Commands/Command.md
```

- 새로운 명령어(학습, 변환, 평가, Docker 등)를 추가하거나 변경할 때 반드시 이 파일을 업데이트한다.
- 명령어는 카테고리별로 구분하여 기록한다.

---

## 11. 파일 변경 이력 요약

| 파일 | 변경 종류 | 이유 |
|------|-----------|------|
| `vmax/config/base_config.yaml` | 수정 | WandB 설정 (`use_wandb`, `wandb_project`) 추가 |
| `vmax/scripts/training/train_utils.py` | 수정 | `setup_wandb()`, `log_metrics()` WandB 연동 추가 |
| `vmax/scripts/training/train.py` | 수정 | WandB init 호출 추가 |
| `vmax/scripts/evaluate/utils.py` | 수정 | TD3 알고리즘 지원 추가 (3곳) |
| `vmax/agents/learning/reinforcement/td3/` | 신규 | TD3 알고리즘 구현 |
| `vmax/config/algorithm/td3.yaml` | 신규 | TD3 기본 설정 |
| `vmax/config/algorithm/td3_trajectory.yaml` | 신규 | TD3 + Trajectory 모드 설정 |
| `vmax/agents/networks/lqr/jax_lqr.py` | 신규 | JAX LQR 컨트롤러 |
| `vmax/agents/networks/lqr/__init__.py` | 신규 | LQR 모듈 init |
| `vmax/agents/networks/__init__.py` | 수정 | `jax_lqr` export 추가 |
| `vmax/agents/learning/reinforcement/td3/td3_factory.py` | 수정 | `trajectory_size` 파라미터, LQR 통합 |
| `vmax/agents/learning/reinforcement/td3/td3_trainer.py` | 수정 | `trajectory_size` 파라미터 전달 |
