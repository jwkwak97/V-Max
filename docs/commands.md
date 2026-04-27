# V-Max 실행 명령어 모음

## 환경 정보

| 항목 | 값 |
|------|-----|
| conda 환경 | `vmax` |
| Python 경로 | `/home/jovyan/.conda/envs/vmax/bin/python` |
| 작업 디렉토리 | `/home/jovyan/workspace/V-Max` |
| 데이터 경로 | `/home/jovyan/workspace/vmax_data/` |

---

## 1. 데이터 변환 (nuPlan → TFRecord)

> ScenarioMax 사용. 자세한 내용은 [ScenarioMax 서버 가이드](https://github.com/jwkwak97/ScenarioMax/blob/main/docs/SERVER_SETUP.md) 참조.

> **주의**: 마운트된 볼륨 이름은 서버 환경마다 다를 수 있습니다 (예: `aitc-plan-vmax-datavol-1`, `aitc-plan-team-1` 등).  
> 아래 `NUPLAN_DATASETS_ROOT`를 본인 환경에 맞게 수정하세요.

```bash
# 본인 환경에 맞게 수정
export NUPLAN_DATASETS_ROOT=/home/jovyan/<마운트된-볼륨-이름>

# 테스트 변환 (10개 시나리오, ~5분)
LD_PRELOAD=/home/jovyan/.conda/envs/vmax/lib/libstdc++.so.6 \
NUPLAN_MAPS_ROOT=${NUPLAN_DATASETS_ROOT}/nuplan-maps-v1.0 \
NUPLAN_DATA_ROOT=${NUPLAN_DATASETS_ROOT}/data \
/home/jovyan/.conda/envs/scenariomax/bin/scenariomax-convert \
  --nuplan_src ${NUPLAN_DATASETS_ROOT}/data/cache/train_boston \
  --dst /home/jovyan/workspace/vmax_data/scenariomax_test \
  --target_format tfexample \
  --num_workers 4 \
  --num_files 10

# 전체 변환 (train_boston 전체)
LD_PRELOAD=/home/jovyan/.conda/envs/vmax/lib/libstdc++.so.6 \
NUPLAN_MAPS_ROOT=${NUPLAN_DATASETS_ROOT}/nuplan-maps-v1.0 \
NUPLAN_DATA_ROOT=${NUPLAN_DATASETS_ROOT}/data \
/home/jovyan/.conda/envs/scenariomax/bin/scenariomax-convert \
  --nuplan_src ${NUPLAN_DATASETS_ROOT}/data/cache/train_boston \
  --dst /home/jovyan/workspace/vmax_data/nuplan_tfrecord/train_boston \
  --target_format tfexample \
  --num_workers 8
```

출력 파일: `<dst>/training.tfrecord`

---

## 2. 학습

```bash
cd /home/jovyan/workspace/V-Max

# 동작 확인용 (소규모 데이터, ~6분)
# scenariomax_test: --num_files 10 으로 변환한 소규모 TFRecord
/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  "network/encoder=wayformer" \
  path_dataset=/home/jovyan/workspace/vmax_data/scenariomax_test/training.tfrecord \
  use_wandb=false \
  total_timesteps=50000 \
  log_freq=1

# 표준 학습 (TD3 + Trajectory + LQR + WayformerEncoder)
/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  "network/encoder=wayformer" \
  path_dataset=/home/jovyan/workspace/vmax_data/scenariomax_test/training.tfrecord \
  use_wandb=false \
  total_timesteps=20_000_000
```

> **주의**: 첫 실행 시 B200 GPU용 XLA JIT 컴파일로 10~30분 소요. 이후 실행은 캐시 사용.

### 장기 학습 (SSH 끊겨도 유지)

```bash
tmux new -s vmax_train

# tmux 세션 안에서 학습 실행
cd /home/jovyan/workspace/V-Max
/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/training/train.py \
  algorithm=td3_trajectory \
  "network/encoder=wayformer" \
  path_dataset=/home/jovyan/workspace/vmax_data/scenariomax_test/training.tfrecord \
  use_wandb=false \
  total_timesteps=20_000_000

# 세션 분리 (학습 유지): Ctrl+B → D
# 재접속: tmux attach -t vmax_train
```

---

## 3. 학습 결과 확인

### 저장 구조

```
runs/
└── TD3_VEC_WAYFORMER_날짜_시간/
    ├── train.log              ← 학습 지표 텍스트 로그
    ├── events.out.tfevents.*  ← TensorBoard 데이터
    └── model/
        ├── model_<step>.pkl   ← 중간 체크포인트
        └── model_final.pkl    ← 최종 모델
```

### 최신 run 확인

```bash
ls -t /home/jovyan/workspace/V-Max/runs/ | head -5
```

### 로그 직접 확인

```bash
tail -50 /home/jovyan/workspace/V-Max/runs/<run_name>/train.log
```

### TensorBoard

```bash
# 서버에서 실행
/home/jovyan/.conda/envs/vmax/bin/tensorboard \
  --logdir /home/jovyan/workspace/V-Max/runs/ \
  --port 6007

# 로컬 PC에서 SSH 터널 (별도 터미널)
ssh -p 31842 -L 6007:localhost:6007 jhlee2@ssh-nipagpu.kakaocloud.com
# 브라우저: http://localhost:6007
```

자세한 내용은 [TensorBoard 모니터링 가이드](tensorboard_monitoring.md) 참조.

---

## 4. 평가

### 수치 평가 (Accuracy, V-Max Score)

```bash
cd /home/jovyan/workspace/V-Max

/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/evaluate_traj/evaluate.py \
  --sdc_actor ai \
  --path_model "<run_name>" \
  --path_dataset /home/jovyan/workspace/vmax_data/scenariomax_test/training.tfrecord \
  --batch_size 4 \
  --eval_name /home/jovyan/workspace/eval_results
```

출력 예시:
```
Accuracy: 0.3000
V-Max Score: 0.0895
```

결과 파일: `/home/jovyan/workspace/eval_results/<run_name>/model_final/evaluation_results.txt`

### 주행 영상 생성 (mp4)

```bash
# PATH 설정 필수 (ffmpeg 경로)
PATH="/home/jovyan/.conda/envs/vmax/bin:$PATH" \
/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/evaluate_traj/evaluate.py \
  --sdc_actor ai \
  --path_model "<run_name>" \
  --path_dataset /home/jovyan/workspace/vmax_data/scenariomax_test/training.tfrecord \
  --render true \
  --batch_size 1 \
  --eval_name /home/jovyan/workspace/eval_results

# trajectory 오버레이 포함 (cyan 16 waypoints)
PATH="/home/jovyan/.conda/envs/vmax/bin:$PATH" \
/home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/evaluate_traj/evaluate.py \
  --sdc_actor ai \
  --path_model "<run_name>" \
  --path_dataset /home/jovyan/workspace/vmax_data/scenariomax_test/training.tfrecord \
  --render true \
  --show_trajectory true \
  --batch_size 1 \
  --eval_name /home/jovyan/workspace/eval_results
```

mp4 저장 위치:
```
/home/jovyan/workspace/eval_results/<run_name>/model_final/mp4/
├── eval_0.mp4
├── eval_1.mp4
└── ...
```

> `--batch_size 1` 필수 (render 시)  
> `PATH` 접두사 없으면 ffmpeg 못 찾아 "Error writing video" 발생

---

## 5. GPU 상태 확인

```bash
# GPU 점유 프로세스 확인
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# GPU 전체 상태
nvidia-smi
```

---

## 6. 핵심 지표 기준

| 지표 | 의미 | 좋은 값 |
|------|------|---------|
| `vmax_aggregate_score` | V-Max 종합 점수 | 높을수록 좋음 (0~1) |
| `nuplan_aggregate_score` | nuPlan 기준 점수 | 높을수록 좋음 (0~1) |
| `overlap` | 충돌 비율 | 낮을수록 좋음 |
| `progress_ratio` | 목적지 도달률 | 높을수록 좋음 |
| `run_red_light` | 신호 위반 비율 | 낮을수록 좋음 |
| `train/policy_loss` | Actor 손실 | 초반 0 → 이후 음수 방향 |
| `train/value_loss` | Critic 손실 | 점진적 감소 |
