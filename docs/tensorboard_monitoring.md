# Monitoring Training with TensorBoard

## 1. Start TensorBoard (on server)

```bash
/home/jovyan/.conda/envs/vmax/bin/tensorboard \
  --logdir /home/jovyan/workspace/V-Max/runs/ \
  --port 6007
```

Can be started before, during, or after training — TensorBoard scans the `runs/` folder automatically.

## 2. Open in Browser (on local PC)

Open a separate terminal and create an SSH tunnel:

```bash
ssh -p 31842 -L 6007:localhost:6007 jhlee2@ssh-nipagpu.kakaocloud.com
```

Then open in browser: `http://localhost:6007`

## 3. Key Metrics

| Metric | Good Pattern |
|--------|-------------|
| `train/value_loss` | Gradually decreasing |
| `train/policy_loss` | ~0 at start → decreasing (negative direction) |
| `eval/vmax_aggregate_score` | Gradually increasing (0~1) |
| `eval/nuplan_aggregate_score` | Gradually increasing (0~1) |
| `eval/overlap` | Gradually decreasing |
| `eval/progress_ratio` | Gradually increasing |

## 4. Compare Multiple Runs

To compare V-Max runs with RITP baseline simultaneously:

```bash
/home/jovyan/.conda/envs/vmax/bin/tensorboard \
  --logdir_spec=vmax:/home/jovyan/workspace/V-Max/runs/,ritp:/home/jovyan/workspace/nuplan_zigned/experiments/ritp_planner/training_ritp_planner_experiment/train_motionformer \
  --port 6007
```

## 5. Run Folder Structure

```
runs/
└── TD3_VEC_WAYFORMER_날짜_시간/
    ├── train.log          ← text log (tail -f for quick check)
    ├── events.out.tfevents.*  ← TensorBoard data
    └── model/
        ├── model_10240.pkl
        └── model_final.pkl
```

Quick check without TensorBoard:
```bash
tail -50 /home/jovyan/workspace/V-Max/runs/<run_name>/train.log
```
