#!/bin/bash
# vmax conda 환경 자동 설치 스크립트
# 사용법: bash /home/jovyan/workspace/vmax_setup.sh
set -e

VMAX_ENV=/home/jovyan/.conda/envs/vmax
PYTHON=$VMAX_ENV/bin/python
PIP=$VMAX_ENV/bin/pip

echo "================================================================"
echo "  vmax conda 환경 설치 시작"
echo "================================================================"

echo ""
echo "[1/6] conda 환경 생성 (Python 3.10)..."
/opt/conda/bin/conda create -n vmax python=3.10 -y

echo ""
echo "[2/6] JAX 0.5.x (CUDA 12) 설치..."
$PIP install "jax[cuda12]<0.6"

echo ""
echo "[3/6] V-Max 의존 패키지 설치..."
$PIP install \
  "git+https://github.com/waymo-research/waymax.git@main" \
  "flax==0.8.3" \
  distrax einops tf_keras \
  gymnasium hydra-core \
  "matplotlib<3.10" \
  "tensorboard<2.19" \
  "tensorboard-plugin-profile<2.19" \
  tensorboardX \
  psutil pandas seaborn \
  ipython ipykernel \
  tensorflow wandb

echo ""
echo "[3-1/6] JAX 버전 고정 (flax가 올릴 수 있으므로 재다운그레이드)..."
$PIP install "jax[cuda12]<0.6" --force-reinstall

echo ""
echo "[3-2/6] ffmpeg 설치 (mp4 렌더링용)..."
$PIP install "imageio[ffmpeg]"

echo ""
echo "[3-3/6] setuptools 버전 고정 (pkg_resources 누락 방지)..."
# setuptools 70+ 에서 pkg_resources 디렉토리가 생성되지 않는 경우가 있어
# tensorboard 등 pkg_resources에 의존하는 패키지가 실패함
$PIP install "setuptools==69.5.1"

echo ""
echo "[5/6] V-Max editable install..."
$PIP install -e /home/jovyan/workspace/V-Max --no-deps

echo ""
echo "[6/6] distrax 패치 적용..."
$PYTHON - << 'PYEOF'
import re

path = "/home/jovyan/.conda/envs/vmax/lib/python3.10/site-packages/distrax/_src/utils/transformations.py"
with open(path, "r") as f:
    src = f.read()

if "_jax_core_Var" in src:
    print("  distrax: 이미 패치됨, skip")
else:
    shim = (
        "\n# Compatibility shim for jax >= 0.5 (jax.core.Var/Literal moved to jax._src.core)\n"
        "import jax._src.core as _jax_src_core\n"
        "_jax_core_Var = _jax_src_core.Var\n"
        "_jax_core_Literal = _jax_src_core.Literal\n"
    )
    src = src.replace("import jax.numpy as jnp", "import jax.numpy as jnp" + shim)
    src = src.replace("jax.core.Var", "_jax_core_Var")
    src = src.replace("jax.core.Literal", "_jax_core_Literal")
    with open(path, "w") as f:
        f.write(src)
    print("  distrax: 패치 완료")
PYEOF

echo ""
echo "[7/6] flax tracer 패치 적용..."
$PYTHON - << 'PYEOF'
path = "/home/jovyan/.conda/envs/vmax/lib/python3.10/site-packages/flax/core/tracers.py"
with open(path, "r") as f:
    src = f.read()

if "hasattr(trace, 'main')" in src:
    print("  flax: 이미 패치됨, skip")
else:
    old_current = (
        "def current_trace():\n"
        "  \"\"\"Returns the innermost Jax tracer.\"\"\"\n"
        "  return jax.core.find_top_trace(())"
    )
    new_current = (
        "def current_trace():\n"
        "  \"\"\"Returns the innermost Jax tracer.\"\"\"\n"
        "  trace = jax.core.find_top_trace(())\n"
        "  # JAX 0.5+: find_top_trace returns the trace object; extract .main\n"
        "  if hasattr(trace, 'main'):\n"
        "    return trace.main\n"
        "  return trace"
    )
    old_level = (
        "def trace_level(main):\n"
        "  \"\"\"Returns the level of the trace of -infinity if it is None.\"\"\"\n"
        "  if main:\n"
        "    return main.level\n"
        "  return float('-inf')"
    )
    new_level = (
        "def trace_level(main):\n"
        "  \"\"\"Returns the level of the trace of -infinity if it is None.\"\"\"\n"
        "  if main is None:\n"
        "    return float('-inf')\n"
        "  if hasattr(main, 'level'):\n"
        "    return main.level\n"
        "  # JAX 0.5+: TraceTag has no .level\n"
        "  return float('-inf')"
    )
    src = src.replace(old_current, new_current)
    src = src.replace(old_level, new_level)
    with open(path, "w") as f:
        f.write(src)
    print("  flax: 패치 완료")
PYEOF

echo ""
echo "================================================================"
echo "  설치 검증"
echo "================================================================"
$PYTHON - << 'PYEOF'
import jax, flax, distrax
import jax.numpy as jnp
from flax import linen as nn

print(f"JAX    : {jax.__version__}")
print(f"flax   : {flax.__version__}")
print(f"distrax: {distrax.__version__}")
print(f"devices: {jax.devices()}")

x = jnp.ones((3, 3))
assert jnp.sum(x).item() == 9.0

model = nn.Dense(4)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((2, 3)))
out = model.apply(params, jnp.ones((2, 3)))
assert out.shape == (2, 4)

print("검증 완료! 학습 실행 명령어:")
print("  cd /home/jovyan/workspace/V-Max")
print("  /home/jovyan/.conda/envs/vmax/bin/python vmax/scripts/training/train.py \\")
print("    algorithm=td3 \\")
print("    path_dataset=/home/jovyan/workspace/vmax_data/nuplan_tfrecord/test/train_boston_test.tfrecord \\")
print("    use_wandb=false")
PYEOF

echo ""
echo "================================================================"
echo "  설치 완료!"
echo "================================================================"
