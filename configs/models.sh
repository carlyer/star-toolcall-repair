# Override these with downloaded model paths before GPU training.
export STUDENT_MODEL="${STUDENT_MODEL:-models/Qwen3-0.6B}"
export TEACHER_MODEL="${TEACHER_MODEL:-models/Qwen3-8B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
