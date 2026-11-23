#!/usr/bin/env bash
# Полный эксперимент: 5 моделей x 3 seed + замеры + устойчивость + Grad-CAM + таблицы.
# Запускать из корня репозитория: bash scripts/run_all.sh /путь/к/DeepWeeds
set -e
DATA=${1:?укажите папку с данными}
EPOCHS=${EPOCHS:-15}
mkdir -p results
python src/loader_throughput.py --data_dir "$DATA"
python src/benchmark.py
for M in resnet50 resnet18 mobilenet_v3_small mobilenet_v3_large efficientnet_b0; do
  for S in 0 1 2; do
    python src/train.py --data_dir "$DATA" --model $M --seed $S --epochs $EPOCHS
  done
  python src/robustness.py --data_dir "$DATA" --run_dir results/${M}_s0
  python src/gradcam.py --data_dir "$DATA" --run_dir results/${M}_s0
done
python src/aggregate.py
echo "Готово. Таблицы и рисунки в results/"
