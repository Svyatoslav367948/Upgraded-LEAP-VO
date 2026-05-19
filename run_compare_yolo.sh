#!/bin/bash
# run_compare_yolo.sh — сравнение LeapVO baseline vs LeapVO+YOLOv11 на DynaKITTI и Shibuya
# Запуск в контейнере leapvo: bash /workspace/leapvo/run_compare_yolo.sh

set -e
cd /workspace/leapvo

SEQS_KITTI="00_1 01_0 02_1 02_2 03_0 04_0 07_1 08_0 10_1"   # короткие последовательности для быстрого теста
DATA_KITTI=/workspace/data/DynaKITTI

SEQS_SHIB="RoadCrossing03 Standing01"
DATA_SHIB=/workspace/data/shibuya

RES_BASE=/workspace/leapvo/results/dynakitti
RES_YOLO=/workspace/leapvo/results/dynakitti_yolo
SHIB_BASE=/workspace/leapvo/results/shibuya
SHIB_YOLO=/workspace/leapvo/results/shibuya_yolo

LOG=/workspace/leapvo/results/comparison.txt

# Устанавливаем ultralytics если нет
python -c "import ultralytics" 2>/dev/null || pip install ultralytics -q

mkdir -p $RES_BASE $RES_YOLO $SHIB_BASE $SHIB_YOLO

echo "============================================" | tee $LOG
echo "  LeapVO: Baseline vs YOLO Comparison"       | tee -a $LOG
echo "  $(date)"                                    | tee -a $LOG
echo "============================================" | tee -a $LOG

# ── DynaKITTI Baseline ─────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== DynaKITTI BASELINE ===" | tee -a $LOG

for SEQ in $SEQS_KITTI; do
    IMG=$DATA_KITTI/$SEQ/image_2
    CALIB=$DATA_KITTI/$SEQ/calib_numeric.txt
    GT=$DATA_KITTI/$SEQ/pose_left.txt

    [ ! -d "$IMG" ] && echo "[skip] $SEQ: нет image_2" && continue
    [ ! -f "$GT"  ] && echo "[skip] $SEQ: нет pose_left.txt" && continue

    echo "--- baseline $SEQ ---" | tee -a $LOG
    python main/eval.py \
        --config-path=/workspace/leapvo/configs \
        --config-name=dynakitti \
        data.imagedir=$IMG \
        data.calib=$CALIB \
        data.gt_traj=$GT \
        data.name=dynakitti-$SEQ \
        data.savedir=$RES_BASE \
        data.traj_format=kitti \
        save_video=false 2>&1 | grep -E "(ATE|RPE|t_rel|Error)" | tee -a $LOG
done

# ── DynaKITTI + YOLO ──────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== DynaKITTI + YOLOv11n ===" | tee -a $LOG

for SEQ in $SEQS_KITTI; do
    IMG=$DATA_KITTI/$SEQ/image_2
    CALIB=$DATA_KITTI/$SEQ/calib_numeric.txt
    GT=$DATA_KITTI/$SEQ/pose_left.txt

    [ ! -d "$IMG" ] && continue
    [ ! -f "$GT"  ] && continue

    echo "--- yolo $SEQ ---" | tee -a $LOG
    python main/eval_yolo.py \
        --config-path=/workspace/leapvo/configs \
        --config-name=dynakitti \
        data.imagedir=$IMG \
        data.calib=$CALIB \
        data.gt_traj=$GT \
        data.name=dynakitti-$SEQ \
        data.savedir=$RES_YOLO \
        data.traj_format=kitti \
        save_video=false \
        use_yolo=true \
        yolo_size=n \
        yolo_conf=0.4 2>&1 | grep -E "(ATE|RPE|t_rel|YOLO|Error)" | tee -a $LOG
done

# ── Shibuya Baseline ───────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== Shibuya BASELINE ===" | tee -a $LOG

for SEQ in $SEQS_SHIB; do
    IMG=$DATA_SHIB/$SEQ/image_0
    [ ! -d "$IMG" ] && IMG=$DATA_SHIB/$SEQ/image_2
    GT=$DATA_SHIB/$SEQ/gt_pose.txt

    [ ! -d "$IMG" ] && echo "[skip] $SEQ: нет image_0" && continue
    [ ! -f "$GT"  ] && echo "[skip] $SEQ: нет gt_pose.txt" && continue

    # Генерируем числовой calib из стандартных параметров камеры Shibuya
    # (fx fy cx cy из TartanAir: 320.0 320.0 320.0 240.0)
    CALIB_TMP=$(mktemp /tmp/calib_XXXXXX.txt)
    echo "320.0 0.0 320.0 0.0 0.0 320.0 240.0 0.0 0.0 0.0 1.0 0.0" > $CALIB_TMP

    echo "--- baseline $SEQ ---" | tee -a $LOG
    python main/eval.py \
        --config-path=/workspace/leapvo/configs \
        --config-name=dynakitti \
        data.imagedir=$IMG \
        data.calib=$CALIB_TMP \
        data.gt_traj=$GT \
        data.name=shibuya-$SEQ \
        data.savedir=$SHIB_BASE \
        data.traj_format=kitti \
        save_video=false 2>&1 | grep -E "(ATE|RPE|t_rel|Error)" | tee -a $LOG

    rm -f $CALIB_TMP
done

# ── Shibuya + YOLO ─────────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== Shibuya + YOLOv11n ===" | tee -a $LOG

for SEQ in $SEQS_SHIB; do
    IMG=$DATA_SHIB/$SEQ/image_0
    [ ! -d "$IMG" ] && IMG=$DATA_SHIB/$SEQ/image_2
    GT=$DATA_SHIB/$SEQ/gt_pose.txt

    [ ! -d "$IMG" ] && continue
    [ ! -f "$GT"  ] && continue

    CALIB_TMP=$(mktemp /tmp/calib_XXXXXX.txt)
    echo "320.0 0.0 320.0 0.0 0.0 320.0 240.0 0.0 0.0 0.0 1.0 0.0" > $CALIB_TMP

    echo "--- yolo $SEQ ---" | tee -a $LOG
    python main/eval_yolo.py \
        --config-path=/workspace/leapvo/configs \
        --config-name=dynakitti \
        data.imagedir=$IMG \
        data.calib=$CALIB_TMP \
        data.gt_traj=$GT \
        data.name=shibuya-$SEQ \
        data.savedir=$SHIB_YOLO \
        data.traj_format=kitti \
        save_video=false \
        use_yolo=true \
        yolo_size=n \
        yolo_conf=0.4 2>&1 | grep -E "(ATE|RPE|t_rel|YOLO|Error)" | tee -a $LOG

    rm -f $CALIB_TMP
done

# ── Итоговое сравнение ─────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "  ИТОГОВОЕ СРАВНЕНИЕ ATE RMSE"              | tee -a $LOG
echo "============================================" | tee -a $LOG

python3 << 'PYEOF' 2>&1 | tee -a $LOG
import json
from pathlib import Path

dirs = {
    "DynaKITTI Baseline": Path("/workspace/leapvo/results/dynakitti"),
    "DynaKITTI +YOLO":    Path("/workspace/leapvo/results/dynakitti_yolo"),
    "Shibuya Baseline":   Path("/workspace/leapvo/results/shibuya"),
    "Shibuya +YOLO":      Path("/workspace/leapvo/results/shibuya_yolo"),
}

for label, d in dirs.items():
    jsons = sorted(d.rglob("metrics.json"))
    if not jsons:
        print(f"\n{label}: нет результатов")
        continue
    print(f"\n{label}:")
    print(f"  {'Seq':<25} {'ATE':>8} {'RPE-t':>8} {'cov%':>7}")
    print(f"  {'-'*55}")
    ates = []
    for jf in jsons:
        try:
            m = json.loads(jf.read_text())
            cov = m.get('mask_coverage_pct_mean', None)
            cov_s = f"{cov:.1f}" if cov is not None else "  n/a"
            print(f"  {m['seq']:<25} {m['ate_rmse']:>8.4f} {m['rpe_trans']:>8.4f} {cov_s:>7}")
            ates.append(m['ate_rmse'])
        except Exception:
            pass
    if ates:
        import numpy as np
        print(f"  {'MEAN':<25} {np.mean(ates):>8.4f}")
PYEOF

echo "" | tee -a $LOG
echo "Результаты: $LOG"
echo "Готово ✓"
