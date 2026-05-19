#!/bin/bash
# eval_shibuya_yolo.sh — запуск LeapVO+YOLOv11 на Shibuya
# Запуск: bash /workspace/leapvo/eval_shibuya_yolo.sh

DATASET=/workspace/data/shibuya
SAVEDIR=/workspace/leapvo/results/shibuya_yolo
CALIB=/workspace/leapvo/calibs/tartan_shibuya.txt
CONFIG_PATH=/workspace/leapvo/configs

mkdir -p $SAVEDIR
echo "$(date '+%Y-%m-%d %H:%M:%S') Shibuya+YOLO eval start" >> $SAVEDIR/error_sum.txt

for SCENE in Standing01 RoadCrossing07 RoadCrossing05 Standing02 RoadCrossing03 RoadCrossing04  RoadCrossing06
do
    IMG=$DATASET/$SCENE/image_0
    GT=$DATASET/$SCENE/gt_pose.txt

    if [ ! -d "$IMG" ]; then
        echo "[skip] $SCENE: нет image_0"
        continue
    fi
    if [ ! -f "$GT" ]; then
        echo "[skip] $SCENE: нет gt_pose.txt"
        continue
    fi

    echo "====== $SCENE + YOLO ======"
    python main/eval_yolo.py \
        --config-path=$CONFIG_PATH \
        --config-name=shibuya \
        data.imagedir=$IMG \
        data.gt_traj=$GT \
        data.savedir=$SAVEDIR \
        data.calib=$CALIB \
        data.name=$SCENE \
        save_video=false \
        use_yolo=true \
        yolo_size=n \
        yolo_conf=0.4
done

echo "====== Готово. Результаты: $SAVEDIR ======"
