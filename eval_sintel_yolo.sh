#!/bin/bash
DATASET=/workspace/data/MPI-Sintel-complete/training
SAVEDIR=/workspace/leapvo/results/sintel_yolo
CONFIG_PATH=/workspace/leapvo/configs

mkdir -p $SAVEDIR
echo "$(date '+%Y-%m-%d %H:%M:%S') Sintel+YOLO eval start" >> $SAVEDIR/error_sum.txt

for SCENE in alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3
do
    SCENE_PATH=$DATASET/final/$SCENE
    CAM_PATH=$DATASET/camdata_left/$SCENE

    if [ ! -d "$SCENE_PATH" ]; then
        echo "[skip] $SCENE: нет $SCENE_PATH"
        continue
    fi
    if [ ! -d "$CAM_PATH" ]; then
        echo "[skip] $SCENE: нет $CAM_PATH"
        continue
    fi

    echo "====== $SCENE + YOLO ======"
    python main/eval_yolo.py \
        --config-path=$CONFIG_PATH \
        --config-name=sintel \
        data.imagedir=$SCENE_PATH \
        data.gt_traj=$CAM_PATH \
        data.savedir=$SAVEDIR \
        data.calib=$CAM_PATH \
        data.name=M-$SCENE \
        save_video=false \
        use_yolo=true \
        yolo_size=n \
        yolo_conf=0.4
done