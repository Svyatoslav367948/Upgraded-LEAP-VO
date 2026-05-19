DATASET=/workspace/data/shibuya
SAVEDIR=/workspace/leapvo/results/shibuya
CALIB=/workspace/leapvo/calibs/tartan_shibuya.txt
CONFIG_PATH=/workspace/leapvo/configs

mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt

for SCENE in Standing01 Standing02 RoadCrossing03 RoadCrossing04 RoadCrossing05 RoadCrossing06 RoadCrossing07
do
    python main/eval.py \
    --config-path=$CONFIG_PATH \
    --config-name=shibuya \
    data.imagedir=$DATASET/$SCENE/image_0 \
    data.gt_traj=$DATASET/$SCENE/gt_pose.txt \
    data.savedir=$SAVEDIR \
    data.calib=$CALIB \
    data.name=$SCENE \
    save_video=false 
done


