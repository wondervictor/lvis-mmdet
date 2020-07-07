#!/usr/bin/env sh

CWD=`pwd` 
HDFS=hdfs://hobot-bigdata/
#set this to enable reading from hdfs
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms512m -Xmx10000m"

cd ${WORKING_PATH}
cp -r ${WORKING_PATH}/* /job_data



####get conda env mmdetection
echo get detectron2 conda env
mkdir conda_env 
cd conda_env 
hdfs dfs -get hdfs://hobot-bigdata/user/shaoyu.chen/envs/detectron2.tar.gz && tar xf detectron2.tar.gz
export PATH=/running_package/lvis-mmdet/conda_env/detectron2/bin:$PATH
cd ..


####get gcc-5.5.0
echo get gcc-5.5.0
hdfs dfs -get hdfs://hobot-bigdata/user/shaoyu.chen/envs/gcc-5.5.0.tar && tar xf gcc-5.5.0.tar  

export PATH=/running_package/lvis-mmdet/gcc-5.5.0/bin/:$PATH
export LD_LIBRARY_PATH=/running_package/lvis-mmdet/gcc-5.5.0/lib:/running_package/lvis-mmdet/gcc-5.5.0/lib64:$LD_LIBRARY_PATH
export PATH=/running_package/lvis-mmdet/conda_env/detectron2/bin:$PATH

which python

find ./mmdet -name  '*.so' | xargs  rm
rm build -r
rm -rf mmdet.egg-info

python -m pip install mmcv

# python -m pip install -v -e .
python setup.py build develop

# rm maskrcnn_benchmark.egg-info -r
# python setup.py build develop


#mkdir ~/.torch/models -p
#mkdir ~/.torch/fvcore_cache/detectron2/ImageNetPretrained/MSRA/ -p

#cp ./pretrained_model/*  ~/.torch/models
#cp ./pretrained_model/* ~/.torch/fvcore_cache/detectron2/ImageNetPretrained/MSRA/


####get coco data
echo get coco data and lvis json
mkdir data && cd data
hdfs dfs -get hdfs://hobot-bigdata/user/shaoyu.chen/data/coco.tar && tar xf coco.tar 
hdfs dfs -get hdfs://hobot-bigdata/user/shaoyu.chen/data/lvis
cp lvis/lvis_v0.5_train.json coco/annotations/
cp lvis/lvis_v0.5_val.json coco/annotations/
cd ..


GPUS=2
# ask_rcnn_r50_fpn_ga_2x_lvis
# 
# CONFIG=mask_rcnn_r50_fpn_2x_lvis
CONFIG=mask_rcnn_r50_fpn_ga_2x_lvis

####train

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=23333 tools/train.py configs/lvis/${CONFIG}.py  --launcher pytorch --work-dir output/${CONFIG}


# python tools/train.py configs/$1.py  --resume_from /job_data/work_dirs/$1/epoch_53.pth
# ./tools/dist_train.sh configs/$1.py 4 --validate   #--resume_from /job_data/work_dirs/$1/latest.pth

mkdir -p /job_data/final_package

# cd ${WORKING_PATH}/projects/GaussianRCNN/
# ln -s ../../datasets ./datasets

# python train_net.py --config-file configs/mask_rcnn_R_50_highres_FPN_c128_1x.yaml --num-gpus 4

# python train_net.py --config-file configs/LVIS-RPN/rpn_R_50_FPN_atss_1x.yaml --num-gpus 8
# python train_net.py --config-file configs/mask_rcnn_hrnetv2_w18_FPN_thr_1x.yaml --num-gpus 4
# python train_net.py --config-file configs/mask_rcnn_R_50_FPN_atssrpn_thr_1x.yaml  --num-gpus 8
# python train_net.py --config-file configs/retinanet_R_50_FPN_rs_1x.yaml  --num-gpus 8
# python train_net.py --config-file configs/LVIS-RPN/rpn_R_50_FPN_aug_thr_1x.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS weights/mask_rcnn_r50_aug_anchors_thrv1_1x.pth
# python train_net.py --config-file configs/mask_rcnn_R_50_FPN_anchorthr_coco_1x.yaml --num-gpus 8
# python train_net.py --config-file configs/mask_rcnn_R_50_FPN_aug_anchor_thrv2_1x.yaml --num-gpus 8
# python train_net.py --config-file configs/LVIS-RPN/rpn_R_50_FPN_aug_thrv2_1x.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS weights/mask_rcnn_r50_aug_anchors_thrv2_1x.pth

# python train_net.py --config-file configs/mask_rcnn_R_50_FPN_aug_anchors_1x.yaml --num-gpus 4
# python train_net.py --config-file configs/LVIS-RPN/rpn_R_50_FPN_aug_small_1x.yaml --num-gpus 4
# python train_net.py --config-file configs/mask_rcnn_R_50_FPN_dh_1x.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS output/mask_rcnn_r50_dh_1x/model_final.pth
#python train_net.py --config-file configs/mask_rcnn_R_50_FPN_dh_1x.yaml --num-gpus 8
#python train_net.py --config-file configs/mask_rcnn_R_50_FPN_dh_agnostic_1x.yaml --num-gpus 8
#python train_net.py --config-file configs/mask_rcnn_R_50_FPN_coco_dh_1x.yaml --num-gpus 8

# python train_net.py --config-file configs/LVIS-RPN/rpn_R_50_FPN_1x.yaml --num-gpus 4
# python train_net.py --config-file configs/LVIS-RPN/rpn_R_50_FPN_rs_1x.yaml --num-gpus 4

cp -r ${WORKING_PATH}/output   /job_data/final_package
mv ${WORKING_PATH}/output   /job_data

