#!/bin/bash

# ==========obs_ablation==========

#  task: square
#    dataset type: ph
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/low_dim/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/low_dim/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/low_dim/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/low_dim/bc_rnn_add_proprio.json

#  task: square
#    dataset type: ph
#      hdf5 type: image
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_remove_rand.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_rnn_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_rnn_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/ph/image/bc_rnn_remove_rand.json

#  task: square
#    dataset type: mh
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/low_dim/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/low_dim/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/low_dim/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/low_dim/bc_rnn_add_proprio.json

#  task: square
#    dataset type: mh
#      hdf5 type: image
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_remove_rand.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_rnn_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_rnn_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/square/mh/image/bc_rnn_remove_rand.json

#  task: transport
#    dataset type: ph
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/low_dim/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/low_dim/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/low_dim/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/low_dim/bc_rnn_add_proprio.json

#  task: transport
#    dataset type: ph
#      hdf5 type: image
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_remove_rand.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_rnn_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_rnn_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/ph/image/bc_rnn_remove_rand.json

#  task: transport
#    dataset type: mh
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/low_dim/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/low_dim/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/low_dim/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/low_dim/bc_rnn_add_proprio.json

#  task: transport
#    dataset type: mh
#      hdf5 type: image
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_remove_rand.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_rnn_add_eef_vel.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_rnn_add_proprio.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_rnn_remove_wrist.json
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/obs_ablation/transport/mh/image/bc_rnn_remove_rand.json

