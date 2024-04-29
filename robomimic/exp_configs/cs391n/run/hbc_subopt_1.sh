#  task: square
#    dataset type: mh/worse
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/square/mh/worse/low_dim/hbc.json

#  task: square
#    dataset type: mh/okay
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/square/mh/okay/low_dim/hbc.json

#  task: square
#    dataset type: mh/better
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/square/mh/better/low_dim/hbc.json

#  task: square
#    dataset type: mh/worse_okay
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/square/mh/worse_okay/low_dim/hbc.json

#  task: square
#    dataset type: mh/worse_better
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/square/mh/worse_better/low_dim/hbc.json

#  task: square
#    dataset type: mh/okay_better
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/square/mh/okay_better/low_dim/hbc.json
