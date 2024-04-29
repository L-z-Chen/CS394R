#  task: can
#    dataset type: mh/worse
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/can/mh/worse/low_dim/cql.json

#  task: can
#    dataset type: mh/okay
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/can/mh/okay/low_dim/cql.json

#  task: can
#    dataset type: mh/better
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/can/mh/better/low_dim/cql.json

#  task: can
#    dataset type: mh/worse_okay
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/can/mh/worse_okay/low_dim/cql.json

#  task: can
#    dataset type: mh/worse_better
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/can/mh/worse_better/low_dim/cql.json

#  task: can
#    dataset type: mh/okay_better
#      hdf5 type: low_dim
python /u/shuhan/install/robomimic/robomimic/robomimic/scripts/train.py --config /vision/shuhan/projects/imitation/robomimic/configs/subopt/can/mh/okay_better/low_dim/cql.json
