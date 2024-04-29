import os
import time
import argparse
from os import system

import glob
import GPUtil
import tqdm
import math

import numpy as np
import subprocess as sp

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = {i: int(x.split()[0]) for i, x in enumerate(memory_free_info)}
    return memory_free_values

def tmux(command):
    system('tmux %s' % command)

def tmux_shell(command):
    tmux('send-keys "%s" "C-m"' % command)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="paths to config list file that contains paths to config file needed to be run",
  )
  parser.add_argument(
        "--gpu_num",
        type=int,
        default=8,
  )
  parser.add_argument(
        "--block_gpu_ids",
        type=list,
        default=[],
  )
  parser.add_argument(
        "--command",
        type=str,
        default='robo ; gpu{} irun ',
  )
  parser.add_argument(
        "--depth",
        type=int,
        default=2,
  )
  args = parser.parse_args()

  with open(args.config_file) as f:
    config_files = f.readlines()

  gpu_num = args.gpu_num
  block_gpu_ids = args.block_gpu_ids
  
  gpu_to_use = [id for id in list(range(gpu_num)) if id not in block_gpu_ids]

  print('obtained {} config files:'.format(len(config_files)))
  print(config_files)
  
  print('gpus to use: {}'.format(gpu_to_use))

  conf_per_batch = math.ceil(len(config_files) / len(gpu_to_use))

  # lineup configs sequentially on each of the gpus
  for batch_id, gpu_id in enumerate(gpu_to_use):
    start = conf_per_batch*batch_id
    end = start + conf_per_batch
    config_batch = config_files[start:end]
    batch_cmd = ""
    for conf in config_batch:
      batch_cmd += args.command.format(gpu_id) + conf + ' ; '
    batch_cmd = batch_cmd[:-2]

    print(batch_cmd)

    if batch_id > 0:
      if batch_id % 4 == 0:
        tmux('new-window')
      else:
        tmux('split-window -v')

    tmux('select-layout even-vertical')
    tmux_shell(batch_cmd)

if __name__ == "__main__":
    main()