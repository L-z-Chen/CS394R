import numpy as np
import torch
import math
import random
from robomimic.utils.registry import registry
from bisect import bisect_right
from torch.utils.data import WeightedRandomSampler, DataLoader
from robomimic.utils.dataset import custom_collate

class Curriculm():
  def __init__(self, config, dataset):
    self.config = config
    if type(dataset) is not list:
      self.sample_num = len(dataset)
    else:
      self.sample_num = sum([len(dset) for dset in dataset])
    print('Curriculum: dataset total sample number', self.sample_num)
  
  def get_sample_weight(self):
    '''
    obtain sampling weight for each sample in the dataset
    '''
    raise NotImplementedError
  
  def get_weighted_sampler(self, **kwargs):
    weights = self.get_sample_weight(**kwargs)
    sampler = WeightedRandomSampler(weights, self.sample_num, replacement=False)

    return sampler
  
@registry.register_curriculm(name='uniform')
class UniformCurr(Curriculm):
  def get_sample_weight(self, **kwargs):
    self.weight_sum = self.sample_num

    return torch.ones(self.sample_num)

@registry.register_curriculm(name='demo_quality')
class DemoQualityLabelCurr(Curriculm):
  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    
    self._index_to_mask_names = {}
    for index, demo_id in dataset._index_to_demo_id.items():
      self._index_to_mask_names[index] = dataset._demo_mask_names[demo_id]
    
    self.curr_config = self.config.curriculm.demo_quality

  def _get_current_mask_names(self, cur_epoch):
    milestones = sorted(self.curr_config.milestones)
    mask_idx = bisect_right(milestones, cur_epoch)

    current_mask_names = self.curr_config.mask_names[mask_idx]
    return current_mask_names

  def get_sample_weight(self, cur_epoch, **kwargs):
    self.current_mask_names = set(self._get_current_mask_names(cur_epoch))

    weights = torch.zeros(self.sample_num)
    for index in range(self.sample_num):
      sample_mask_names = set(self._index_to_mask_names[index])
      if len(set.intersection(sample_mask_names, self.current_mask_names)) > 0:
        weight = self.curr_config.sample_weights[0]
      else:
        weight = self.curr_config.sample_weights[1]
      weights[index] = weight

    self.weight_sum = torch.sum(weights)
    
    print('Demo Quality Curriculm: ')
    print('       current mask names: {}'.format(self.current_mask_names))
    print('       weight sum: {}/{}'.format(self.weight_sum, self.sample_num))

    return weights


@registry.register_curriculm(name='soft_demo_quality')
class SoftDemoQualityLabelCurr(Curriculm):
  '''
  assume two kinds of demo quality
  sample weights changing from 1:0 (epoch 0) to 0:1 (epoch milestone) 
  '''
  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    
    self._index_to_mask_names = {}
    for index, demo_id in dataset._index_to_demo_id.items():
      self._index_to_mask_names[index] = dataset._demo_mask_names[demo_id]
    
    self.curr_config = self.config.curriculm.demo_quality

  def get_sample_weight(self, cur_epoch, **kwargs):
    all_mask_names = self.curr_config.mask_names
    assert len(all_mask_names) == 2
    mask_name_low_quality = all_mask_names[0]
    mask_name_high_quality = all_mask_names[1]

    assert len(self.curr_config.milestones) == 1
    milestone = sorted(self.curr_config.milestones)[0]

    weights = torch.zeros(self.sample_num)

    weights_quality = [max((milestone - cur_epoch) / milestone, 0.0), 
                       min(cur_epoch / milestone, 1.0)]  # for low-quality and high-quality demo respectively
    for index in range(self.sample_num):
      sample_mask_names = set(self._index_to_mask_names[index])
      if len(set.intersection(sample_mask_names, mask_name_low_quality)) > 0:
        weight = weights_quality[0]
      elif len(set.intersection(sample_mask_names, mask_name_high_quality)) > 0:
        weight = weights_quality[1]
      else:
        raise NotImplementedError
      weights[index] = weight

    self.weight_sum = torch.sum(weights)
    print('Demo Quality Curriculm: ')
    print('       weight sum: {}/{}'.format(self.weight_sum, self.sample_num))

    return weights


@registry.register_curriculm(name='loss_filter')
class LossFilterBinsCurr(Curriculm):
  '''
  No assumption on the kinds of demonstration qualities
  Assume every state in the dataset has a loss value (rated by a 20_percent_proficient model)
  Sample weights based on filter_value and bins:
    filter_value: usually be a scaled sigma value, filter data with loss larger than the value
    bins: devide the training stage into substages, every stage only use 1/bins of all data (ranked by loss value)
  '''
  def __init__(self, config, datasets):
    super().__init__(config, datasets)
    
    assert type(datasets) is list

    self.config = config

    self._all_index_to_make_names = []
    self._all_sample_num = []
    for dataset in datasets:
      _index_to_mask_names = {}
      for index, demo_id in dataset._index_to_demo_id.items():
        _index_to_mask_names[index] = dataset._demo_mask_names[demo_id]
      self._all_index_to_make_names.append(_index_to_mask_names)
      self._all_sample_num.append(len(dataset))
    
    self.curr_config = self.config.curriculm.loss_filter
    self.total_train_epoch = self.config.train.num_epochs
    self.stage_idx = 0

    if self.curr_config.milestone == [0]:
      num_epoch_per_stage = self.total_train_epoch // self.curr_config.bins[0]
      self.milestones = list(range(0, self.total_train_epoch, num_epoch_per_stage))
    else:
      self.milestones = self.curr_config.milestone
    print('Curriculum milestones', self.milestones)

    self.get_all_losses(datasets)
    self.weights = None

  def get_all_losses(self, datasets):
    print('Processing losses of all samples')
    # get loss values of all states in all datasets
    self._all_losses = []
    for dataset in datasets:
      loader = DataLoader(dataset=dataset,
                          batch_size=self.config.train.batch_size,
                          shuffle=False,
                          num_workers=self.config.train.num_data_workers,
                          drop_last=False,
                          collate_fn=custom_collate)
      print('Current subset length {}'.format(len(loader)))
      cur_losses = []
      for _, batch in enumerate(loader):
        cur_losses.append(batch['losses'].float().view(-1))
      cur_losses = torch.cat(cur_losses)
      self._all_losses.append(cur_losses)
      print('Current sub-dataset losses: Mean {}, Std {}, Median {}'.format(
        torch.mean(cur_losses), torch.std(cur_losses), torch.median(cur_losses)
      ))
  
  def get_sample_weight(self, cur_epoch, **kwargs):
    cur_epoch = max(0, cur_epoch - 1) # training epoch strat at 1

    if cur_epoch in self.milestones:
      print('Epoch {}, computing curriculum sampling weights'.format(cur_epoch))
      weights = torch.zeros(self.sample_num)

      all_losses_1d = torch.cat(self._all_losses)
      assert all_losses_1d.shape[0] == self.sample_num
      all_losses_1d_log10 = torch.log10(all_losses_1d)
      all_losses_1d_log10_valid = all_losses_1d_log10[~torch.isnan(all_losses_1d_log10)]

      mean, std = torch.mean(all_losses_1d_log10_valid), torch.std(all_losses_1d_log10_valid)
      scale = float(self.curr_config.filter_value[0].replace('sigma', ''))
      loss_filter_weights = (all_losses_1d < 10 ** (mean + scale * std)).float()
      num_valid_samples = int(torch.sum(loss_filter_weights))
      num_invalid_samples = all_losses_1d.shape[0] - num_valid_samples
      print('{}/{} data ({}%) are filtered by loss value'.format(
        num_invalid_samples, all_losses_1d.shape[0], 100 * num_invalid_samples / all_losses_1d.shape[0]))

      num_samples_per_stage = num_valid_samples / self.curr_config.bins[0]

      _, sorted_idx = torch.sort(all_losses_1d, descending=True)
      min_idx = int(num_invalid_samples + num_samples_per_stage * self.stage_idx)
      max_idx = min(sorted_idx.shape[0], int(num_invalid_samples + num_samples_per_stage * (self.stage_idx + 1)))
      curriculm_idx = sorted_idx[min_idx: max_idx]
      print('{} sampels are chosen by curriculum, index from {} to {}'.format(curriculm_idx.shape[0], min_idx, max_idx))

      weights[curriculm_idx] = 1.0
      weights *= loss_filter_weights
      print('Curriculum stage {}/{}, using {} sampels ({} in total)'.format(
        self.stage_idx + 1, self.curr_config.bins[0], curriculm_idx.shape[0], self.sample_num))

      self.stage_idx += 1
      self.weights = weights
      self.weight_sum = torch.sum(weights)
    else:
      print('Epoch {}, use cached curriculum sampling weights (stage not finished)'.format(cur_epoch))
    
    self.get_loss_statistics(print_stat=True)
    return self.weights

  def get_loss_statistics(self, print_stat=False):
    all_losses_1d = torch.cat(self._all_losses)
    weights_idx = self.weights == 1.0

    losses = all_losses_1d[weights_idx]

    mean = torch.mean(losses)
    std = torch.std(losses)
    median = torch.median(losses)

    stat = {
      'curr_loss_mean': mean,
      'curr_loss_std': std,
      'curr_loss_median': median
    }
    if print_stat:
      tmp = ''
      for k, v in stat.items():
        tmp += '{}: {}, '.format(k, v)
      print(tmp[:-2])
    return stat


@registry.register_curriculm(name='loss_filter_stageNormalized')
class LossFilterBinsCurr(Curriculm):
  '''
  No assumption on the kinds of demonstration qualities
  Assume every state in the dataset has a loss value (rated by a 20_percent_proficient model)
  Sample weights based on filter_value and bins:
    filter_value: usually be a scaled sigma value, filter data with loss larger than the value
    bins: devide the training stage into substages, every stage only use 1/bins of all data (ranked by loss value)
  '''
  def __init__(self, config, datasets):
    super().__init__(config, datasets)
    
    assert type(datasets) is list

    self.config = config

    self._all_index_to_make_names = []
    self._all_sample_num = []
    for dataset in datasets:
      _index_to_mask_names = {}
      for index, demo_id in dataset._index_to_demo_id.items():
        _index_to_mask_names[index] = dataset._demo_mask_names[demo_id]
      self._all_index_to_make_names.append(_index_to_mask_names)
      self._all_sample_num.append(len(dataset))
    
    self.curr_config = self.config.curriculm.loss_filter
    self.total_train_epoch = self.config.train.num_epochs
    self.bin_idx = 0
    self.num_stages = self.curr_config.demo_stages

    if self.curr_config.milestone == [0]:
      # using even bins
      num_epoch_per_stage = self.total_train_epoch // self.curr_config.bins[0]
      self.milestones = list(range(0, self.total_train_epoch, num_epoch_per_stage))
    else:
      # using uneven bins
      assert self.curr_config.bins[0] == len(self.curr_config.milestone)
      self.milestones = self.curr_config.milestone
    print('Curriculum milestones', self.milestones)

    self.get_all_losses(datasets)
    self.weights = None
  
  def get_all_losses(self, datasets):
    print('Processing losses of all samples')
    # get loss values of all states in all datasets
    self._all_losses = []
    for dataset in datasets:
      loader = DataLoader(dataset=dataset,
                          batch_size=self.config.train.batch_size,
                          shuffle=False,
                          num_workers=self.config.train.num_data_workers,
                          drop_last=False,
                          collate_fn=custom_collate)
      print('Current subset length {}'.format(len(loader)))
      cur_losses = []
      for _, batch in enumerate(loader):
        cur_losses.append(batch['losses'].float().view(-1))
      cur_losses = torch.cat(cur_losses)
      self._all_losses.append(cur_losses)
      print('Current sub-dataset losses: Mean {}, Std {}, Median {}'.format(
        torch.mean(cur_losses), torch.std(cur_losses), torch.median(cur_losses)
      ))
    self.get_all_stages()

  def get_all_stages(self):
    self._all_stages = []
    for it in self._all_losses:
      demo_len = it.shape[0]
      cur_stages = torch.ones(demo_len)
      stage_len = demo_len / self.num_stages
      for i in range(self.num_stages):
        min_idx = int(stage_len * i)
        max_idx = min(demo_len, int(stage_len * (i + 1)))
        cur_stages[min_idx: max_idx] *= i
      self._all_stages.append(cur_stages)
  
  def get_sample_weight(self, cur_epoch, **kwargs):
    cur_epoch = max(0, cur_epoch - 1) # training epoch strat at 1

    if cur_epoch in self.milestones:
      print('Epoch {}, computing curriculum sampling weights'.format(cur_epoch))
      weights = torch.zeros(self.sample_num)

      all_losses_1d = torch.cat(self._all_losses)
      assert all_losses_1d.shape[0] == self.sample_num
      all_losses_1d_log10 = torch.log10(all_losses_1d)
      all_losses_1d_log10_valid = all_losses_1d_log10[~torch.isnan(all_losses_1d_log10)]

      # filter the state-action by scaled STD of the Gaussian distribution
      mean, std = torch.mean(all_losses_1d_log10_valid), torch.std(all_losses_1d_log10_valid)
      scale = float(self.curr_config.filter_value[0].replace('sigma', ''))
      loss_filter_weights = (all_losses_1d < 10 ** (mean + scale * std)).float()
      num_valid_samples = int(torch.sum(loss_filter_weights))
      num_invalid_samples = all_losses_1d.shape[0] - num_valid_samples
      print('{}/{} data ({}%) are filtered by loss value'.format(
        num_invalid_samples, all_losses_1d.shape[0], 100 * num_invalid_samples / all_losses_1d.shape[0]))

      # count the number of filtered state-actions in each stage
      all_stages_1d = torch.cat(self._all_stages)
      filter_stages = all_stages_1d[~loss_filter_weights.bool()]
      count_filter_stage = {}
      for stage in range(self.num_stages):
        count_filter_stage['stage_' + str(stage)] = torch.sum(filter_stages == stage).item()
      print('The filtered data are from demo stage of demonstration', count_filter_stage)

      # each demostation is split into sub-stages
      num_samples_per_bin = num_valid_samples / self.curr_config.bins[0] / self.num_stages
      all_stages_1d = torch.cat(self._all_stages)
      for stage in range(self.num_stages):
        idx_stage = torch.where(all_stages_1d == stage)[0]
        all_losses_1d_curStage = all_losses_1d[idx_stage]
        _, sorted_idx = torch.sort(all_losses_1d_curStage, descending=True)
        min_idx = int(count_filter_stage['stage_' + str(stage)] + num_samples_per_bin * self.bin_idx)
        max_idx = min(sorted_idx.shape[0], int(count_filter_stage['stage_' + str(stage)] + num_samples_per_bin * (self.bin_idx + 1)))
        curriculm_idx_curStage = sorted_idx[min_idx: max_idx]
        curriculm_idx = idx_stage[curriculm_idx_curStage]
        weights[curriculm_idx] = 1.0
        print('For demo stage {}, {}/{} samples are chosen by curriculum ({} bins)'.format(
          stage, max_idx - min_idx, all_losses_1d_curStage.shape[0], self.curr_config.bins[0]
        ))
      
      weights *= loss_filter_weights
      print('Curriculum bin {}/{}, using {} samples ({} in total)'.format(
        self.bin_idx + 1, self.curr_config.bins[0], torch.sum(weights), self.sample_num
      ))

      self.bin_idx += 1
      self.weights = weights
      self.weight_sum = torch.sum(weights)
    else:
      print('Epoch {}, use cached curriculum sampling weights (stage not finished)'.format(cur_epoch))
    
    self.get_loss_statistics(print_stat=True)
    return self.weights

  def get_loss_statistics(self, print_stat=False):
    all_losses_1d = torch.cat(self._all_losses)
    weights_idx = self.weights == 1.0

    losses = all_losses_1d[weights_idx]

    mean = torch.mean(losses)
    std = torch.std(losses)
    median = torch.median(losses)

    stat = {
      'curr_loss_mean': mean,
      'curr_loss_std': std,
      'curr_loss_median': median
    }
    if print_stat:
      tmp = ''
      for k, v in stat.items():
        tmp += '{}: {}, '.format(k, v)
      print(tmp[:-2])
    return stat


@registry.register_curriculm(name='loss_filter_demo')
class LossFilterBinsCurr(Curriculm):
  '''
  No assumption on the kinds of demonstration qualities
  Assume every state in the dataset has a loss value (rated by a 20_percent_proficient model)
  Sample weights based on filter_value and bins:
    filter_value: usually be a scaled sigma value, filter data with loss larger than the value
    bins: devide the training stage into substages, every stage only use 1/bins of all data (ranked by loss value)
  '''
  def __init__(self, config, datasets):
    super().__init__(config, datasets)
    
    assert type(datasets) is list

    # curriculum and dataset configuration
    self.config = config    
    self.curr_config = self.config.curriculm.loss_filter
    self.total_train_epoch = self.config.train.num_epochs
    self.bin_idx = 0

    if self.curr_config.milestone == [0]:
      num_epoch_per_stage = self.total_train_epoch // self.curr_config.bins[0]
      self.milestones = list(range(0, self.total_train_epoch, num_epoch_per_stage))
    else:
      self.milestones = self.curr_config.milestone
    assert len(self.milestones) == self.curr_config.bins[0]
    print('Curriculum milestones', self.milestones)

    # get demo_name of all state-action pairs
    self._all_index_to_mask_names = []  # length is number of sub-datasets
    self._all_index_to_demo_names = []
    self._all_sample_num = []
    for dataset in datasets:
      _index_to_mask_names = {}
      _index_to_demo_names = {}
      for index, demo_id in dataset._index_to_demo_id.items():
        _index_to_mask_names[index] = dataset._demo_mask_names[demo_id]
        _index_to_demo_names[index] = demo_id
      self._all_index_to_mask_names.append(_index_to_mask_names)
      self._all_index_to_demo_names.append(_index_to_demo_names)
      self._all_sample_num.append(len(dataset))

    self.get_all_losses(datasets)
    self.weights = None

  def get_all_losses(self, datasets):
    print('Processing losses of all samples')
    self.qualities = self.config.train.hdf5_filter_key.split('_')
    self.name_to_quality = {}

    self._all_losses, self._all_demo_names = [], []   # length is number of sub-datasets
    self.demo_losses = {}
    for dataset_idx, dataset in enumerate(datasets):
      loader = DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.config.train.num_data_workers,
                          drop_last=False,
                          collate_fn=custom_collate)
      print('Current subset length {}'.format(len(loader)))
      cur_losses = []
      quality = self.qualities[dataset_idx]
      for idx, meta in enumerate(loader):
        loss = meta['losses'].float().item()
        demo_name = 'dataset_{}_{}'.format(dataset_idx, dataset._index_to_demo_id[idx])
        self._all_losses.append(loss)
        self._all_demo_names.append(demo_name)
        self.demo_losses[demo_name] = self.demo_losses.get(demo_name, []) + [loss]
        self.name_to_quality[quality] = self.name_to_quality.get(quality, []) + [demo_name]
        cur_losses.append(loss)
      cur_losses = torch.tensor(cur_losses)
      print('Current sub-dataset losses: Mean {}, Std {}, Median {}'.format(
        torch.mean(cur_losses), torch.std(cur_losses), torch.median(cur_losses)
      ))
    print('For the full dataset, using {} demos in total'.format(len(self.demo_losses.keys())))

  def filter_highest_loss(self):
    '''State-level operation'''
    all_losses_1d = torch.tensor(self._all_losses)
    assert all_losses_1d.shape[0] == self.sample_num
    all_losses_1d_log10 = torch.log10(all_losses_1d)
    all_losses_1d_log10_valid = all_losses_1d_log10[~torch.isnan(all_losses_1d_log10)]

    # filter the state-action by scaled STD of the Gaussian distribution
    mean, std = torch.mean(all_losses_1d_log10_valid), torch.std(all_losses_1d_log10_valid)
    scale = float(self.curr_config.filter_value[0].replace('sigma', ''))
    loss_filter_weights = (all_losses_1d < 10 ** (mean + scale * std)).float()
    num_valid_samples = int(torch.sum(loss_filter_weights))
    num_invalid_samples = all_losses_1d.shape[0] - num_valid_samples
    print('{}/{} data ({}%) are filtered by loss value'.format(
      num_invalid_samples, all_losses_1d.shape[0], 100 * num_invalid_samples / all_losses_1d.shape[0]))
    return loss_filter_weights
  
  def get_sample_weight(self, cur_epoch, **kwargs):
    cur_epoch = max(0, cur_epoch - 1) # training epoch strat at 1

    if cur_epoch in self.milestones:
      print('Epoch {}, computing curriculum sampling weights'.format(cur_epoch))
      assert len(self._all_demo_names) == self.sample_num

      demo_losses_avg = {}
      for demo_name, losses in self.demo_losses.items():
        demo_losses_avg[demo_name] = torch.sum(torch.tensor(losses)) / len(losses)
      demo_names_ranked = [k for k, v in sorted(demo_losses_avg.items(), key=lambda x: x[1], reverse=True)]

      num_demos = len(self.demo_losses.keys())
      num_demos_per_bin = num_demos / self.curr_config.bins[0]
      min_idx = int(num_demos_per_bin * self.bin_idx)
      max_idx = min(num_demos, int(num_demos_per_bin * (self.bin_idx + 1)))
      valid_demo_names = demo_names_ranked[min_idx: max_idx]

      weights = torch.tensor([it in valid_demo_names for it in self._all_demo_names]).float()
      
      cur_bin_info = 'Curriculum bin {}/{}, {}/{} demos ({}/{} state-actions) are chosen'.format(
        self.bin_idx + 1, self.curr_config.bins[0], len(valid_demo_names), 
        len(demo_names_ranked), torch.sum(weights).int().item(), self.sample_num
      )

      if self.curr_config.filter_highest:
        weights_loss = self.filter_highest_loss()
        weights *= weights_loss

      self.weights = weights
      self.bin_idx += 1
      self.weight_sum = torch.sum(weights)
      self.cur_bin_info = cur_bin_info
      self.count_quality_demo_number(valid_demo_names)
    else:
      print('Epoch {}, use cached curriculum sampling weights (stage not finished)'.format(cur_epoch))

    print(self.cur_bin_info)
    self.get_loss_statistics(print_stat=True)
    return self.weights

  def get_loss_statistics(self, print_stat=True):
    all_losses = torch.tensor(self._all_losses)
    weights_idx = self.weights == 1.0

    losses = all_losses[weights_idx]

    mean = torch.mean(losses)
    std = torch.std(losses)
    median = torch.median(losses)

    stat = {
      'curr_loss_mean': mean,
      'curr_loss_std': std,
      'curr_loss_median': median
    }
    if print_stat:
      tmp = ''
      for k, v in stat.items():
        tmp += '{}: {}, '.format(k, v)
      print(tmp[:-2])
    return stat

  def count_quality_demo_number(self, demos):
    count = {}
    for k in self.name_to_quality.keys():
      count[k] = 0
    
    for demo in demos:
      for k, v in self.name_to_quality.items():
        if demo in v:
          count[k] += 1

    print('Current bin distribution of demo qualities', count)



      
        
@registry.register_curriculm(name='loss_filter_demo_soft')
class LossFilterBinsCurr(Curriculm):
  def __init__(self, config, datasets):
    super().__init__(config, datasets)
    
    assert type(datasets) is list

    # curriculum and dataset configuration
    self.config = config    
    self.curr_config = self.config.curriculm.loss_filter
    self.total_train_epoch = self.config.train.num_epochs

    # get demo_name of all state-action pairs
    self._all_index_to_mask_names = []  # length is number of sub-datasets
    self._all_index_to_demo_names = []
    self._all_sample_num = []
    for dataset in datasets:
      _index_to_mask_names = {}
      _index_to_demo_names = {}
      for index, demo_id in dataset._index_to_demo_id.items():
        _index_to_mask_names[index] = dataset._demo_mask_names[demo_id]
        _index_to_demo_names[index] = demo_id
      self._all_index_to_mask_names.append(_index_to_mask_names)
      self._all_index_to_demo_names.append(_index_to_demo_names)
      self._all_sample_num.append(len(dataset))

    self.get_all_losses(datasets)
    self.weights = None

    self.start_weights_bins = [1.0 / self.curr_config.bins[0]] * self.curr_config.bins[0]
    if self.curr_config.bins[0] == 10:
      self.end_weights_bins = [0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.35]
      #self.start_weights_bins = self.end_weights_bins[::-1]
    else:
      assert NotImplementedError
    self.start_weights_bins = torch.tensor(self.start_weights_bins) / sum(self.start_weights_bins)
    self.end_weights_bins = torch.tensor(self.end_weights_bins) / sum(self.end_weights_bins)
    self.alpha = 0.999975

  def get_all_losses(self, datasets):
    print('Processing losses of all samples')

    self.qualities = self.config.train.hdf5_filter_key.split('_')
    self.name_to_quality = {}

    self._all_losses, self._all_demo_names = [], []   # length is number of sub-datasets
    self.demo_losses = {}
    for dataset_idx, dataset in enumerate(datasets):
      loader = DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.config.train.num_data_workers,
                          drop_last=False,
                          collate_fn=custom_collate)
      print('Current subset length {}'.format(len(loader)))
      cur_losses = []
      quality = self.qualities[dataset_idx]
      for idx, meta in enumerate(loader):
        loss = meta['losses'].float().item()
        demo_name = 'dataset_{}_{}'.format(dataset_idx, dataset._index_to_demo_id[idx])
        self.name_to_quality[quality] = self.name_to_quality.get(quality, []) + [demo_name]
        self._all_losses.append(loss)
        self._all_demo_names.append(demo_name)
        self.demo_losses[demo_name] = self.demo_losses.get(demo_name, []) + [loss]
        cur_losses.append(loss)
      cur_losses = torch.tensor(cur_losses)
      print('Current sub-dataset losses: Mean {}, Std {}, Median {}'.format(
        torch.mean(cur_losses), torch.std(cur_losses), torch.median(cur_losses)
      ))
    print('For the full dataset, using {} demos in total'.format(len(self.demo_losses.keys())))

  def filter_highest_loss(self):
    '''State-level operation'''
    all_losses_1d = torch.tensor(self._all_losses)
    assert all_losses_1d.shape[0] == self.sample_num
    all_losses_1d_log10 = torch.log10(all_losses_1d)
    all_losses_1d_log10_valid = all_losses_1d_log10[~torch.isnan(all_losses_1d_log10)]

    # filter the state-action by scaled STD of the Gaussian distribution
    mean, std = torch.mean(all_losses_1d_log10_valid), torch.std(all_losses_1d_log10_valid)
    scale = float(self.curr_config.filter_value[0].replace('sigma', ''))
    loss_filter_weights = (all_losses_1d < 10 ** (mean + scale * std)).float()
    num_valid_samples = int(torch.sum(loss_filter_weights))
    num_invalid_samples = all_losses_1d.shape[0] - num_valid_samples
    print('{}/{} data ({}%) are filtered by loss value'.format(
      num_invalid_samples, all_losses_1d.shape[0], 100 * num_invalid_samples / all_losses_1d.shape[0]))
    return loss_filter_weights
  
  def get_sample_weight(self, cur_epoch, **kwargs):
    cur_epoch = max(0, cur_epoch - 1) # training epoch strat at 1

    print('Epoch {}, computing curriculum sampling weights'.format(cur_epoch))
    assert len(self._all_demo_names) == self.sample_num

    demo_losses_avg = {}
    for demo_name, losses in self.demo_losses.items():
      demo_losses_avg[demo_name] = torch.sum(torch.tensor(losses)) / len(losses)
    demo_names_ranked = [k for k, v in sorted(demo_losses_avg.items(), key=lambda x: x[1], reverse=True)] # highest to lowest

    num_demos = len(self.demo_losses.keys())
    num_demos_per_bin = num_demos / self.curr_config.bins[0]
    cur_iter = cur_epoch * self.config.experiment.epoch_every_n_steps
    sample_probs = (self.start_weights_bins - self.end_weights_bins) * (self.alpha ** cur_iter) + self.end_weights_bins
    data_ratio = sample_probs / torch.sum(sample_probs)
    
    name_demos_chosen = []
    for bin_idx in range(self.curr_config.bins[0]):
      min_idx = int(num_demos_per_bin * bin_idx)
      max_idx = min(num_demos, int(num_demos_per_bin * (bin_idx + 1)))
      bin_demo_names = demo_names_ranked[min_idx: max_idx]
      sample_prob_bin = sample_probs[bin_idx]
      sample_num = round(sample_prob_bin.item() * len(bin_demo_names))
      name_demos_chosen_bin = random.sample(bin_demo_names, sample_num)
      name_demos_chosen += name_demos_chosen_bin
      print('Demo bin {}/{}, choose {}/{} demos in the bin (sample prob {:.6f}), {:.3f}% of all data'.format(
        bin_idx+1, self.curr_config.bins[0], len(name_demos_chosen_bin), len(bin_demo_names), sample_prob_bin, 100.0 * data_ratio[bin_idx]
      ))

    weights = torch.tensor([it in name_demos_chosen for it in self._all_demo_names]).float()    

    if self.curr_config.filter_highest:
      weights_loss = self.filter_highest_loss()
      weights *= weights_loss

    print('{} demos including {} state-action pairs are chosen'.format(
      len(name_demos_chosen), torch.sum(weights)
    ))

    self.weights = weights
    self.weight_sum = torch.sum(weights)

    self.get_loss_statistics(print_stat=True)
    self.count_quality_demo_number(name_demos_chosen)
    return self.weights

  def get_loss_statistics(self, print_stat=True):
    all_losses = torch.tensor(self._all_losses)
    weights_idx = self.weights == 1.0

    losses = all_losses[weights_idx]

    mean = torch.mean(losses)
    std = torch.std(losses)
    median = torch.median(losses)

    stat = {
      'curr_loss_mean': mean,
      'curr_loss_std': std,
      'curr_loss_median': median
    }
    if print_stat:
      tmp = ''
      for k, v in stat.items():
        tmp += '{}: {}, '.format(k, v)
      print(tmp[:-2])
    return stat

  def count_quality_demo_number(self, demos):
    count = {}
    for k in self.name_to_quality.keys():
      count[k] = 0
    
    for demo in demos:
      for k, v in self.name_to_quality.items():
        if demo in v:
          count[k] += 1

    print('Current bin distribution of demo qualities', count)

