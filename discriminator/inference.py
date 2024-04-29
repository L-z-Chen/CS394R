import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import h5py


class ObvClassifier(pl.LightningModule):
    def __init__(self, feature_num):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_num, 256), nn.ReLU(), nn.Linear(256, 128))
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=-1))
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, input):
        input = torch.tensor(input).float()
        z = self.encoder(input)
        preds = self.classifier(z)
        return preds

    def forward_loss(self, batch, mode='train'):
        demoA = batch['demoA']
        demoB = batch['demoB']
        inputs = torch.cat([demoA, demoB], dim=0)
        labels = torch.cat([torch.zeros(demoA.shape[0]), torch.ones(demoB.shape[0])], dim=0)
        labels = labels.to(inputs.device).long()

        preds = self.forward(inputs)
        loss = F.cross_entropy(preds, labels)

        acc = getattr(self, '{}_acc'.format(mode))
        acc(preds.detach().cpu(), labels.detach().cpu())

        on_epoch = False if mode == 'train' else True
        self.log('{}_accuracy'.format(mode), acc.compute(), on_step=True, on_epoch=on_epoch)
        self.log('loss/{}_loss'.format(mode), loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_loss(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_loss(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.forward_loss(batch, 'test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class ObvDataDiscrete(Dataset):
    def __init__(self, obv_dict, tgt_demo_names, tgt_ranges, indices):
        self.demo_names = tgt_demo_names
        self.demo_obvs = {}
        self.demo_num = -1

        for demo_name in tgt_demo_names:
            demo_obv = []
            for demo_range in tgt_ranges:
                demo_obv.append(obv_dict[demo_name][demo_range])

            demo_obv = np.concatenate(demo_obv, axis=0)
            self.demo_obvs[demo_name] = torch.tensor(demo_obv[indices]).float()

            if self.demo_num > -1:
                assert self.demo_num == demo_obv.shape[0]
            else:
                self.demo_num = demo_obv.shape[0]

    def __len__(self):
        return self.demo_num

    def __getitem__(self, index):
        data = {}
        data['demoA'] = self.demo_obvs[self.demo_names[0]][index]
        data['demoB'] = self.demo_obvs[self.demo_names[1]][index]
        return data


def train_model():
    all_obv_data = np.load('./data/sampled_obv/can.npy', allow_pickle=True).item()
    save_folder = './checkpoints'

    tgt_demo_names = ['worse', 'better']
    for tgt_ranges in [['1/4'], ['2/4'], ['3/4'], ['4/4'], ['1/4', '2/4', '3/4', '4/4']]:
        total_num = all_obv_data['okay'][tgt_ranges[0]].shape[0] * len(tgt_ranges)
        train_num = int(total_num * 0.8)

        train_idx = random.sample(list(range(total_num)), train_num)
        test_idx = list(set(range(total_num)) - set(train_idx))
        val_idx = train_idx[:128]

        model_indices = {'train': train_idx, 'test': test_idx, 'val': val_idx}

        data_loaders = {}

        for mode, indices in model_indices.items():
            dataset = ObvDataDiscrete(all_obv_data, tgt_demo_names, tgt_ranges, indices)
            data_loaders[mode] = DataLoader(dataset, batch_size=32, num_workers=8)

        # init model
        classifier = ObvClassifier(feature_num=23)

        # Initialize a trainer
        trainer = pl.Trainer(gpus=1, max_epochs=20)

        print(tgt_ranges)

        # Train the model ⚡
        trainer.fit(classifier, data_loaders['train'], data_loaders['val'])
        trainer.test(classifier, data_loaders['test'])

        for idx, tgt_range in enumerate(tgt_ranges):
            tgt_ranges[idx] = tgt_range.replace('/', '-')

        save_name = '{}_{}.ckpt'.format('_'.join(tgt_demo_names), '_'.join(tgt_ranges))
        save_file = os.path.join(save_folder, save_name)
        trainer.save_checkpoint(save_file)



def get_discriminate_masks(classifier, features, gt_mode='worse', threshold=0.4):
    pred = classifier(features)

    pred_idx = 0 if gt_mode == 'worse' else 1
    shared_mask = pred[:, pred_idx] < threshold
    separate_mask = ~shared_mask

    return {'shared': shared_mask, 'separate': separate_mask}


if __name__ == '__main__':
    qualities = ['worse', 'better']
    # task_name = 'square'
    task_name = 'can'

    dataset_root = '/vision/shuhan/projects/imitation/mix_demo/datasets'
    checkpoint_root = '/vision/shuhan/projects/curriculm_il/mixture-demo-imitation/data/model/obv_classify'

    model_path = checkpoint_root + '/{}/{}_{}_1-4_2-4_3-4_4-4.ckpt'.format(task_name, qualities[0], qualities[1])
    classifier = ObvClassifier(feature_num=23)
    classifier = ObvClassifier.load_from_checkpoint(checkpoint_path=model_path, feature_num=23)

    # file = h5py.File('/vision/hwjiang/robot_learning/robomimic/datasets/{}/mh/low_dim.hdf5'.format(task_name), 'r')
    # file2 = h5py.File('/vision/hwjiang/robot_learning/robomimic/datasets/{}/mh/low_dim_worse_separate.hdf5'.format(task_name), 'r+')

    file = h5py.File(dataset_root + '/{}/mh/low_dim.hdf5'.format(task_name), 'r')
    file2 = h5py.File(dataset_root + '/{}/mh/low_dim_worse_separate.hdf5'.format(task_name), 'r+')

    target_demo_names = [it for it in file['mask'][qualities[0]]]

    print(target_demo_names)

    obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
    new_demo_dict = {}

    for name in target_demo_names:
        data = file['data'][name]
        features = []
        for obs_key in obs_keys:
            features.append(file['data'][name]['obs'][obs_key])
        features = torch.from_numpy(np.concatenate(features, axis=1))
        mask = get_discriminate_masks(classifier, features)
        separate_mask = mask['separate'].detach().cpu().numpy()
        separate_mask = np.where(separate_mask > 0)[0]
        if len(separate_mask) == 0:
            separate_mask = np.array([0])


        actions = np.array([it for it in file['data'][name]['actions']])[separate_mask]
        dones = np.array([it for it in file['data'][name]['dones']])[separate_mask]
        rewards = np.array([it for it in file['data'][name]['rewards']])[separate_mask]
        states = np.array([it for it in file['data'][name]['states']])[separate_mask]
        next_obs, obs = {}, {}
        for key in file['data'][name]['next_obs'].keys():
            content = np.array([it for it in file['data'][name]['next_obs'][key]])[separate_mask]
            next_obs[key] = content
        for key in file['data'][name]['obs'].keys():
            content = np.array([it for it in file['data'][name]['obs'][key]])[separate_mask]
            obs[key] = content

        remove_num = file['data'][name]['actions'].shape[0] - actions.shape[0]
        print('remove {} observations from demo {}'.format(remove_num, name))

        name_save = file2['data'][name]['actions'].name
        del file2[name_save]
        dset = file2.create_dataset(name_save, data=actions)

        name_save = file2['data'][name]['dones'].name
        del file2[name_save]
        dset = file2.create_dataset(name_save, data=dones)

        name_save = file2['data'][name]['rewards'].name
        del file2[name_save]
        dset = file2.create_dataset(name_save, data=rewards)

        name_save = file2['data'][name]['states'].name
        del file2[name_save]
        dset = file2.create_dataset(name_save, data=states)

        for key in file['data'][name]['next_obs'].keys():
            name_save = file['data'][name]['next_obs'][key].name
            del file2[name_save]
            dset = file2.create_dataset(name_save, data=next_obs[key])

        for key in file['data'][name]['obs'].keys():
            name_save = file['data'][name]['obs'][key].name
            del file2[name_save]
            dset = file2.create_dataset(name_save, data=obs[key])

        file2['data'][name].attrs["num_samples"] = actions.shape[0]
