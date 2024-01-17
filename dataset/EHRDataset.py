import torch
import torch.utils.data as data
import numpy as np


class EhrDataset(data.Dataset):
    def __init__(self, data_x_nor, data_x_nor_mask, data_y, data_pid, task_name, fill_methods, sample_ratio=1):
        super().__init__()
        self.data_x_nor = data_x_nor
        self.data_y = data_y
        self.data_pid = data_pid
        self.data_x_nor_mask = data_x_nor_mask
        self.task_name = task_name

        # print(len(self.data_x_nor), len(self.data_y), len(self.data_pid), len(self.data_x_nor_mask))

        if sample_ratio != 1:
            np.random.seed(2024)
            label = [self.data_y[index][-1][0] for index in range(len(data_y))]
            # 分层抽样
            sample_size_0 = int((len(label)-np.sum(label)) * sample_ratio)
            sample_size_1 = int(np.sum(label) * sample_ratio)
            sampled_indices_0 = np.random.choice(np.where(np.array(label) == 0)[0], size=sample_size_0, replace=False)
            sampled_indices_1 = np.random.choice(np.where(np.array(label) == 1)[0], size=sample_size_1, replace=False)

            # 获取抽样后的数据
            self.data_x_nor = [self.data_x_nor[i] for i in sampled_indices_0] + [self.data_x_nor[i] for i in sampled_indices_1]
            self.data_y = [self.data_y[i] for i in sampled_indices_0] + [self.data_y[i] for i in sampled_indices_1]
            self.data_pid = [self.data_pid[i] for i in sampled_indices_0] + [self.data_pid[i] for i in sampled_indices_1]
            self.data_x_nor_mask = [self.data_x_nor_mask[i] for i in sampled_indices_0] + [self.data_x_nor_mask[i] for i in sampled_indices_1]
        # print(len(self.data_x_nor), len(self.data_y), len(self.data_pid), len(self.data_x_nor_mask))

        if fill_methods != '':
            for idx, data_sample in enumerate(data_x_nor):
                data_sample_mask = data_x_nor_mask[idx]
                data_sample = np.array(data_sample)
                data_sample_mask = np.array(data_sample_mask)
                if np.sum(data_sample_mask) != 0:
                    data_sample_ori_shape = data_sample.shape
                    data_sample[data_sample_mask==1] = np.nan
                    data_sample = data_sample[np.newaxis, :, :]
                    print(data_sample.shape)
                    if data_sample_ori_shape[0] == 1:
                        data_sample = np.repeat(data_sample, repeats=2, axis=1)
                    dataset = {"X": data_sample}
                    if fill_methods == 'SAITS':
                        from pypots.imputation import SAITS
                        imputation_model = SAITS(n_steps=data_sample.shape[1], n_features=data_sample.shape[2], n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
                    if fill_methods == 'TimesNet':
                        from pypots.imputation import TimesNet
                        imputation_model = TimesNet(n_steps=data_sample.shape[1], n_features=data_sample.shape[2], n_layers=2, top_k=1, d_model=128, d_ffn=256, n_kernels=3, dropout=0.1, epochs=10)
                    if fill_methods == 'CSDI':
                        from pypots.imputation import CSDI
                        imputation_model = CSDI(n_features=data_sample.shape[2], n_layers=1, n_channels=8, d_time_embedding=32, d_feature_embedding=3, d_diffusion_embedding=32, n_diffusion_steps=10, n_heads=1, epochs=10)
                    if fill_methods == 'USGAN':
                        from pypots.imputation import USGAN
                        imputation_model = USGAN(n_steps=data_sample.shape[1], n_features=data_sample.shape[2], rnn_hidden_size=256, epochs=10)
                    if fill_methods == 'GPVAE':
                        from pypots.imputation import GPVAE
                        imputation_model = GPVAE(n_steps=data_sample.shape[1], n_features=data_sample.shape[2], latent_size=256, epochs=10)
                    imputation_model.fit(dataset)
                    data_sample = imputation_model.predict(dataset)["imputation"].reshape(dataset['X'].shape)
                    if data_sample_ori_shape[0] == 1:
                        data_sample = data_sample[:, 0, :]
                    if data_sample.shape != data_sample_ori_shape:
                        data_sample = data_sample.reshape(data_sample_ori_shape)
                data_x_nor[idx] = data_sample

    def __len__(self):
        return len(self.data_y) # number of patients

    def __getitem__(self, index):
        # 这里的y是每一次visit都有一个y，取最后一个y作为最终预测
        # 这里的y有多个，代表多个任务，第1个任务，outcome，第2个任务，los
        if self.task_name == 'outcome':
            return self.data_x_nor[index], self.data_x_nor_mask[index], self.data_y[index][-1][0], self.data_pid[index]
        elif self.task_name == 'los':
            return self.data_x_nor[index], self.data_x_nor_mask[index], self.data_y[index][-1][1], self.data_pid[index]


def pad_collate(batch):
    data_x_nor, data_x_nor_mask, data_y, data_pid = zip(*batch)
    lens = torch.as_tensor([len(x) for x in data_x_nor])
    # convert to tensor
    data_x_nor = [torch.tensor(x_nor) for x_nor in data_x_nor]
    data_x_nor_mask = [torch.tensor(x_nor_mask) for x_nor_mask in data_x_nor_mask]
    data_y = torch.tensor([torch.tensor(y) for y in data_y])
    data_x_nor_pad = torch.nn.utils.rnn.pad_sequence(data_x_nor, batch_first=True, padding_value=0)
    data_x_nor_mask_pad = torch.nn.utils.rnn.pad_sequence(data_x_nor_mask, batch_first=True, padding_value=0)
    return data_x_nor_pad.float(), data_x_nor_mask_pad.float(), data_y.float(), lens, data_pid