from torch.utils.data import DataLoader
import torch
import numpy as np
class LyricsDataLoader(object):
    def __init__(self,dataset,config):
        self._dataset = dataset
        self._genre = self._dataset.genre
        # self._dataloader = DataLoader(self._dataset, batch_size=config.hparams.batch_size,
        # shuffle=config.train_shuffle, num_workers=1)
        self.config = config

    def __iter__(self):
        #overwriting iterator for data loader
        #return a tuple of tensor
        #first element: input.shape == [batch_size,2 * w_size, max_ssm_size]
        #second element : label.shape == [batch_size,2]
        # import pdb; pdb.set_trace()
        # for sample in self._dataloader:
        #     #padding input and label
        #     x = torch.tensor(np.pad(s['input'], [(0, self.config.max_ssm_size - s['input'].shape[0]), (0, 0),[0,0]], mode='constant') for s in sample)
        #     y = torch.tensor(np.pad(s['label'], [(0, self.config.max_ssm_size - s['input'].shape[0])], mode='constant') for s in sample)
        #     yield x,y
        for i in range(0,len(self._dataset) - self.config.hparams.batch_size,self.config.hparams.batch_size):
            sample = []
            for t in range(i,i + self.config.hparams.batch_size):
                sample.append(self._dataset[t])
            x = torch.tensor([np.pad(s['input'], [(0, self.config.max_ssm_size - s['input'].shape[0]), (0, 0),[0,0]], mode='constant') for s in sample])
            y = torch.tensor([np.pad(s['label'], [(0, self.config.max_ssm_size - s['label'].shape[0]),(0,0)], mode='constant') for s in sample])
            yield x,y

    def __len__(self):
        return len(self._dataset)