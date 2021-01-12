import os, sys
import glob
import numpy as np
import torch
import pickle 
import random 
random.seed(1000)

class ViDataLoader(object):
    def __init__(self, data_path, bsz, bptt, vocab, split_rate=0.8, device='cpu', ext_len=None, shuffle=True):
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.seq_len = bptt
        self.device = device
        self.vocab = vocab
        self.shuffle = shuffle        
        self.split_rate = split_rate
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f) # data format [num_sample, seq_len], default seq_len = 1024

    def _prepare_batches(self, batch_size=None):
        """
            prepare global batches from [samples, seq_len (with padding)] to [seq_len, bsz]
        """
        global_batches = []
        batch_size = self.bsz if batch_size == None else batch_size
        self.bsz = batch_size
        # Process data loaded from pickle file
        # 
        for sample in self.data:
            try:
                last_idx = np.where(sample==self.vocab.pad_idx)[0][0]
                # import pdb; pdb.set_trace()
            except:
                last_idx = sample.shape[0]
            if self.seq_len >= last_idx:
                continue
            else:
                for i in range(0, last_idx-self.seq_len-1):
                    x = sample[i:self.seq_len+i]
                    y = sample[i+1:self.seq_len+i+1]
                    global_batches.append([x,y])
        # shuffle data for training or not. Recommend shuffle == True
        if self.shuffle:
            random.shuffle(global_batches)       
        # convert to LongTensor
        self.global_batches = torch.LongTensor(global_batches)
        # delete variables for saving mem
        del global_batches
        
        self.n_step = self.global_batches.size(0) // batch_size
       

        # Trim off extra element that not cleanly fit with batch_size
        self.global_batches = self.global_batches.narrow(0, 0, self.n_step * batch_size)
        self.global_batches = self.global_batches.view(self.n_step, batch_size, 2, self.seq_len)
        # split train and test data batches base on split rate
        self.train_data = self.global_batches[:int(self.split_rate*self.n_step)]
        self.test_data = self.global_batches[int(self.split_rate*self.n_step):]
        self.train_step = self.train_data.size(0)
        self.eval_step = self.test_data.size(0)

    def get_batch(self, i, mode):
        if not hasattr(self, 'global_batches'):
            self._prepare_batches()
        if mode == 'train':
            mini_batch = self.train_data[i].permute(1,0,2) # permute from [bsz,'x,y', seq_len] -> ['x,y', bsz, seq_len]
        else: 
            mini_batch = self.test_data[i].permute(1,0,2)
        inp = mini_batch[0].t().contiguous().to(self.device)
        tgt = mini_batch[1].t().contiguous().to(self.device)
        # import pdb; pdb.set_trace()
        return inp, tgt, tgt.size(0)
    
    def _get_train_iter(self):
        for i in range(0, self.train_step):
            yield self.get_batch(i, mode = 'train')
    
    def _get_eval_iter(self):
        for i in range(0, self.eval_step):
            yield self.get_batch(i, mode = 'eval')

    def get_fixlen_iter(self, start=0, mode='train'):
        if mode == 'train':
            return self._get_train_iter()
        else:
            return self._get_eval_iter()

    # def __iter__(self):
    #     return self.get_fixlen_iter()
class ViDataLoaderV2(ViDataLoader):
    def __init__(self, data_path, bsz, bptt, vocab, split_rate=0.8, device='cpu', ext_len=None, shuffle=True):
        super().__init__(data_path, bsz, bptt, vocab, split_rate, device, ext_len, shuffle) 
        # data format [num_sample, seq_len], default seq_len = 1024

    def _prepare_batches(self, batch_size=None):
        """
            prepare global batches from [samples, seq_len] to [seq_len]
        """
        global_batches = []
        batch_size = self.bsz if batch_size == None else batch_size
        self.bsz = batch_size
        # Process data loaded from pickle file
        # 
        for sample in self.data:
            try:
                last_idx = np.where(sample==self.vocab.pad_idx)[0][0]
                # import pdb; pdb.set_trace()
            except:
                last_idx = sample.shape[0]
            if self.seq_len >= last_idx:
                continue
            else:
                for i in range(0, last_idx-self.seq_len-1, self.seq_len):
                    x = sample[i: i + self.seq_len]
                    y = sample[i+1:i+1 + self.seq_len]
                    global_batches.append([x,y])
        # shuffle data for training or not. Recommend shuffle == True
        if self.shuffle:
            random.shuffle(global_batches)       
        # convert to LongTensor
        self.global_batches = torch.LongTensor(global_batches)
        # delete variables for saving mem
        del global_batches
        
        self.n_step = self.global_batches.size(0) // batch_size
       

        # Trim off extra element that not cleanly fit with batch_size
        self.global_batches = self.global_batches.narrow(0, 0, self.n_step * batch_size)
        self.global_batches = self.global_batches.view(self.n_step, batch_size, 2, self.seq_len)
        # split train and test data batches base on split rate
        self.train_data = self.global_batches[:int(self.split_rate*self.n_step)]
        self.test_data = self.global_batches[int(self.split_rate*self.n_step):]
        self.train_step = self.train_data.size(0)
        self.eval_step = self.test_data.size(0)
        

class ViDataLoaderSyllable(ViDataLoader):
    def __init__(self,data_path,syll_data_path,bsz, bptt, syll_dim, vocab, split_rate=0.8, device='cpu', ext_len=None, shuffle=True):
        super(ViDataLoaderSyllable,self).__init__(data_path, bsz, bptt, vocab, split_rate, device, ext_len, shuffle)

        with open(syll_data_path,'rb') as f:
            self.syll_data = pickle.load(f)

        self.syll_dim = syll_dim

    def _prepare_batches(self,batch_size=None):
        """
            prepare global batches from [samples, seq_len] to [seq_len]
        """
        global_batches = []
        batch_size = self.bsz if batch_size == None else  batch_size
        self.bsz = batch_size
        #process data
        for t,sample in enumerate(self.data):
            syll_idx = 0
            try:
                last_idx = np.where(sample==self.vocab.pad_idx)[0][0]
            except:
                last_idx = sample.shape[0]

            if self.seq_len >= last_idx:
                continue
            else:
                for i in range(0,last_idx - self.seq_len - 1,self.seq_len):
                    x = np.zeros((self.seq_len,self.syll_dim + 1))
                    y = np.zeros((self.seq_len,self.syll_dim + 1))
                    #concatenate embedding vector
                    for idx in range(i,i + self.seq_len - 1):
                        if self.syll_data[t].shape[0] > syll_idx:
                            if self.syll_data[t][syll_idx][0] == idx:
                                event = np.concatenate([np.array([sample[idx]]),self.syll_data[t][syll_idx][1]])
                                syll_idx = syll_idx + 1
                            else:
                                event = np.concatenate([np.array([sample[idx]]),np.zeros(self.syll_dim)])
                        else:
                            event = np.concatenate([np.array([sample[idx]]),np.zeros(self.syll_dim)])
                        x[idx - i] = event
                        #FIXME: Unused embedding vector on label
                        y_event = np.concatenate([np.array([sample[idx + 1]]),np.zeros(self.syll_dim)])
                        y[idx + 1 - i] = y_event
                    global_batches.append([x,y])
        #shuffle data
        if self.shuffle:
            random.shuffle(global_batches)
        self.global_batches = torch.FloatTensor(global_batches)
        del global_batches
        self.n_step = self.global_batches.size(0) // batch_size

        # Trim off extra element that not cleanly fit with batch_size
        self.global_batches = self.global_batches.narrow(0, 0, self.n_step * batch_size)
        self.global_batches = self.global_batches.view(self.n_step, batch_size, self.syll_dim + 1,2, self.seq_len)
        # split train and test data batches base on split rate
        self.train_data = self.global_batches[:int(self.split_rate*self.n_step)]
        self.test_data = self.global_batches[int(self.split_rate*self.n_step):]
        self.train_step = self.train_data.size(0)
        self.eval_step = self.test_data.size(0)

    def get_batch(self, i, mode):
        if not hasattr(self, 'global_batches'):
            self._prepare_batches()
        if mode == 'train':
            mini_batch = self.train_data[i].permute(2,0,3,1) # permute from [bsz,'x,y', seq_len] -> ['x,y', bsz, seq_len]
        else: 
            mini_batch = self.test_data[i].permute(2,0,3,1)
        inp = mini_batch[0].permute(1,0,2).contiguous().to(self.device)
        tgt = mini_batch[1].permute(1,0,2).contiguous().to(self.device)
        return inp, tgt, tgt.size(0)
