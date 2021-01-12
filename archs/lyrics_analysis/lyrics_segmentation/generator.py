from archs.lyrics_analysis.lyrics_segmentation.utils.model import LyricsSegmentationModel
from pipelines.data_processing.configs import pop_lyrics_analysis_config
from common.func import ssm
import torch
import numpy as np

class Segmentator(object):
    def __init__(self,
    config=pop_lyrics_analysis_config,
    checkpoint_path = "./models/lyrics_analysis/checkpoints/12_05_2020_04-36-51.pt",
    cuda_index=0):
        #create model
        self._model = LyricsSegmentationModel(config,mode='not_train')
        self._model.load_state_dict(torch.load(checkpoint_path,map_location='cuda:{}'.format(cuda_index)))
        self._model.to('cuda:{}'.format(cuda_index))
        self.min_ssm_size = config.min_ssm_size
        self.max_ssm_size = config.max_ssm_size
        self.w_size = config.w_size
        self.cuda_index = cuda_index

    def predict(self,lyrics):
        #1. Convert to ssm
        ssm_input = ssm(lyrics)
        x = []

        max_size = ssm_input.shape[0]
        if ssm_input.shape[0] > self.max_ssm_size:
            ssm_input = ssm_input[:self.max_ssm_size,:self.max_ssm_size]
            max_size = self.max_ssm_size
        if ssm_input.shape[0] < self.min_ssm_size:
            return None
        if ssm_input.shape[0] < self.max_ssm_size:
            pad_value = self.max_ssm_size - ssm_input.shape[0]
            ssm_input = np.pad(ssm_input,((0,pad_value),(0,pad_value)),mode='constant')

        for index in range(max_size):
            line = []
            line_range = range(index - self.w_size + 1,index + self.w_size + 1)
            for i in line_range:
                if i < 0:
                    result_line = np.zeros(self.max_ssm_size)
                else:
                    result_line = ssm_input[index]
                line.append(result_line)
            x.append(line)
        x = torch.tensor(x,device=torch.device('cuda:{}'.format(self.cuda_index)))

        y = self._model(x)

        #find maximum arg on dim
        y = torch.argmax(y,dim=1)

        #return numpy list of index whose sentence is the boundary of segment
        indices =  (y == 1).nonzero().squeeze(1).cpu().numpy()


        #heuristic: merge close index to prevent too much boundaries
        #remove index 0 and last index
        if indices[0] == 0:
            indices = np.delete(indices,0)
        if indices[-1] == max_size - 1:
            indices = np.delete(indices,-1)

        result = []
        group_list = []
        for idx in indices:
            if len(group_list) == 0 or (len(group_list) > 0 and group_list[-1] + 1 == idx):
                group_list.append(idx)
            else:
                merge_idx = group_list[int(len(group_list) / 2)]
                result.append(merge_idx)
                group_list = []
                group_list.append(idx)

        if len(group_list) > 0:
            result.append(group_list[int(len(group_list) / 2)])

        if result is None:
            return []
        else:
            return result
