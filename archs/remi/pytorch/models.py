import torch  
from torch import nn
import torch.nn.functional as F
from archs.remi.pytorch.modules import \
RelPartialLearnableDecoderLayer,AdaptiveEmbedding, RelLearnableDecoderLayer, DecoderLayer, ProjectedAdaptiveLogSoftmax, PositionalEmbedding
from archs.remi.pytorch.utils.log_uniform_sampler import sample_logits, LogUniformSampler
import numpy as np 
from common.music_item import ItemProcessor
from common.object import Note
from itertools import cycle as itercycle
from common.constants import CHORD
import random
import copy
import sys
class Transformer(nn.Module):
    """
        max_seq: max sequence length
        num_vocab: vocab size
        n_layer: number of relative attention layer
        d_model: length of input sequence
        d_embed: embedding dimension of each token in sequence
        n_head: number of head in multi head attention
        d_head: dimensions of each head in multi attention head 
        d_ff: output dimensions of linear layer

    """
    def __init__(self, num_vocab, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1, restart = False):
        super(Transformer, self).__init__()
        
        
        self.n_token = num_vocab
        self.d_embed = d_embed if d_embed is not None else d_model
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(num_vocab, d_embed, d_model, cutoffs, 
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, num_vocab)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(num_vocab, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(num_vocab, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and (d_model) != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def __set_param(self, **kwargs):
        for k, v in kwargs:
            self.__setattr__(k, v)

    def _save_param(self):
        pass

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        """
            Update mem for relative attention for extended context
        """
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)

        
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]

        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        core_out = self.drop(core_out)
        
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    # def __load_config(self, **kwargs):
    #     for k, v in kwargs.item():
    #         self.__setattr__(self, k, v)

    def forward(self, data, target, *mems):
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.
        if not mems: mems = self.init_mems()
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems

    def _temp_sampling(self, logits, temp, topk = 5):
        if topk == 1:
            probs = torch.exp(logits[-1] / temp) / torch.sum(torch.exp(logits[-1] / temp))
            pred_token = int(torch.argmax(probs, dim=-1))
        else: 
            probs = torch.exp(logits[-1] / temp) / torch.sum(torch.exp(logits[-1] / temp))
            candi_values, candi_indexes = torch.topk(probs,k=topk)
            # choose by predicted probs
            candi_indexes = candi_indexes.to('cpu').numpy()
            pred_token = np.random.choice(candi_indexes)
        return pred_token


    def _replace_if_chord(self, pred_token, vocab, chord_progress):

        if chord_progress == []:
            return pred_token
        else:
            replace_chord = next(chord_progress)
            if '#' in replace_chord:
                pitch_name = replace_chord[:2]
                qual = replace_chord[2:]
            else:
                pitch_name = replace_chord[:1]
                qual = replace_chord[1:]
            pred_token = vocab.chord_idx(CHORD.PITCH_NAME.index(pitch_name), qual)
            return pred_token

    def generate(self,vocab,n_syll,primer_tokens = [],temperature_range = (0.5,1.0),beam_width=4,look_back=10,device='cuda:0',**kwargs):
        #Initialize parameters
        mems = self.init_mems()
        self.eval()

        torch.manual_seed(10000)
        np.random.seed(10000)

        #Initialize primer melodies and count number of syllables
        if len(primer_tokens) == 0:
            initial_num_syllable = 0
            #primer_tokens not exists
            primer = []
            primer.append(vocab.stoi['xxbar'])
            primer.append(vocab.stoi['p0'])
            chord_idx = np.random.choice(vocab.chord_idx_range[-1])
            primer.append(chord_idx)
            full_tokens = primer.copy()
            cur_predicted_bar_tokens = primer.copy()
            primer = torch.tensor([primer])
            primer = primer.view(-1,1).to(device)
        else: 
            #Check that bar_idx always occurs first
            if primer_tokens[0] != vocab.stoi['xxbar']:
                primer_tokens = [vocab.stoi['xxbar']] + primer_tokens
            #find current number of syllable
            initial_num_syllable = 0
            #find last occurence of bar index
            last_bar_idx = len(primer_tokens) - primer_tokens[::-1].index(vocab.stoi['xxbar']) - 1
            if last_bar_idx > 0:
                bar_seq = ItemProcessor.tokens_to_barsequence(np.asarray(primer_tokens[:last_bar_idx + 1]),vocab)
                for bar in bar_seq:
                    timestep = [x.start for x in bar if isinstance(x,Note)]
                    initial_num_syllable += len(set(timestep))
            #copy full tokens
            primer = torch.tensor([primer_tokens])
            primer = primer.view(-1,1).to(device)
        
        #first pass: choosing the top[beam_width] result, using the last element
        try:
            hidden, _ = self._forward(torch.tensor([[primer[-1]]]).to(device), mems=mems)
        except:
            # import pdb ; pdb.set_trace()
            pass
        logits = self.crit._compute_logit(hidden.view(-1, hidden.size(-1)), 
        self.crit.out_layers[0].weight, self.crit.out_layers[0].bias, self.crit.out_projs[0])[-1]
        top_beam_value,top_beam_indices = torch.topk(logits,beam_width)

        #it is a list of dictionary containing a dictionary
        beam_sequences = []
        for idx,val in zip(top_beam_indices,top_beam_value):
            if idx != 0:
                beam_sequences.append({
                    "sequence" : torch.cat((primer, torch.tensor([[idx]]).to(device))),
                    "score" : val,
                    "num_syll" : initial_num_syllable}
                )
            elif initial_num_syllable == 0: #if primer melody does not end with 0 AND idx == 0
                new_primer = torch.cat((primer, torch.tensor([[idx]]).to(device)))
                bar_seq = ItemProcessor.tokens_to_barsequence(new_primer.squeeze().cpu().numpy(),vocab)
                count = len(set([x.start for x in bar_seq[-1] if isinstance(x,Note)]))
                beam_sequences.append({
                    "sequence" :new_primer,
                    "score" : val,
                    "num_syll" : count}
                )

        #Iterate until a beam sequence has enough number of syllable
        while all([beam["num_syll"] - initial_num_syllable < n_syll for beam in beam_sequences]):
            #for each beam sequence, predict softmax of new token whose shape is (1,1)
            candidates = []
            #for each beam
            for beam in beam_sequences:
                if beam["num_syll"] - initial_num_syllable < n_syll:
                    with torch.no_grad():
                        #get last value to prevent 2 consecutive bar
                        last_value = beam['sequence'][-1][0]
                        #calculate new token
                        back = look_back if beam['sequence'].shape[0] > look_back else beam['sequence'].shape[0]
                        hidden, _ = self._forward(beam['sequence'][-back : ], mems=mems)
                        logits = self.crit._compute_logit(hidden.view(-1, hidden.size(-1)), 
                        self.crit.out_layers[0].weight, self.crit.out_layers[0].bias, self.crit.out_projs[0])[-1]
                        #for each new candidate
                        for i, logit in enumerate(logits):
                            #skip for 2 consecutive bar
                            if last_value == 0 and i == 0:
                                continue
                            #concat sequence
                            concat_seq = torch.cat((beam['sequence'] , torch.tensor([[i]]).to(device)))
                            #distorting score of new logit
                            temperature = np.random.uniform(low=temperature_range[0], high=temperature_range[1]) #heuristic 
                            candidate = {
                                "sequence" : concat_seq,
                                "score" : beam['score'] + (logit / temperature),
                                "num_syll" : beam['num_syll']
                            }
                            #Check if new token if a bar -> update number of syllable
                            if i == vocab.bar_idx:
                                current = candidate['sequence'].squeeze().cpu().numpy()
                                bar_seq = ItemProcessor.tokens_to_barsequence(current, vocab)
                                timestep = [x.start for x in bar_seq[-1] if isinstance(x,Note)]
                                new_syll = candidate['num_syll'] + len(set(timestep))
                                candidate['num_syll'] = new_syll
                            candidates.append(candidate)
                else: #stop and keep storing in candidates
                    candidates.append(beam)
            #choosing the top k beam_sequence
            candidates = sorted(candidates,key = lambda x : x['score'])
            beam_sequences = candidates[-beam_width:][::-1] #decreasing
            # print([x['num_syll'] for x in beam_sequences])

        #Choosing most suitable sequence
        topk = -1
        min_syllable = float('inf')
        for i,beam in enumerate(beam_sequences):
            tokens = beam['sequence'].squeeze().cpu().numpy()
            if (tokens[-1] == np.asarray([vocab.stoi['xxbar']]))[0] == False:
                tokens = np.append(tokens,np.asarray([vocab.stoi['xxbar']]))
            #Check number of new syllable
            bar_seq = ItemProcessor.tokens_to_barsequence(tokens,vocab)
            total_num_syllable = 0
            for bar in bar_seq:
                total_num_syllable += len(set([x.start for x in bar if isinstance(x,Note)]))
            if total_num_syllable - initial_num_syllable >= n_syll:
                if total_num_syllable - initial_num_syllable < min_syllable:
                    min_syllable = total_num_syllable - initial_num_syllable
                    topk = i

        if topk == -1:
            import pdb ; pdb.set_trace()

        print("Returning sequence top {}".format(topk + 1))

        return tokens.tolist()





        
