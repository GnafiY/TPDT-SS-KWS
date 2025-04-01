import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.enh.kws_encoder.aux_encoder import EcapaTdnn
from espnet2.enh.kws_encoder.conformer import Conformer
from espnet2.asr.frontend.default import DefaultFrontend

class conformer_fusioner(nn.Module):
    def __init__(self, n_fft=512, hidden_dim=128, fusion_strategy='concat', condition = None):
        super().__init__()

        assert fusion_strategy in ['text-only', 'audio-only', 'append', 'FiLM', 'concat', 'concat-avgpooling']
        self.fusion_strategy = fusion_strategy

        # text-only, we only need a trainable vector
        if fusion_strategy == 'text-only':
            self.cond = nn.Parameter(torch.rand(1,hidden_dim))
        else:
            assert condition is not None, 'Except for text-only, other fusion strategies require a non-blank condition scp file.'
            # 1. extract wav conditions
            wavs = SoundScpReader(condition)
            frontend = DefaultFrontend(n_fft=n_fft)
            raws = []
            raw_lens = []
            for k in wavs:
                fs, v = wavs[k]
                raws.append(torch.tensor(v,dtype=torch.float32))
                raw_lens.append(len(v))
                # out, lens = stft(v)
            raws = pad_list(raws,0)
            raw_lens = torch.tensor(raw_lens,dtype = torch.long)
            fbank, f_lens = frontend(raws,raw_lens)
            max_length = torch.max(f_lens)
            mask = torch.zeros(f_lens.size(0),max_length, dtype=torch.float32)
            for i, length in enumerate(f_lens):
                mask[i, :length] = 1.0
            self.cond = fbank
            self.mask = mask

            # 2. initialize aux_encoder
            self.aux_encoder = EcapaTdnn(80,embd_dim=hidden_dim)
            self.fuse_module = Conformer()

            # 3. setup fusion_strategy
            if self.fusion_strategy.startswith('concat'):
                logging.info("10 + 1 -> 10 * 256 -> 10 * 128")  # concat + avg-poling
                self.text_cond = nn.Parameter(torch.rand(1,hidden_dim))
                self.proj = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                )
            elif self.fusion_strategy == 'FiLM':
                self.linear_layer_w = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(),
                )
                self.linear_layer_b = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(),
                )
    


    def forward(self,input):
        if self.fusion_strategy == 'text-only':
            condition = self.cond
        else:
            # 1. keep the raw waves unchanged, wav -> condition embeddings
            condition = torch.tensor(self.cond, device=input.device, requires_grad=False)
            masks = torch.tensor(self.mask,device=input.device,requires_grad=False)
            condition = self.aux_encoder(condition,masks)
            # here, we have already completed condition calculation of 'audio-only' fusion

            # 2. execute fusion_strategy
            if self.fusion_strategy.startswith('concat'):
                condition = torch.cat((condition, self.text_cond.repeat(condition.shape[0], 1)), dim=1)
                condition = self.proj(condition)
                if self.fusion_strategy == 'concat-avgpooling':
                    condition = torch.mean(condition, dim=0).unsqueeze(0)
            elif self.fusion_strategy == 'FiLM':
                text_cond_w = self.linear_layer_w(self.text_cond)
                text_cond_b = self.linear_layer_b(self.text_cond)
                condition = text_cond_w * condition + text_cond_b
            elif self.fusion_strategy == 'append':
                condition = torch.cat((self.text_cond, condition))

        output = self.fuse_module(input, condition)
        
        return output

def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad                
                
                
if __name__ == '__main__':
    # model = fusion_model()
    model = conformer_fusioner(condition=None)
    x = torch.rand(10,123,128)
    model(x)
    
        

# # original
# class conformer_fusioner(nn.Module):
#     def __init__(self,n_fft=512, hop_length=256, feature_dim=256, hidden_dim=128, condition = None):
#         super().__init__()
#         # condition Encoder
#         if condition == None or condition == 'None':
#             self.vector_condition = True
#             self.cond = nn.Parameter(torch.rand(1,hidden_dim))
#         else:
#             self.vector_condition = False
#             wavs = SoundScpReader(condition)
#             # kaldiio.load_wav
#             stft = Stft(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=False)
#             frontend = DefaultFrontend(n_fft=n_fft)
#             logmel = LogMel(n_fft=n_fft)
#             raws = []
#             raw_lens = []
#             for k in wavs:
#                 fs, v = wavs[k]
#                 raws.append(torch.tensor(v,dtype=torch.float32))
#                 raw_lens.append(len(v))
#                 # out, lens = stft(v)
#             raws = pad_list(raws,0)
#             raw_lens = torch.tensor(raw_lens,dtype = torch.long)
#             fbank, f_lens = frontend(raws,raw_lens)
#             max_length = torch.max(f_lens)
#             mask = torch.zeros(f_lens.size(0),max_length, dtype=torch.float32)
#             # 根据长度信息设置 mask，对应序列长度之前的部分设为 1，之后的部分设为 0
#             for i, length in enumerate(f_lens):
#                 mask[i, :length] = 1.0
#             # st = stft(raws)[0]
#             # st = ComplexTensor(st[..., 0],st[..., 1])
#             # st = st.real**2 + st.imag**2
#             # st = abs(st)
#             # # import pdb; pdb.set_trace()
#             # self.cond = logmel(st)[0]
#             self.cond = fbank
#             self.mask = mask
#             self.aux_encoder = EcapaTdnn(80,embd_dim=hidden_dim)
        
#         # fusion module
#         self.fuse_module = conformer_fusion()
        
#     def forward(self,input):
#         if self.vector_condition:
#             condition = self.cond
#         else:
#             #keep the raw waves unchanged
#             condition = torch.tensor(self.cond, device=input.device, requires_grad=False)
#             masks = torch.tensor(self.mask,device=input.device,requires_grad=False)
#             condition = self.aux_encoder(condition,masks)

#         output = self.fuse_module(input,condition)
        
#         return output
#         # return output, ilens

# 10 + 1 -> 10 * 256 -> 10 * 128

# # FiLM
# class conformer_fusioner(nn.Module):
#     def __init__(self,n_fft=512, hop_length=256, feature_dim=256, hidden_dim=128, condition = None):
#         super().__init__()
#         # condition Encoder
#         if condition == None or condition == 'None':
#             self.vector_condition = True
#             self.cond = nn.Parameter(torch.rand(1,hidden_dim))
#         else:
#             self.vector_condition = False
#             wavs = SoundScpReader(condition)
#             # kaldiio.load_wav
#             stft = Stft(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=False)
#             frontend = DefaultFrontend(n_fft=n_fft)
#             logmel = LogMel(n_fft=n_fft)
#             raws = []
#             raw_lens = []
#             for k in wavs:
#                 fs, v = wavs[k]
#                 raws.append(torch.tensor(v,dtype=torch.float32))
#                 raw_lens.append(len(v))
#                 # out, lens = stft(v)
#             raws = pad_list(raws,0)
#             raw_lens = torch.tensor(raw_lens,dtype = torch.long)
#             fbank, f_lens = frontend(raws,raw_lens)
#             max_length = torch.max(f_lens)
#             mask = torch.zeros(f_lens.size(0),max_length, dtype=torch.float32)
#             # 根据长度信息设置 mask，对应序列长度之前的部分设为 1，之后的部分设为 0
#             for i, length in enumerate(f_lens):
#                 mask[i, :length] = 1.0
#             # st = stft(raws)[0]
#             # st = ComplexTensor(st[..., 0],st[..., 1])
#             # st = st.real**2 + st.imag**2
#             # st = abs(st)
#             # # import pdb; pdb.set_trace()
#             # self.cond = logmel(st)[0]
#             self.cond = fbank
#             self.mask = mask
#             self.aux_encoder = EcapaTdnn(80,embd_dim=hidden_dim)

#             # 10 + 1 -> 10 * 256 -> 10 * 128
#             # 20230215 appended
#             logging.info("FiLM")
#             self.text_cond = nn.Parameter(torch.rand(1,hidden_dim))
#             self.linear_layer_w = nn.Sequential(
#                 nn.Linear(128, 128),
#                 nn.ReLU(),
#             )
#             self.linear_layer_b = nn.Sequential(
#                 nn.Linear(128, 128),
#                 nn.ReLU(),
#             )
    
#         # fusion module
#         self.fuse_module = conformer_fusion()
        
#     def forward(self,input):
#         if self.vector_condition:
#             condition = self.cond
#         else:
#             #keep the raw waves unchanged
#             condition = torch.tensor(self.cond, device=input.device, requires_grad=False)
#             masks = torch.tensor(self.mask,device=input.device,requires_grad=False)
#             condition = self.aux_encoder(condition,masks)
            
#             text_cond_w = self.linear_layer_w(self.text_cond)
#             text_cond_b = self.linear_layer_b(self.text_cond)
            
#             condition = text_cond_w * condition + text_cond_b

#         output = self.fuse_module(input,condition)
        
#         return output
#         # return output, ilens

# # 10 + 1 -> 11
# class conformer_fusioner(nn.Module):
#     def __init__(self,n_fft=512, hop_length=256, feature_dim=256, hidden_dim=128, condition = None):
#         super().__init__()
#         # condition Encoder
#         if condition == None or condition == 'None':
#             self.vector_condition = True
#             self.cond = nn.Parameter(torch.rand(1,hidden_dim))
#         else:
#             # 20230215 appended
#             self.text_cond = nn.Parameter(torch.rand(1,hidden_dim))
            
#             self.vector_condition = False
#             wavs = SoundScpReader(condition)
#             # kaldiio.load_wav
#             stft = Stft(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=False)
#             frontend = DefaultFrontend(n_fft=n_fft)
#             logmel = LogMel(n_fft=n_fft)
#             raws = []
#             raw_lens = []
#             for k in wavs:
#                 fs, v = wavs[k]
#                 raws.append(torch.tensor(v,dtype=torch.float32))
#                 raw_lens.append(len(v))
#                 # out, lens = stft(v)
#             raws = pad_list(raws,0)
#             raw_lens = torch.tensor(raw_lens,dtype = torch.long)
#             fbank, f_lens = frontend(raws,raw_lens)
#             max_length = torch.max(f_lens)
#             mask = torch.zeros(f_lens.size(0),max_length, dtype=torch.float32)
#             # 根据长度信息设置 mask，对应序列长度之前的部分设为 1，之后的部分设为 0
#             for i, length in enumerate(f_lens):
#                 mask[i, :length] = 1.0
#             # st = stft(raws)[0]
#             # st = ComplexTensor(st[..., 0],st[..., 1])
#             # st = st.real**2 + st.imag**2
#             # st = abs(st)
#             # # import pdb; pdb.set_trace()
#             # self.cond = logmel(st)[0]
#             self.cond = fbank
#             self.mask = mask
#             self.aux_encoder = EcapaTdnn(80,embd_dim=hidden_dim)
        
#         # fusion module
#         self.fuse_module = conformer_fusion()
        
#     def forward(self,input):
#         if self.vector_condition:
#             condition = self.cond
#         else:
#             #keep the raw waves unchanged
#             condition = torch.tensor(self.cond, device=input.device, requires_grad=False)
#             masks = torch.tensor(self.mask,device=input.device,requires_grad=False)
#             condition = self.aux_encoder(condition,masks)
            
#             # append text condition
#             condition = torch.cat((self.text_cond, condition))

#         output = self.fuse_module(input,condition)
        
#         return output
#         # return output, ilens
 
#     def forward(self,input):
#         if self.vector_condition:
#             condition = self.cond
#         else:
#             #keep the raw waves unchanged
#             condition = torch.tensor(self.cond, device=input.device, requires_grad=False)
#             masks = torch.tensor(self.mask,device=input.device,requires_grad=False)
#             condition = self.aux_encoder(condition,masks)
            
#             # append text condition
#             condition = torch.cat((self.text_cond, condition))
            

#         output = self.fuse_module(input,condition)
        
#         return output
#         # return output, ilens
           