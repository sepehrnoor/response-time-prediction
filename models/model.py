import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import regex as re
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import tensorflow as tf
import h5py
import copy
import time
import datetime
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.special import softmax
from copy import deepcopy
from sklearn.metrics import roc_curve, auc, roc_auc_score, r2_score, mean_absolute_error, accuracy_score
import sklearn.metrics as metrics
import seaborn
from torch.autograd import Variable
from pathlib import Path

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0)
        #print(pe.shape)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None, relation_matrix = None, q_trick = False, blending= 0.5 ):
    "Compute 'Scaled Dot Product Attention'"
    # Data is in the format (b,h,n,d_k)
    d_k = key.size(-1)
    n = key.size(-2)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
    # scores is in the shape (b, h, n, n)
    # mask is in the shape (b, h, n, n)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    p_attn = torch.where(torch.isnan(p_attn), torch.zeros_like(p_attn), p_attn)
    if relation_matrix is not None:
        p_attn = blending * p_attn + (1 - blending) * relation_matrix
        # scores = (relation_matrix + LAMBDA) * scores
    if mask is not None:
        p_attn = p_attn.masked_fill(~mask, 0.)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiheadAttention(nn.Module):
    def __init__(self, number_of_heads, dim_model, dropout=0.1, blending = 0.5):
        "Take in model size and number of heads."
        super(MultiheadAttention, self).__init__()
        assert dim_model % number_of_heads == 0
        # We assume d_v always equals d_k
        self.d_k = dim_model // number_of_heads
        self.h = number_of_heads
        self.linears = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.blending = blending
        
    def forward(self, query, key, value, mask=None, padding_mask=None, relation_matrix=None, q_trick = False):

        "Implements Figure 2"
        # if mask is not None:
        #     # Same mask applied to all h heads.
        #     mask = mask.unsqueeze(1)
        nbatches = key.size(0)
        seq_len = key.size(1)
        
        # right now the data is in the shape (b,n,d_model)
        # 1) Do all the linear projections in batch from d_model => h x d_k and then transposed to swap dim of h and n
        # q,k,v will each have shape (b, h, n, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        # mask has shape (n, n)
        # mask = mask.unsqueeze(0).unsqueeze(0).repeat(nbatches, self.h, 1, 1) # (n,n) -> (b,h,n,n)
        # padding mask has shape (b, n)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.h, 1, 1) # (b,n) -> (b,h,n,n)
        # data has shape (b, h, n, d_k)
        x, self.attn = attention(query, key, value,                       
                                 mask=padding_mask if q_trick else mask * padding_mask,
                                 dropout=self.dropout,
                                 relation_matrix=relation_matrix,
                                 q_trick = q_trick,
                                 blending = self.blending)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )
        

class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self , dim_model, heads_en, total_ex ,total_cat, seq_len, dropout, blending=0.5):
        super().__init__()
        self.seq_len = seq_len
        
        self.multi_en = MultiheadAttention( number_of_heads = heads_en, dim_model = dim_model, dropout = dropout, blending=blending  )     # multihead attention ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block( dim_model )                                            # feedforward block   ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, in_ex, relation_matrix = None, padding_mask = None, q_trick = False):

        # input is batch-first (b,n,d)
        out = in_ex                                      # Applying positional embedding

        # out = out.permute(1,0,2)                         # -> (n,b,d)

        # index for last query
        if (q_trick):
            last_indices = torch.count_nonzero(padding_mask, 1) - 1 
            last_indices = last_indices.unsqueeze(1).unsqueeze(1).repeat(1, 1, out.size(2))
            query = out.gather(1, last_indices.cuda())
        else:
            query = out

        #Multihead attention                            
        b,n,d = out.shape
        out = self.layer_norm1( out )                           # Layer norm
        skip_out = out
        # attention function accepts batch-first format
        out = self.multi_en( query , out , out ,
                                mask=subsequent_mask(n)
                                ,padding_mask = padding_mask 
                                ,relation_matrix = relation_matrix
                                ,q_trick = q_trick
                                )  # attention mask upper triangular
        out = out + skip_out                                    # skip connection
        out = self.dropout(out)

        #feed forward
        # out = out.permute(1,0,2)                                # -> (b,n,d)
        out = self.layer_norm2( out )                           # Layer norm 
        skip_out = out
        out = self.ffn_en( out )
        out = out + skip_out                                    # skip connection
        out = self.dropout(out)

        return out

class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self,dim_model ,total_in, heads_de,seq_len, dropout):
        super().__init__()
        self.seq_len    = seq_len
        self.embd_in    = nn.Embedding(  total_in , embedding_dim = dim_model )                  #interaction embedding
        self.embd_pos   = nn.Embedding(  seq_len , embedding_dim = dim_model )                  #positional embedding
        #self.embd_pos   = PositionalEncoding(  dim_model, 0., seq_len )
        self.multi_de1  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = Feed_Forward_block( dim_model )                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )
        self.layer_norm3 = nn.LayerNorm( dim_model )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, in_in, en_out,first_block=True):

         ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_in = self.embd_in( in_in )

            #combining the embedings
            out = in_in                         # (b,n,d)
        else:
            out = in_in

        in_pos = get_pos(self.seq_len)
        in_pos = self.embd_pos( in_pos )
        out = out + in_pos                                          # Applying positional embedding

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape )
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1( out )
        skip_out = out
        out, attn_wt = self.multi_de1( out , out , out, 
                                     attn_mask=get_mask(seq_len=n)) # attention mask upper triangular
        out = skip_out + out                                        # skip connection
        out = self.dropout(out)

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2( en_out )
        skip_out = out
        out, attn_wt = self.multi_de2( out , en_out , en_out,
                                    attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = out + skip_out
        out = self.dropout(out)

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3( out )                               # Layer norm 
        skip_out = out
        out = self.ffn_en( out )                                    
        out = out + skip_out                                        # skip connection
        out = self.dropout(out)


        return out

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    ##todo add this to device
    return torch.from_numpy( np.triu(np.ones((seq_len ,seq_len)), k=1).astype('bool')).cuda()

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (size, size)
    subsequent_mask = np.tril(np.ones(attn_shape), k=0).astype('uint8')
    return torch.from_numpy(subsequent_mask).cuda() == 1

def get_pos(seq_len):
    # use sine positional embeddinds
    return torch.arange( seq_len ).unsqueeze(0)
    #return torch.arange( seq_len ).unsqueeze(1)

class SAKT(nn.Module):  
    def __init__(self , ex_total, cat_total , seq_len, dim, heads, dout, 
                        use_id_embed=True, 
                        use_cat_embed=True,
                        use_mean_elapsed=True,
                        use_mean_correctness=True,
                        use_pos_embed=True,
                        use_past_correctness=True,
                        use_elapsed_time=True,
                        use_timestamp=True,
                        use_sinusoidal_pos=True,
                        mode = 'correctness'):
        super(SAKT, self).__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.mode = mode
        self.use_id_embed = use_id_embed 
        self.use_cat_embed = use_cat_embed
        self.use_mean_elapsed = use_mean_elapsed
        self.use_mean_correctness = use_mean_correctness
        self.use_pos_embed = use_pos_embed
        self.use_past_correctness = use_past_correctness
        self.use_elapsed_time = use_elapsed_time
        self.use_timestamp = use_timestamp
        dim_embed = dim
        dim_model = dim

        self.embd_ex =   nn.Embedding( ex_total, embedding_dim = dim_embed ) # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( cat_total, embedding_dim = dim_embed )
        self.embd_correct = nn.Embedding( 3, embedding_dim = dim_embed )
        self.embd_pos = PositionalEncoding(d_model=dim_embed, dropout=dout, max_len=seq_len) if use_sinusoidal_pos else nn.Embedding( seq_len , embedding_dim = dim_embed )

        self.embed_vector_elapsed = nn.Linear(1, dim_embed)
        self.embed_vector_timestamp = nn.Linear(1, dim_embed)        
        self.embed_vector_me = nn.Linear(1, dim_embed)        
        self.embed_vector_mc = nn.Linear(1, dim_embed)        

        self.query_ffn = nn.Linear((use_id_embed + use_cat_embed + use_mean_elapsed + use_mean_correctness + use_pos_embed) * dim_embed, dim_model)
        self.kv_ffn = nn.Linear((use_past_correctness + use_elapsed_time + use_timestamp + use_pos_embed) * dim_embed, dim_model)
        
        self.linear = nn.ModuleList( [nn.Linear(in_features= dim_model , out_features= dim_model ) for x in range(3)] )   # Linear projection for each embedding
        self.attn = nn.MultiheadAttention(embed_dim= dim_model , num_heads= heads, dropout= dout )
        self.ffn = nn.ModuleList([nn.Linear(in_features= dim_model , out_features=dim_model, bias= True) for x in range(2)])  # feed forward layers post attention
        self.relu = nn.ReLU()  # feed forward layers post attention

        self.linear_out = nn.Linear(in_features= dim_model , out_features= 1 , bias=True)
        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )                           # output with correctnness prediction 
        self.drop = nn.Dropout(dout)


    def forward(self, in_id = None, in_cat = None, in_in = None, in_elapsed = None, in_ts = None, in_me = None, in_mc = None):       

        padding_mask = in_id > 0  # (b,n). pytorch wants it in this format
        padding_mask = padding_mask.cuda()

        n_batch = in_id.size(0)
        seq_len = in_id.size(1)

        sequence_lengths = torch.count_nonzero(in_id > 0, 1).cpu()

        # Empty array with compatible shape for adding embeddings
        in_qu = torch.empty((n_batch,seq_len,0)).cuda()
        in_ex = torch.empty((n_batch,seq_len,0)).cuda()

        in_id = self.embd_ex( in_id ) # Exercise ID
        in_cat = self.embd_cat( in_cat ) # Exercise category
        in_in = self.embd_correct( in_in ) # Past interactions

        in_pos = get_pos(self.seq_len) 
        in_pos = self.embd_pos( in_pos )
        in_pos = in_pos.repeat(in_id.size(0), 1, 1) # Positional embedding

        in_elapsed = self.embed_vector_elapsed(in_elapsed) # Past elapsed times
        in_ts = self.embed_vector_timestamp(in_ts) # Timestamp difference since last question
        in_me = self.embed_vector_me(in_me) # Mean elapsed time for current exercise ID
        in_mc = self.embed_vector_mc(in_mc) # Mean correctness for current exercise ID

        if self.use_id_embed:
            in_qu = torch.cat((in_qu, in_id), 2)
        if self.use_cat_embed:
            in_qu = torch.cat((in_qu, in_cat), 2)
        if self.use_mean_elapsed:
            in_qu = torch.cat((in_qu, in_me), 2)
        if self.use_mean_correctness:
            in_qu = torch.cat((in_qu, in_mc), 2)


        if self.use_past_correctness:
            in_ex = torch.cat((in_ex, in_in), 2)
        if self.use_elapsed_time:
            in_ex = torch.cat((in_ex, in_elapsed), 2)
        if self.use_timestamp:
            in_ex = torch.cat((in_ex, in_ts), 2)

            
        if self.use_pos_embed:
            in_ex = torch.cat((in_ex, in_pos), 2)
            in_qu = torch.cat((in_qu, in_pos), 2)

        out_in = self.kv_ffn(in_ex)

        out_in = self.relu(out_in)

        ## split the interaction embeding into v and k ( needs to verify if it is slpited or not)
        value_in = out_in
        key_in   = out_in
        
        ## get the excercise embedding output
        query_ex = self.query_ffn(in_qu)
        query_ex = self.relu(query_ex)

        ## Linearly project all the embedings
        value_in = self.linear[0](value_in).permute(1,0,2)        # (b,n,d) --> (n,b,d)
        key_in = self.linear[1](key_in).permute(1,0,2)
        query_ex =  self.linear[2](query_ex).permute(1,0,2)

        ## pass through multihead attention
        atn_out, weights = self.attn(query_ex, 
                                key_in, value_in, 
                                attn_mask=get_mask(seq_len=self.seq_len) # lower triangular mask, bool, torch (n,b,d)
                                # ,key_padding_mask = padding_mask # padding mask 
                                )
        # Residual connection ; added excercise embd as residual because previous ex may have imp info, suggested in paper.
        atn_out = query_ex + atn_out                                  
        
        atn_out = self.layer_norm1( atn_out )                          # Layer norm    ##print('atn',atn_out.shape) #n,b,d = atn_out.shape

        #take batch on first axis 
        atn_out = atn_out.permute(1,0,2)                              #  (n,b,d) --> (b,n,d)
        
        ## FFN 2 layers
        ffn_out = self.drop(self.ffn[1]( self.relu( self.ffn[0]( atn_out ) )))   # (n,b,d) -->    .view([n*b ,d]) is not needed according to the kaggle implementation
        ffn_out = self.layer_norm2( ffn_out + atn_out )                # Layer norm and Residual connection

        ffn_out = self.linear_out(ffn_out)
        if self.mode=='correctness':
            ffn_out = torch.sigmoid(ffn_out)

        return ffn_out


class Saint_Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self , dim_model, heads_en, total_ex ,total_cat, seq_len, dropout, use_sinusoidal_pos = True, ):
        super().__init__()
        self.seq_len = seq_len
        self.dim_model = dim_model
        dim_embed = dim_model
        
        self.embd_ex =   nn.Embedding( total_ex, embedding_dim = dim_embed ) # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = dim_embed )
        self.embd_correct = nn.Embedding( 3, embedding_dim = dim_embed )
        self.embd_pos = PositionalEncoding(d_model=dim_embed, dropout=dropout, max_len=seq_len) if use_sinusoidal_pos else nn.Embedding( seq_len , embedding_dim = dim_embed )

        self.embed_vector_elapsed = nn.Linear(1, dim_embed)
        self.embed_vector_timestamp = nn.Linear(1, dim_embed)        
        self.embed_vector_me = nn.Linear(1, dim_embed)        
        self.embed_vector_mc = nn.Linear(1, dim_embed)        
        
        self.multi_en = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_en,  )     # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block( dim_model )                                            # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, in_ex, in_cat, first_block=True, query=None, padding_mask = None):

        ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_ex = self.embd_ex( in_ex )
            in_cat = self.embd_cat( in_cat )
            #combining the embedings
            out = in_ex + in_cat                      # (b,n,d)
        else:
            out = in_ex
        
        in_pos = get_pos(self.seq_len)
        in_pos = self.embd_pos( in_pos )
        out = out + in_pos                                      # Applying positional embedding

        out = out.permute(1,0,2) # -> (n,b,d)
        
        #Multihead attention                            
        out = self.layer_norm1( out )                           # Layer norm
        skip_out = out
        if query is None:
            query = out            
        else:
            query = query.permute(1,0,2) # -> (n,b,d)
        out, attn_wt = self.multi_en( query , out , out ,
                                attn_mask=get_mask(seq_len=self.seq_len),
                                    key_padding_mask = padding_mask)  # attention mask upper triangular
        out = out + skip_out                                    # skip connection
        out = self.dropout(out)

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2( out )                           # Layer norm 
        skip_out = out
        out = self.ffn_en( out )
        out = out + skip_out                                    # skip connection
        out = self.dropout(out)

        return out


class Saint_Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self,dim_model ,total_in, heads_de,seq_len, dropout, use_elapsed_time = True, use_timestamp = True, use_sinusoidal_pos = True,  ):
        super().__init__()
        self.seq_len    = seq_len
        self.embd_in    = nn.Embedding(  total_in , embedding_dim = dim_model )                  #interaction embedding
        self.embd_pos = PositionalEncoding(d_model=dim_model, dropout=dropout, max_len=seq_len) if use_sinusoidal_pos else nn.Embedding( seq_len , embedding_dim = dim_model )
        self.multi_de1  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = Feed_Forward_block( dim_model )                                         # feed forward layer

        self.use_elapsed_time = use_elapsed_time
        self.use_timestamp = use_timestamp

        self.embed_vector_elapsed = nn.Linear(1, dim_model)
        self.embed_vector_ts= nn.Linear(1, dim_model)

        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )
        self.layer_norm3 = nn.LayerNorm( dim_model )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, in_in, en_out, first_block=True, in_elapsed = None, in_ts = None):

         ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_in = self.embd_in( in_in )
            in_elapsed = self.embed_vector_elapsed(in_elapsed)
            in_ts = self.embed_vector_ts(in_ts)

            #combining the embedings
            if self.use_elapsed_time:
                out = in_in + in_elapsed                         # (b,n,d)
            if self.use_timestamp:
                out = in_in + in_ts
        else:
            out = in_in

        in_pos = get_pos(self.seq_len)
        in_pos = self.embd_pos( in_pos )
        out = out + in_pos                                          # Applying positional embedding

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape )
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1( out )
        skip_out = out
        out, attn_wt = self.multi_de1( out , out , out, 
                                     attn_mask=get_mask(seq_len=n)) # attention mask upper triangular
        out = skip_out + out                                        # skip connection
        out = self.dropout(out)

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2( en_out )
        skip_out = out
        out, attn_wt = self.multi_de2( out , en_out , en_out,
                                    attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = out + skip_out
        out = self.dropout(out)

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3( out )                               # Layer norm 
        skip_out = out
        out = self.ffn_en( out )                                    
        out = out + skip_out                                        # skip connection
        out = self.dropout(out)


        return out

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    ##todo add this to device
    return torch.from_numpy( np.triu(np.ones((seq_len ,seq_len)), k=1).astype('bool')).cuda()

def get_pos(seq_len):
    # use sine positional embeddinds
    return torch.arange( seq_len ).unsqueeze(0)
    #return torch.arange( seq_len ).unsqueeze(1)

class SAINT(nn.Module):
    def __init__(self,dim_model,num_en, num_de ,heads_en, total_ex ,total_cat, total_in, heads_de, seq_len, dropout, mode = 'correctness',
                    use_sinusoidal_pos = True):
        super().__init__( )

        self.num_en = num_en
        self.num_de = num_de

        self.mode = mode

        self.encoder = get_clones(Saint_Encoder_block(dim_model, heads_en, total_ex, total_cat, seq_len, dropout, use_sinusoidal_pos=use_sinusoidal_pos), num_en)
        self.decoder = get_clones(Saint_Decoder_block(dim_model ,total_in, heads_de, seq_len, dropout, use_sinusoidal_pos=use_sinusoidal_pos)           , num_de)

        self.out = nn.Linear(in_features= dim_model , out_features=1)
    
    def forward(self, in_ex = None, in_cat = None, in_in = None, in_elapsed = None, in_ts = None, in_me = None, in_mc = None):
        
        padding_mask = in_ex == 0 # (batch, sequence). pytorch wants it in this format for some reason
        padding_mask = padding_mask.cuda()

        ## pass through each of the encoder blocks in sequence
        first_block = True
        for x in range(self.num_en):
            if x>=1:
                first_block = False
            in_ex = self.encoder[x]( in_ex, in_cat ,first_block=first_block)
            in_cat = in_ex                                  # passing same output as q,k,v to next encoder block

        
        ## pass through each decoder block in sequence
        first_block = True
        for x in range(self.num_de):
            if x>=1:
                first_block = False
            in_in = self.decoder[x]( in_in, en_out= in_ex, first_block=first_block, in_elapsed=in_elapsed, in_ts=in_ts )

        ## Output layer
        in_in = self.out( in_in )
        if self.mode=='correctness':
            in_in = torch.sigmoid(in_in)

        return in_in

class KEETAR(nn.Module):
    def __init__(self, dim_model, dim_embed, num_en, heads_en, total_ex ,total_cat, total_in, seq_len, dropout, q_trick = False,
                    relation_matrix = None,
                    preload_relation_matrix = False,
                    use_id_embed=True, 
                    use_cat_embed=True,
                    use_mean_elapsed=True,
                    use_mean_correctness=True,
                    use_pos_embed=True,
                    use_past_correctness=True,
                    use_elapsed_time=True,
                    use_timestamp=True, 
                    use_sinusoidal_pos = True,
                    blending = 0.5,
                    lstm_layers = 2,
                    mode = 'correctness'):
        super().__init__( )

        self.num_en = num_en
        self.seq_len = seq_len
        self.h = heads_en
        self.q_trick = q_trick
        self.relation_matrix = relation_matrix
        self.preload_relation_matrix = preload_relation_matrix

        self.mode = mode
        self.use_id_embed = use_id_embed 
        self.use_cat_embed = use_cat_embed
        self.use_mean_elapsed = use_mean_elapsed
        self.use_mean_correctness = use_mean_correctness
        self.use_pos_embed = use_pos_embed
        self.use_past_correctness = use_past_correctness
        self.use_elapsed_time = use_elapsed_time
        self.use_timestamp = use_timestamp

        self.embd_ex =   nn.Embedding( total_ex, embedding_dim = dim_embed ) # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = dim_embed )
        self.embd_correct = nn.Embedding( total_in, embedding_dim = dim_embed )
        self.embd_pos = PositionalEncoding(d_model=dim_embed, dropout=dropout, max_len=seq_len) if use_sinusoidal_pos else nn.Embedding( seq_len , embedding_dim = dim_embed )

        self.embed_vector_elapsed = nn.Linear(1, dim_embed)
        self.embed_vector_timestamp = nn.Linear(1, dim_embed)        
        self.embed_vector_me = nn.Linear(1, dim_embed)        
        self.embed_vector_mc = nn.Linear(1, dim_embed)        
        
        self.in_ffn = nn.Linear((use_id_embed + use_cat_embed + use_pos_embed + use_past_correctness + use_mean_elapsed + use_mean_correctness + use_elapsed_time + use_timestamp) * dim_embed, dim_model)

        self.encoder = get_clones(Encoder_block(dim_model, heads_en, total_ex, total_cat, seq_len, dropout, blending=blending), num_en)
        self.lstm = nn.LSTM(dim_model, dim_model, lstm_layers, dropout = dropout)
        self.out_ffn = nn.Linear(dim_model, 1)

        self.relu = nn.ReLU()

        self.layer_norm = nn.LayerNorm( dim_model )

        self.dout = nn.Dropout(dropout)
    
    def forward(self, in_id = None, in_cat = None, in_in = None, in_elapsed = None, in_ts = None, in_me = None, in_mc = None):

        padding_mask = in_id > 0  # (b,n). pytorch wants it in this format
        padding_mask = padding_mask.cuda()

        n_batch = in_id.size(0)
        seq_len = in_id.size(1)

        sequence_lengths = torch.count_nonzero(in_id > 0, 1).cpu()

        in_id_np = in_id.cpu().detach().numpy()

        if self.relation_matrix is not None:
            # first select the rows that we need
            if self.preload_relation_matrix:
                relation_matrix_slice = self.relation_matrix[in_id_np,:]
            else:                
                relation_matrix_slice = torch.from_numpy(self.relation_matrix[in_id_np,:]).cuda()
            # then select the columns that we need
            relation_matrix_slice = relation_matrix_slice[:,:,in_id.long()]
            # now we have useless rows, we have to select the right row from each of the batch 'mini-matrices'
            relation_matrix_slice = relation_matrix_slice[torch.arange(n_batch),:,torch.arange(n_batch),:]
            # if we're doing q-trick we have to select only the row corresponding to the query
            if (self.q_trick):
                # indices for the query rows
                last_indices = (torch.count_nonzero(padding_mask, 1) - 1).cuda()
                # select rows according to indices
                relation_matrix_slice = relation_matrix_slice[torch.arange(n_batch), last_indices, :]
                # unsqueeze so shape is consistent with scores
                relation_matrix_slice = relation_matrix_slice.unsqueeze(1)
            # create 'heads' dimension for the relation matrix slice so it's the same shape as scores matrix in attention function
            relation_matrix_slice = relation_matrix_slice.unsqueeze(1).repeat(1, self.h, 1, 1).float()
        else:
            relation_matrix_slice = None

        # Empty array with compatible shape for adding embeddings
        in_ex = torch.empty((n_batch,seq_len,0)).cuda()

        in_id = self.embd_ex( in_id ) # Exercise ID
        in_cat = self.embd_cat( in_cat ) # Exercise category
        in_in = self.embd_correct( in_in ) # Past interactions

        in_pos = get_pos(self.seq_len) 
        in_pos = self.embd_pos( in_pos )
        in_pos = in_pos.repeat(in_id.size(0), 1, 1) # Positional embedding

        in_elapsed = self.embed_vector_elapsed(in_elapsed) # Past elapsed times
        in_ts = self.embed_vector_timestamp(in_ts) # Timestamp difference since last question
        in_me = self.embed_vector_me(in_me) # Mean elapsed time for current exercise ID
        in_mc = self.embed_vector_mc(in_mc) # Mean correctness for current exercise ID

        if self.use_id_embed:
            in_ex = torch.cat((in_ex, in_id), 2)
        if self.use_cat_embed:
            in_ex = torch.cat((in_ex, in_cat), 2)
        if self.use_past_correctness:
            in_ex = torch.cat((in_ex, in_in), 2)
        if self.use_pos_embed:
            in_ex = torch.cat((in_ex, in_pos), 2)
        if self.use_elapsed_time:
            in_ex = torch.cat((in_ex, in_elapsed), 2)
        if self.use_timestamp:
            in_ex = torch.cat((in_ex, in_ts), 2)
        if self.use_mean_elapsed:
            in_ex = torch.cat((in_ex, in_me), 2)
        if self.use_mean_correctness:
            in_ex = torch.cat((in_ex, in_mc), 2)         

        in_ex = self.in_ffn(in_ex)
        
        in_ex = self.relu(in_ex)
        
        ## Pass through each of the encoder blocks in sequence
        # input (b,n,d); output (b,n,d)
        for x in range(self.num_en):
            in_ex = self.encoder[x]( in_ex, relation_matrix = relation_matrix_slice, padding_mask = padding_mask, q_trick = self.q_trick)

        in_ex = self.dout(in_ex)

        ## Res connection
        in_ex = in_ex.permute(1,0,2) # -> (n,b,d)
        res = in_ex
        ## Pass through LSTM
        # input (n,b,d); output (n,b,d)
        in_ex, _ = self.lstm(in_ex)
        in_ex = in_ex + res

        ## Return to batch-first format
        in_ex = in_ex.permute(1,0,2) # -> (b,n,d)
        # Layer norm
        in_ex = self.layer_norm(in_ex)
        ## Pass through final FFN
        in_ex = self.out_ffn(in_ex)
        if self.mode=='correctness':
            in_ex = torch.sigmoid(in_ex)

        return in_ex

class DKT(nn.Module):
    def __init__(self, dim_model, dim_embed, total_ex ,total_cat, total_in, seq_len, dropout,
                        use_id_embed=True, 
                        use_cat_embed=True,
                        use_mean_elapsed=True,
                        use_mean_correctness=True,
                        use_pos_embed=True,
                        use_past_correctness=True,
                        use_elapsed_time=True,
                        use_timestamp=True,
                        lstm_layers = 2,
                        mode='correctness'):
        super().__init__( )

        self.seq_len = seq_len

        self.embd_ex =   nn.Embedding( total_ex, embedding_dim = dim_embed ) # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = dim_embed )
        self.embd_correct = nn.Embedding( total_in, embedding_dim = dim_embed )
        self.embd_pos   = nn.Embedding(  seq_len , embedding_dim = dim_embed ) # positional embedding

        self.mode = mode
        self.use_id_embed = use_id_embed 
        self.use_cat_embed = use_cat_embed
        self.use_mean_elapsed = use_mean_elapsed
        self.use_mean_correctness = use_mean_correctness
        self.use_pos_embed = use_pos_embed
        self.use_past_correctness = use_past_correctness
        self.use_elapsed_time = use_elapsed_time
        self.use_timestamp = use_timestamp

        self.embed_vector_elapsed = nn.Linear(1, dim_embed)
        self.embed_vector_timestamp = nn.Linear(1, dim_embed)       
        self.embed_vector_me = nn.Linear(1, dim_embed)        
        self.embed_vector_mc = nn.Linear(1, dim_embed)        
        
        self.in_ffn = nn.Linear((use_id_embed + use_cat_embed + use_pos_embed + use_past_correctness + use_mean_elapsed + use_mean_correctness + use_elapsed_time + use_timestamp) * dim_embed, dim_model)

        self.rnn = nn.LSTM(dim_model, dim_model, lstm_layers, dropout = dropout, batch_first=True)
        self.out_ffn = nn.Linear(dim_model, 1)

        self.relu = nn.ReLU()

        self.layer_norm = nn.LayerNorm( dim_model )

        self.dout = nn.Dropout(dropout)
    
    def forward(self, in_id = None, in_cat = None, in_in = None, in_elapsed = None, in_ts = None, in_me = None, in_mc = None):

        padding_mask = in_id > 0  # (b,n). pytorch wants it in this format
        padding_mask = padding_mask.cuda()

        n_batch = in_id.size(0)
        seq_len = in_id.size(1)

        sequence_lengths = torch.count_nonzero(in_id > 0, 1).cpu()

        # Empty array with compatible shape for adding embeddings
        in_ex = torch.empty((n_batch,seq_len,0)).cuda()

        in_id = self.embd_ex( in_id ) # Exercise ID
        in_cat = self.embd_cat( in_cat ) # Exercise category
        in_in = self.embd_correct( in_in ) # Past interactions

        in_pos = get_pos(self.seq_len) 
        in_pos = self.embd_pos( in_pos )
        in_pos = in_pos.repeat(in_id.size(0), 1, 1) # Positional embedding

        in_elapsed = self.embed_vector_elapsed(in_elapsed) # Past elapsed times
        in_ts = self.embed_vector_timestamp(in_ts) # Timestamp difference since last question
        in_me = self.embed_vector_me(in_me) # Mean elapsed time for current exercise ID
        in_mc = self.embed_vector_mc(in_mc) # Mean correctness for current exercise ID

        if self.use_id_embed:
            in_ex = torch.cat((in_ex, in_id), 2)
        if self.use_cat_embed:
            in_ex = torch.cat((in_ex, in_cat), 2)
        if self.use_past_correctness:
            in_ex = torch.cat((in_ex, in_in), 2)
        if self.use_pos_embed:
            in_ex = torch.cat((in_ex, in_pos), 2)
        if self.use_elapsed_time:
            in_ex = torch.cat((in_ex, in_elapsed), 2)
        if self.use_timestamp:
            in_ex = torch.cat((in_ex, in_ts), 2)
        if self.use_mean_elapsed:
            in_ex = torch.cat((in_ex, in_me), 2)
        if self.use_mean_correctness:
            in_ex = torch.cat((in_ex, in_mc), 2)   

        in_ex = self.in_ffn(in_ex)

        in_ex = self.relu(in_ex)

        # in_ex = pack_padded_sequence(in_ex, sequence_lengths, batch_first=True, enforce_sorted=False)
        
        ## Pass through LSTM
        in_ex, _ = self.rnn(in_ex)
        
        # in_ex, lengths = pad_packed_sequence(in_ex, batch_first=True, total_length=seq_len)                
        
        ## Pass through final FFN
        in_ex = self.out_ffn(in_ex)
        if self.mode=='correctness':
            in_ex = torch.sigmoid(in_ex)
        
        return in_ex