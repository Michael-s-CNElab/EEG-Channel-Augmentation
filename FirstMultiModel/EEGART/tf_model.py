import numpy as np
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math, copy
from torch.autograd import Variable
from tf_utils import draw
import time
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from tf_config import ARTConfig, ARTEncoder_CLSConfig, SLTConfig

# ---------------------Encoder and Decoder Stacks--------------------
# class ART(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many 
#     other models.
#     """
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(ART, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator

        
#     def forward(self, src, tgt, src_mask, tgt_mask):
#         "Take in and process masked src and target sequences."
#         x = self.encoder(self.src_embed(src), src_mask)
#         x = self.decoder(self.tgt_embed(tgt), x, src_mask, tgt_mask)
    
#         return self.generator(x).transpose(1,2)
        

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        '''
        temp = self.proj(x)
        print("Generator1:", temp.shape)
        draw(3, temp[0, :, :], "Gen(lp)_" + str(2))
        temp = F.log_softmax(self.proj(x), dim=-1)
        print("Generator2:", temp.shape)
        draw(3, temp[0, :, :], "Gen(sm)_" + str(3))
        '''
        #print("linear project")
        #return F.log_softmax(self.proj(x), dim=-1)
        return self.proj(x)
    
class Classifier(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, time_len):
        super(Classifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=time_len, out_channels=1, kernel_size=1)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # print(f"CLS input shape: {x.shape}")
        x = self.conv(x)
        # print(f"CLS shape: {x.shape}")
        x = self.proj(x.squeeze(-1))
        # print(f"Proj shape: {x.shape}")
        return x
        
# ---------------------Encoder--------------------
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        atten_list = []
        for layer in self.layers:
            x, atten = layer(x, mask)
            # print(f"Encoder x size:{x.shape}")
            atten_list.append(atten)
        return self.norm(x), atten_list
        
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2        


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        hidden, atten = sublayer(self.norm(x))
        return x + self.dropout(hidden), atten

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)

        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."

        hidden = self.norm(x)
        hidden, atten = self.self_attn(hidden, hidden, hidden, mask)
        hidden = self.dropout(hidden)
        x = x + hidden
        hidden = self.norm(x)
        hidden = self.feed_forward(hidden)
        hidden = self.dropout(hidden)
        x = x + hidden

        return x, atten

# ---------------------Decoder--------------------
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, l1_cs=False):
        atten_list = []
        for layer in self.layers:
            x, atten = layer(x, memory, src_mask, tgt_mask, l1_cs)
            atten_list.append(atten)
        return self.norm(x), atten_list

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask, 
                l1_cs): # layer one do cross attention
        "Follow Figure 1 (right) for connections."
        # First SA
        if not l1_cs:
            hidden = self.norm(x)
            hidden, atten_1 = self.self_attn(hidden, hidden, hidden, tgt_mask)
            hidden = self.dropout(hidden)
            x = x + hidden
        else:
            hidden = self.norm(x)
            hidden, atten_1 = self.self_attn(hidden, memory, memory)
            hidden = self.dropout(hidden)
            x = x + hidden

        # Second cross SA
        hidden = self.norm(x)
        hidden, atten_2 = self.self_attn(hidden, memory, memory)
        hidden = self.dropout(hidden)
        x = x + hidden

        hidden = self.norm(x)
        hidden = self.feed_forward(hidden)
        hidden = self.dropout(hidden)
        x = x + hidden

        return x, (atten_1, atten_2)



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


#---------------------Attention--------------------        
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # print("attention1:", query.shape, key.transpose(-2, -1).shape, value.shape)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print("attention2:", d_k, scores.shape)
    # draw(3, scores[0, 0, :, :], "scores_" + str(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # draw(3, scores[0, 0, :, :], "scores_" + str(2))
    p_attn = F.softmax(scores, dim = -1)
    # draw(3, p_attn[0, 0, :, :], "scores_" + str(3))
    if dropout is not None:
        p_attn = dropout(p_attn)
        #draw(3, p_attn[0, 0, :, :], "scores_" + str(4))
    return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # print("MultiHeadedAttention1:", query.shape, key.shape, value.shape, mask.shape)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # print("MultiHeadedAttention2:", query.shape, key.shape, value.shape)
        '''
        print("MultiHeadedAttention2:", query.shape, key.shape, value.shape)
        for i in range(self.h):
            draw(3, query[0,i,:,:], "Query_"+str(i))
            draw(3, key[0, i, :, :], "Key_" + str(i))
            draw(3, value[0, i, :, :], "Value_" + str(i))
        '''
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # print("MultiHeadedAttention3:", x.shape, self.attn.shape)
        '''
        print("MultiHeadedAttention3:", x.shape, self.attn.shape)
        for i in range(self.h):
            draw(3, self.attn[0, i, :, :], "Atten map_" + str(i))
            draw(3, x[0, i, :, :], "X_" + str(i))
        '''
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        # print("MultiHeadedAttention4:", x.shape, self.linears[-1])
        '''        
        print("MultiHeadedAttention4:", x.shape, self.linears[-1])
        draw(3, x[0, :, :], "Concat")

        temp = self.linears[-1](x)
        print("MultiHeadedAttention5:", temp.shape)
        draw(3, temp[0, :, :], "Linear Proj")
        '''
        return self.linears[-1](x), self.attn


# ---------------------Position-wise Feed-Forward Networks--------------------
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        temp = self.w_1(x)
        print("PositionwiseFeedForward1:", temp.shape)
        draw(3, temp[0, :, :], "w_1")
        temp = F.relu(self.w_1(x))
        print("PositionwiseFeedForward2:", temp.shape)
        draw(3, temp[0, :, :], "w_1_relu")
        temp = self.dropout(F.relu(self.w_1(x)))
        print("PositionwiseFeedForward3:", temp.shape)
        draw(3, temp[0, :, :], "dropout")
        temp = self.w_2(self.dropout(F.relu(self.w_1(x))))
        print("PositionwiseFeedForward4:", temp.shape)
        draw(3, temp[0, :, :], "w_2")
        '''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MLP_projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, squence_size):
        super(MLP_projector, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(squence_size),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(squence_size),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(squence_size)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x 


# ---------------------Embeddings--------------------
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # a = self.lut(x)
        # print("Embeddings:", a.shape)
        return self.lut(x) * math.sqrt(self.d_model)

# ---------------------ExpandConv--------------------
class ExpandConv(nn.Module):
    def __init__(self, vocab, d_model):
        super(ExpandConv, self).__init__()
        self.lut = nn.Conv1d(in_channels=vocab, out_channels=d_model, kernel_size=1)
        self.d_model = d_model

    def forward(self, x):
        # print("ExpandConv:", x.shape, type(x))
        convoluted_x = self.lut(x)
        convoluted_x = convoluted_x.permute(0, 2, 1)
        return convoluted_x * math.sqrt(self.d_model)
        
# ---------------------Positional Encoding--------------------
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # scaling term that decreases exponentially as the depth (i.e., column index in pe) increases.
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        # print(x.shape)
        return self.dropout(x)

class ARTModel(PreTrainedModel):
    config_class = ARTConfig

    def __init__(self, config):
        super().__init__(config)

        self.c = copy.deepcopy
        self.src_pos_embedding = nn.Embedding(config.src_len, config.d_model)
        self.tgt_pos_embedding = nn.Embedding(config.tgt_len, config.d_model)
        self.attn = MultiHeadedAttention(config.h, config.d_model)
        self.ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.position = PositionalEncoding(config.d_model, config.dropout)
        self.encoder = Encoder(EncoderLayer(config.d_model, self.c(self.attn), self.c(self.ff), config.dropout), config.N)
        self.decoder = Decoder(DecoderLayer(config.d_model, self.c(self.attn), self.c(self.attn),
                                self.c(self.ff), config.dropout), config.N)
        self.src_embed = nn.Sequential(ExpandConv(config.d_model, config.src_channel_size), 
                                       self.c(self.src_pos_embedding))
        self.tgt_embed = nn.Sequential(ExpandConv(config.d_model, config.src_channel_size), 
                                       self.c(self.tgt_pos_embedding))
        # self.src_embed = nn.Sequential(ExpandConv(config.d_model, config.src_channel_size), self.c(self.position))
        # self.tgt_embed = nn.Sequential(ExpandConv(config.d_model, config.src_channel_size), self.c(self.position))
        self.generator = Generator(config.d_model, config.tgt_len)
        self.loss_fct = nn.MSELoss()

    def forward(self,
        src = torch.FloatTensor, 
        tgt = torch.FloatTensor, 
        src_mask: Optional[torch.FloatTensor] = None,
        tgt_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        # return_loss: Optional[bool] = None,
        ):
        "Take in and process masked src and target sequences."
        encoder_output = self.encoder(self.src_embed(src), src_mask)
        decoder_output = self.decoder(self.tgt_embed(src), encoder_output, src_mask, tgt_mask)
    
        logits = self.generator(decoder_output).transpose(1,2)
        
        if not return_dict:
            return logits
        
        loss = None
        if labels is not None:
            # Compute the z-scores
            logits_mean = torch.mean(logits, dim=0, keepdim=True)
            logits_std = torch.std(logits, dim=0, keepdim=True)
            logits_norm = (logits - logits_mean) / (logits_std + 1e-10)

            labels_mean = torch.mean(labels, dim=0, keepdim=True)
            labels_std = torch.std(labels, dim=0, keepdim=True)
            labels_norm = (labels - labels_mean) / (labels_std + 1e-10)
            loss = self.loss_fct(logits_norm, labels_norm)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
class ART_AUG(PreTrainedModel):
    config_class = SLTConfig

    def __init__(self, config):
        super().__init__(config)
        self.c = copy.deepcopy
       
        self.attn = MultiHeadedAttention(config.h, config.d_model)
        self.ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.position = PositionalEncoding(config.d_model, config.dropout)
        self.encoder = Encoder(EncoderLayer(config.d_model, self.c(self.attn), self.c(self.ff), config.dropout), config.N)
        self.decoder = Decoder(DecoderLayer(config.d_model, self.c(self.attn), self.c(self.attn),
                                self.c(self.ff), config.dropout), config.N)
        
        self.src_projector = nn.Sequential(MLP_projector(config.sensor_time, config.d_ff, config.d_model, config.src_channel_size), self.c(self.position))
        self.tgt_projector = nn.Sequential(MLP_projector(config.source_voxel_time, config.d_ff, config.d_model, config.tgt_channel_size), self.c(self.position))

        # self.src_embed = nn.Sequential(ExpandConv(config.sensor_time, config.d_model), self.c(self.position))
        # self.tgt_embed = nn.Sequential(ExpandConv(config.source_voxel_time, config.d_model), self.c(self.position))

        self.generator = nn.Linear(config.d_model, config.source_voxel_time)
        
        self.loss_fct = nn.MSELoss()

    def forward(self,
        src = torch.FloatTensor, 
        tgt = torch.FloatTensor, 
        src_mask: Optional[torch.FloatTensor] = None, 
        tgt_mask: Optional[torch.FloatTensor] = None, 
        tgt_token_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None, 
        return_dict: Optional[bool] = None, 
        # return_loss: Optional[bool] = None, 
        ):
        "Token be a channel"
        encoder_output, encoder_atten = self.encoder(self.src_projector(src), src_mask)
        """ encoder: 
                src shape: (b, 30, 1024) -> embed (b, 30, d_model=128)
                input shape:  (b, 30, d_model)
                output shape: (b, 30, d_model)
        """
        decoder_output, decoder_atten = self.decoder(self.tgt_projector(tgt), encoder_output, src_mask, tgt_mask, False)
        """ decoder: 
                tgt shape: (b, 30, 1024) -> embed (b, 30, d_model)
                input shape:            (b, 30, 100)
                memory encoder shape:   (b, 30, d_model)
                attention score shape:  (b, 30, 30)
                output shape:           (b, 30, d_model)
        """
        logits = self.generator(decoder_output)

        ### for add EEG into tgt
        logits = logits[:, :204, :]

        if not return_dict:
            return logits

        loss = None
        if labels is not None:

            logits_mean = torch.mean(logits, dim=(1, 2), keepdim=True)
            logits_std = torch.std(logits, dim=(1, 2), keepdim=True)
            logits_norm = (logits - logits_mean) / (logits_std)

            labels_mean = torch.mean(labels, dim=(1, 2), keepdim=True)
            labels_std = torch.std(labels, dim=(1, 2), keepdim=True)
            labels_norm = (labels - labels_mean) / (labels_std)

            # Apply mask: tgt_token_mask is (batch, 204), expand to match (batch, 204, 100)
            expanded_mask = tgt_token_mask.unsqueeze(-1).expand(-1, -1, logits.size(-1))  # shape (batch, 204, 100)
            # Select only masked positions
            masked_logits = logits_norm[expanded_mask]
            masked_labels = labels_norm[expanded_mask]

            # Compute loss only on masked elements
            loss = self.loss_fct(masked_logits, masked_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=(encoder_output, decoder_output),
            attentions=(encoder_atten, decoder_atten),
        )


class SLTModel(PreTrainedModel):
    config_class = SLTConfig

    def __init__(self, config):
        super().__init__(config)
        self.c = copy.deepcopy
       
        self.attn = MultiHeadedAttention(config.h, config.d_model)
        self.ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.src_pos_embedding = nn.Parameter(torch.randn(1, config.src_len, config.d_model))
        self.tgt_pos_embedding = nn.Parameter(torch.randn(1, config.tgt_len, config.d_model))
        
        self.position = PositionalEncoding(config.d_model, config.dropout)
        self.encoder = Encoder(EncoderLayer(config.d_model, self.c(self.attn), self.c(self.ff),
                                             config.dropout), config.N)
        self.decoder = Decoder(DecoderLayer(config.d_model, self.c(self.attn), self.c(self.attn),
                                self.c(self.ff), config.dropout), config.N)
        
        # self.src_projector = nn.Sequential(MLP_projector(config.sensor_time, config.d_ff, config.d_model, config.src_channel_size), self.c(self.position))
        # self.tgt_projector = nn.Sequential(MLP_projector(config.source_voxel_time, config.d_ff, config.d_model, config.tgt_channel_size), self.c(self.position))
        self.src_projector = MLP_projector(config.sensor_time, config.d_ff, config.d_model, config.src_len)
        self.tgt_projector = MLP_projector(config.source_voxel_time, config.d_ff, config.d_model, config.tgt_len)

        # self.src_embed = nn.Sequential(ExpandConv(config.sensor_time, config.d_model), self.c(self.position))
        # self.tgt_embed = nn.Sequential(ExpandConv(config.source_voxel_time, config.d_model), self.c(self.position))

        self.generator = nn.Linear(config.d_model, config.source_voxel_time)

        self.loss_fct = nn.MSELoss()

    def forward(self,
        src = torch.FloatTensor, 
        tgt = torch.FloatTensor, 
        src_mask: Optional[torch.FloatTensor] = None, 
        tgt_mask: Optional[torch.FloatTensor] = None, 
        tgt_token_mask: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.FloatTensor] = None, 
        return_dict: Optional[bool] = None, 
        # return_loss: Optional[bool] = None, 
        ):
        "Token be a channel"
        src_embedding = self.src_projector(src)
        src_embedding = src_embedding + self.src_pos_embedding
        encoder_output, encoder_atten = self.encoder(src_embedding, src_mask)
        """ encoder: 
                src shape: (b, 30, 100) -> embed (b, 30, d_model)
                input shape:  (b, 30, d_model)
                output shape:     (b, 30, d_model)
        """
        tgt_embedding = self.tgt_projector(tgt)
        tgt_embedding = tgt_embedding + self.tgt_pos_embedding
        decoder_output, decoder_atten = self.decoder(tgt_embedding, encoder_output, src_mask, tgt_mask, False)
        """ decoder: 
                tgt shape: (b, 204, 100) -> embed (b, 204, d_model)
                input shape:            (b, 204, 100)
                memory encoder shape:   (b, 30, 128)
                attention score shape:  (b, 204, 30)
                output shape:           (b, 204, 128)
        """
        logits = self.generator(decoder_output)

        ### for add EEG into tgt
        logits = logits[:, :204, :]

        if not return_dict:
            return logits

        loss = None
        if labels is not None:

            logits_mean = torch.mean(logits, dim=(1, 2), keepdim=True)
            logits_std = torch.std(logits, dim=(1, 2), keepdim=True)
            logits_norm = (logits - logits_mean) / (logits_std)

            labels_mean = torch.mean(labels, dim=(1, 2), keepdim=True)
            labels_std = torch.std(labels, dim=(1, 2), keepdim=True)
            labels_norm = (labels - labels_mean) / (labels_std)

            # Apply mask: tgt_token_mask is (batch, 204), expand to match (batch, 204, 100)
            expanded_mask = tgt_token_mask.unsqueeze(-1).expand(-1, -1, logits.size(-1))  # shape (batch, 204, 100)
            # Select only masked positions
            masked_logits = logits_norm[expanded_mask]
            masked_labels = labels_norm[expanded_mask]

            # Compute loss only on masked elements
            loss = self.loss_fct(masked_logits, masked_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=(encoder_output, decoder_output),
            attentions=(encoder_atten, decoder_atten),
        )

class pre_SLTModel(PreTrainedModel):
    config_class = SLTConfig

    def __init__(self, config):
        super().__init__(config)
        self.c = copy.deepcopy
       
        self.attn = MultiHeadedAttention(config.h, config.d_model)
        self.ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.position = PositionalEncoding(config.d_model, config.dropout)
        self.encoder = Encoder(EncoderLayer(config.d_model, self.c(self.attn), self.c(self.ff), config.dropout), config.N)
        self.decoder = Decoder(DecoderLayer(config.d_model, self.c(self.attn), self.c(self.attn),
                                self.c(self.ff), config.dropout), config.N)
        
        self.src_projector = nn.Sequential(MLP_projector(config.sensor_time, config.d_ff, config.d_model, config.src_channel_size), self.c(self.position))
        self.tgt_projector = nn.Sequential(MLP_projector(config.source_voxel_time, config.d_ff, config.d_model, config.tgt_channel_size), self.c(self.position))

        # self.src_embed = nn.Sequential(ExpandConv(config.sensor_time, config.d_model), self.c(self.position))
        # self.tgt_embed = nn.Sequential(ExpandConv(config.source_voxel_time, config.d_model), self.c(self.position))


        self.generator = nn.Linear(config.d_model, config.source_voxel_time)
        
        self.loss_fct = nn.MSELoss()

    def forward(self,
        src = torch.FloatTensor, 
        tgt = torch.FloatTensor, 
        src_mask: Optional[torch.FloatTensor] = None, 
        tgt_mask: Optional[torch.FloatTensor] = None, 
        tgt_token_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None, 
        return_dict: Optional[bool] = None, 
        # return_loss: Optional[bool] = None, 
        ):
        "Token be a channel"
        encoder_output, encoder_atten = self.encoder(self.src_projector(src), src_mask)   
        """ encoder: 
                src shape: (b, 30, 100) -> embed (b, 30, d_model)
                input shape:  (b, 30, d_model)
                output shape:     (b, 30, d_model)
        """
        decoder_output, decoder_atten = self.decoder(self.tgt_projector(tgt), encoder_output, src_mask, tgt_mask, False)
        """ decoder: 
                tgt shape: (b, 204, 100) -> embed (b, 204, d_model)
                input shape:            (b, 204, 100)
                memory encoder shape:   (b, 30, 128)
                attention score shape:  (b, 204, 30)
                output shape:           (b, 204, 128)
        """
        logits = self.generator(decoder_output)


        if not return_dict:
            return logits

        loss = None
        if labels is not None:

            logits_mean = torch.mean(logits, dim=(1, 2), keepdim=True)
            logits_std = torch.std(logits, dim=(1, 2), keepdim=True)
            logits_norm = (logits - logits_mean) / (logits_std)

            labels_mean = torch.mean(labels, dim=(1, 2), keepdim=True)
            labels_std = torch.std(labels, dim=(1, 2), keepdim=True)
            labels_norm = (labels - labels_mean) / (labels_std)

            # Apply mask: tgt_token_mask is (batch, 204), expand to match (batch, 204, 100)
            expanded_mask = tgt_token_mask.unsqueeze(-1).expand(-1, -1, logits.size(-1))  # shape (batch, 204, 100)
            # Select only masked positions
            masked_logits = logits_norm[expanded_mask]
            masked_labels = labels_norm[expanded_mask]

            # Compute loss only on masked elements
            loss = self.loss_fct(masked_logits, masked_labels)


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=(encoder_output, decoder_output),
            attentions=(encoder_atten, decoder_atten),
        )
    
class SLTModel_ver2(PreTrainedModel):
    config_class = SLTConfig

    def __init__(self, config):
        super().__init__(config)

        self.c = copy.deepcopy
        self.encoder_attn = MultiHeadedAttention(config.h, config.src_d_model)
        self.decoder_attn = MultiHeadedAttention(config.h, config.tgt_d_model)
        self.encoder_ff = PositionwiseFeedForward(config.src_d_model, config.d_ff, config.dropout)
        self.decoder_ff = PositionwiseFeedForward(config.tgt_d_model, config.d_ff, config.dropout)
        self.position = PositionalEncoding(config.d_model, config.dropout)
        self.encoder = Encoder(EncoderLayer(config.src_d_model, self.c(self.encoder_attn), self.c(self.encoder_ff),
                                             config.dropout), config.N)
        self.decoder = Decoder(DecoderLayer(config.tgt_d_model, self.c(self.decoder_attn), self.c(self.decoder_attn),
                                self.c(self.decoder_ff), config.dropout), config.N)
        self.src_embed = nn.Sequential(ExpandConv(config.src_d_model, config.src_channel_size), self.c(self.position))
        self.tgt_embed = nn.Sequential(ExpandConv(config.src_d_model, config.src_channel_size), self.c(self.position))
        self.generator_chan = Generator(config.src_d_model, config.tgt_channel_size)
        self.generator_time = nn.Linear(config.sensor_time, config.source_voxel_time)
        self.loss_fct = nn.MSELoss()

    def forward(self,
        src = torch.FloatTensor, 
        tgt = torch.FloatTensor, 
        src_mask: Optional[torch.FloatTensor] = None, 
        tgt_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        # return_loss: Optional[bool] = None,
        ):
        "Take in and process masked src and target sequences."
        encoder_output = self.encoder(self.src_embed(src), src_mask)
        tgt_tp = self.tgt_embed(tgt).transpose(1,2)
        encoder_output = encoder_output.transpose(1, 2)
        decoder_output = self.decoder(tgt_tp, encoder_output, tgt_mask, tgt_mask).transpose(1,2)
        # print(f"Decoder_output shape: {decoder_output.shape}")
        output_chan_gen = self.generator_chan(decoder_output).transpose(1,2)
        # print(f"generator outpout shpae: {output_chan_gen.shape}")
        logits = self.generator_time(output_chan_gen)
        
        
        if not return_dict:
            return logits
        
        loss = None
        if labels is not None:
            # Compute the z-scores
            logits_mean = torch.mean(logits, dim=0, keepdim=True)
            logits_std = torch.std(logits, dim=0, keepdim=True)
            logits_norm = (logits - logits_mean) / (logits_std + 1e-10)

            labels_mean = torch.mean(labels, dim=0, keepdim=True)
            labels_std = torch.std(labels, dim=0, keepdim=True)
            labels_norm = (labels - labels_mean) / (labels_std + 1e-10)
            loss = self.loss_fct(logits_norm, labels_norm)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

class ARTCLSModel(PreTrainedModel):
    """
        Huggingface需要兩種Model,BaseModel以及Training Model。
        BaseModel回傳BaseModelOutput。
        TrainingModel回傳Loss。
    """
    config_class = ARTEncoder_CLSConfig

    def __init__(self, config):
        super().__init__(config)

        self.c = copy.deepcopy
        self.attn = MultiHeadedAttention(config.h, config.d_model)
        self.ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.position = PositionalEncoding(config.d_model, config.dropout)
        """
            For Classification Problem Generator be an MLP Classifier.
            tgt_vocab be the number of classes
        """
        self.encoder = Encoder(EncoderLayer(config.d_model, self.c(self.attn), self.c(self.ff), config.dropout), config.N)
        self.src_embed = nn.Sequential(ExpandConv(config.d_model, config.src_channel_size), self.c(self.position))
        self.cls = Classifier(config.d_model, config.tgt_channel_size, config.time_len)

        # for p in self.model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform(p)

    def forward(self, 
        src = Optional[torch.FloatTensor], 
        src_mask = Optional[torch.FloatTensor],
        return_dict: Optional[bool] = None,
        ):

        "Take in and process masked src and target sequences."
        encoder_outputs = self.encoder(self.src_embed(src), src_mask)

        last_hidden_state = encoder_outputs
        
        last_hidden_state =  self.cls(last_hidden_state)

        # if not return_dict:
        #     return (last_hidden_state,) + encoder_outputs[1:]
        
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=last_hidden_state,
            attentions=None,
        )

class ART_CLS_PreTrain(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = ARTCLSModel(config)

    def forward(self,
        src: Optional[torch.Tensor] = None, 
        src_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
        ):

        modeloutput = self.model(src, src_mask)
        logits = modeloutput.last_hidden_state.squeeze(dim=1)   # shape: [32, 2]
        loss_fct = CrossEntropyLoss()

        loss = None
        if labels is not None:
            loss = loss_fct(logits, labels)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values = None, 
            hidden_states = None,
            attentions = None,
        )