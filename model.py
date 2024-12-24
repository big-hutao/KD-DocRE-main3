import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import AFLoss
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialImageTransformer
import numpy as np
import math
from itertools import accumulate
import copy



# 原版轴向注意力
# class AxialTransformer_by_entity(nn.Module):
#     def  __init__(self, emb_size = 768, dropout = 0.1, num_layers = 2, dim_index = -1, heads = 8, num_dimensions=2, ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.dim_index = dim_index
#         self.heads = heads
#         self.emb_size = emb_size
#         self.dropout = dropout
#         self.num_dimensions = num_dimensions
#         self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, ) for i in range(num_layers)])
#         self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
#         self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
#         self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
#         self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
#     def forward(self, x):
#         for idx in range(self.num_layers):
#           x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
#           x = self.ffns[idx](x)
#           x = self.ffn_dropouts[idx](x)
#           x = self.lns[idx](x)
#         return x

#第一版，轴向注意力加门控，没有很大的改进
# class AxialTransformer_by_entity(nn.Module):
#     def __init__(self, emb_size=768, dropout=0.1, num_layers=2, dim_index=-1, heads=8, num_dimensions=2):
#         super().__init__()
#         self.num_layers = num_layers
#         self.dim_index = dim_index
#         self.heads = heads
#         self.emb_size = emb_size
#         self.dropout = dropout
#         self.num_dimensions = num_dimensions
#
#         # Axial Attention layers with gating
#         self.axial_attns = nn.ModuleList([
#             GatedAxialAttention(
#                 dim=self.emb_size,
#                 dim_index=dim_index,
#                 heads=heads,
#                 num_dimensions=num_dimensions
#             ) for _ in range(num_layers)
#         ])
#         self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for _ in range(num_layers)])
#         self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for _ in range(num_layers)])
#         self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
#         self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
#
#     def forward(self, x):
#         for idx in range(self.num_layers):
#             # Axial attention with gating
#             x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
#             # Feed-forward network
#             x = self.ffns[idx](x)
#             x = self.ffn_dropouts[idx](x)
#             # Layer normalization
#             x = self.lns[idx](x)
#         return x
#
# class GatedAxialAttention(nn.Module):
#     def __init__(self, dim, dim_index=-1, heads=8, num_dimensions=2):
#         super().__init__()
#         self.dim = dim
#         self.dim_index = dim_index
#         self.heads = heads
#         self.num_dimensions = num_dimensions
#
#         # Axial attention core
#         self.axial_attention = AxialAttention(
#             dim=dim, dim_index=dim_index, heads=heads, num_dimensions=num_dimensions
#         )
#
#         # Gating mechanism
#         self.gate_proj = nn.Linear(dim, dim)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # Compute axial attention
#         attn_output = self.axial_attention(x)
#         # Compute gate values
#         gate = self.sigmoid(self.gate_proj(x))
#         # Apply gating to attention output
#         gated_output = gate * attn_output
#         return gated_output

#第二版，层次化
class HierarchicalAxialTransformer(nn.Module):
    def __init__(self, emb_size=768, dropout=0.1, num_local_layers=2, num_global_layers=2, heads=8, num_dimensions=2):
        """
        层次化 Axial Transformer:
        - 局部层 (Local Transformer): 处理局部上下文 (如句子/段落内关系)
        - 全局层 (Global Transformer): 捕捉全局上下文 (如跨段落/文档级关系)
        """
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_local_layers = num_local_layers
        self.num_global_layers = num_global_layers

        # 局部 Transformer 层
        self.local_transformer = nn.ModuleList([
            AxialAttention(dim=emb_size, dim_index=-1, heads=heads, num_dimensions=num_dimensions)
            for _ in range(num_local_layers)
        ])
        self.local_ffns = nn.ModuleList([nn.Linear(emb_size, emb_size) for _ in range(num_local_layers)])
        self.local_layer_norms = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(num_local_layers)])
        self.local_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_local_layers)])

        # 全局 Transformer 层
        self.global_transformer = nn.ModuleList([
            AxialAttention(dim=emb_size, dim_index=-1, heads=heads, num_dimensions=num_dimensions)
            for _ in range(num_global_layers)
        ])
        self.global_ffns = nn.ModuleList([nn.Linear(emb_size, emb_size) for _ in range(num_global_layers)])
        self.global_layer_norms = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(num_global_layers)])
        self.global_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_global_layers)])

    def forward(self, x, local_mask=None, global_mask=None):
        """
        :param x: [batch_size, seq_len, emb_size] 输入特征
        :param local_mask: [batch_size, seq_len] 局部上下文的 mask
        :param global_mask: [batch_size, seq_len] 全局上下文的 mask
        :return: [batch_size, seq_len, emb_size] 输出特征
        """
        # 局部关系建模
        for i in range(self.num_local_layers):
            residual = x
            x = self.local_transformer[i](x, mask=local_mask)  # 局部注意力
            x = self.local_layer_norms[i](x + residual)  # 残差连接 + LayerNorm
            x = self.local_ffns[i](x)
            x = self.local_dropouts[i](x)

        # 全局关系建模
        for i in range(self.num_global_layers):
            residual = x
            x = self.global_transformer[i](x, mask=global_mask)  # 全局注意力
            x = self.global_layer_norms[i](x + residual)  # 残差连接 + LayerNorm
            x = self.global_ffns[i](x)
            x = self.global_dropouts[i](x)

        return x

def create_local_mask(batch_size, seq_len, block_size):
    """
    创建局部 mask，用于限制注意力范围在每个 block 内
    :param batch_size: 批次大小
    :param seq_len: 序列长度
    :param block_size: 每个 block 的大小
    :return: [batch_size, seq_len, seq_len] 局部 mask
    """
    mask = torch.zeros((batch_size, seq_len, seq_len))  # 初始化全 0 mask
    for i in range(0, seq_len, block_size):
        mask[:, i:i + block_size, i:i + block_size] = 1  # 设置 block 范围内的注意力为 1
    return mask

def create_global_mask(batch_size, seq_len):
    """
    创建全局 mask，允许所有位置的注意力
    :param batch_size: 批次大小
    :param seq_len: 序列长度
    :return: [batch_size, seq_len, seq_len] 全局 mask
    """
    return torch.ones((batch_size, seq_len, seq_len))  # 全 1 mask

class DocREModel_KD(nn.Module):
    def __init__(self, args, config, model, emb_size=1024, block_size=64, num_labels=-1, teacher_model=None):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.loss_fnt = AFLoss(gamma_pos = args.gamma_pos, gamma_neg = args.gamma_neg,)
        if teacher_model is not None:
            self.teacher_model = teacher_model
            self.teacher_model.requires_grad = False
            self.teacher_model.eval()
        else:
            self.teacher_model = None
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        #self.head_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        #self.entity_classifier = nn.Linear( config.hidden_size, 7)
        self.entity_criterion = nn.CrossEntropyLoss()
        self.bin_criterion = nn.CrossEntropyLoss()
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.projection = nn.Linear(emb_size * block_size, config.hidden_size, bias=False)
        self.classifier = nn.Linear(config.hidden_size , config.num_labels)
        self.mse_criterion = nn.MSELoss()
        # self.axial_transformer = AxialTransformer_by_entity(emb_size = config.hidden_size, dropout=0.0, num_layers=6, heads=8)
        self.axial_transformer = HierarchicalAxialTransformer(emb_size=config.hidden_size, dropout=0.0, num_local_layers=6, num_global_layers=6, heads=8)
        self.emb_size = emb_size
        self.threshold = nn.Threshold(0,0)
        self.block_size = block_size
        self.num_labels = num_labels
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        sent_embs = []
        batch_entity_embs = []
        b, seq_l, h_size = sequence_output.size()
        #n_e = max([len(x) for x in entity_pos])
        n_e = 42
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_lens = []
            '''
            sid_mask = sentid_mask[i]
            sentids = [x for x in range(torch.max(sid_mask).cpu().long() + 1)]
            local_mask  = torch.tensor([sentids] * sid_mask.size()[0] ).T
            local_mask = torch.eq(sid_mask , local_mask).long().to(sequence_output)
            sentence_embs = local_mask.unsqueeze(2) * sequence_output[i]
            sentence_embs = torch.sum(sentence_embs, dim=1)/local_mask.unsqueeze(2).sum(dim=1)
            seq_sent_embs = sentence_embs.unsqueeze(1) * local_mask.unsqueeze(2)
            seq_sent_embs = torch.sum(seq_sent_embs, dim=0)
            sent_embs.append(seq_sent_embs)
            '''

            for e in entity_pos[i]:
                #entity_lens.append(self.ent_num_emb(torch.tensor(len(e)).to(sequence_output).long()))
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            #e_emb.append(sequence_output[i, start + offset] + seq_sent_embs[start + offset])
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                     

                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        #e_emb = sequence_output[i, start + offset] + seq_sent_embs[start + offset]
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            
            pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)
            pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)


            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            #print(h_att.size())
            #ht_att = (h_att * t_att).mean(1)
            m = torch.nn.Threshold(0,0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)
            hss.append(pad_hs)
            tss.append(pad_ts)
            rss.append(pad_rs)
            batch_entity_embs.append(entity_embs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        batch_entity_embs = torch.cat(batch_entity_embs, dim=0)
        return hss, rss, tss, batch_entity_embs



    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                mention_pos=None,
                mention_hts=None,
                padded_mention=None,
                padded_mention_mask=None,
                sentid_mask=None,
                return_logits = None,
                teacher_logits = None,
                entity_types = None,
                segment_spans = None,
                negative_mask = None,
                label_loader = None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        bs, seq_len, h_size = sequence_output.size()
        bs, num_heads, seq_len, seq_len = attention.size()
        ctx_window = 300
        stride = 25
        device = sequence_output.device.index
        #ne = max([len(x) for x in entity_pos])
        ne = 42
        nes = [len(x) for x in entity_pos]
        hs_e, rs_e, ts_e, batch_entity_embs = self.get_hrt(sequence_output, attention, entity_pos, hts)
        #h_t_s = hs_e - ts_e
        #t_h_s = ts_e - hs_e
        #hxt = hs_e * ts_e
        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3))) 
        #hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e,  t_h_s], dim=3)))
        #ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e,  t_h_s], dim=3)))         
    
        b1_e = hs_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, self.emb_size * self.block_size)
        
        if negative_mask is not None:
            bl_e = bl_e * negative_mask.unsqueeze(-1)
              
        create_local_mask(args.train_batch_size,)

        feature =  self.projection(bl_e)
        feature = self.axial_transformer(feature) + feature
        label_embeddings = []
        '''
        for stp, batch in enumerate(label_loader):
            
            #label_output = self.model(batch[0].to(input_ids), batch[1].to(attention_mask), return_dict=False )[0]
            label_output = self.encode(batch[0].to(input_ids), batch[1].to(attention_mask) )[0]
            #print(label_output.size())

            #label_emb = label_output[:, 0, :]
            label_emb = label_output.mean(dim = 1)
            label_embeddings.append(label_emb)
        '''
        #label_embeddings = torch.cat(label_embeddings, dim=0).detach()
        #label_embeddings = torch.cat(label_embeddings, dim=0)

        if False:
            query = feature.view(bs, -1, self.config.hidden_size)
            query = query.permute(1, 0, 2)
            tgt_len, bsz, embed_dim = query.size()
            #batch_size, batch_seq_len, hid_dim = relation_embedding_context.size()
            key = label_embeddings.unsqueeze(1).permute(1, 0, 2)
            #input_mask = input_masks
            input_mask = torch.ones(key.size()).cuda()
            #pair, attn_weights = self.multihead_attn(query, key, key, key_padding_mask=input_mask)
            attn_feature, attn_weights = self.multihead_attn(query, key, key)
            attn_feature = attn_feature.view(bs, ne, ne, self.config.hidden_size)
            #print(pair.size())
        
        
        #logits_l = torch.matmul(feature.clone(), label_embeddings.T)
        logits_c = self.classifier(feature)
        
        self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        logits_classifier = logits_c * self_mask
        #logits_label = logits_l * self_mask
        #print(logits_e.size())
        logits_classifier = torch.cat([logits_classifier.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])
        #logits_label = torch.cat([logits_label[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])

        if labels is None:
            logits = logits_classifier.view(-1, self.config.num_labels)
            #logits = logits_classifier.view(-1, self.config.num_labels) + logits_label.view(-1, self.config.num_labels)
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(device)
            
            loss_classifier = self.loss_fnt(logits_classifier.view(-1, self.config.num_labels).float(), labels.float())
            #loss_label, _, _ = self.loss_fnt2(logits_label.float(), labels.float())
            #loss_s, loss1_s, loss2_s = self.loss_fnt(logits_seg.float(), segment_labels.float())
            #output = loss_e.to(sequence_output) + loss_s.to(sequence_output)
            output =  loss_classifier
            '''
            if entity_types is not None:
                entity_types = torch.tensor(entity_types).long().to(logits_0)
                entity_type_preds = self.entity_classifier(batch_entity_embs)
                ent_loss = self.entity_criterion(entity_type_preds, entity_types.long())
                output_0 = output_0 + 0.1*ent_loss
            '''
            if teacher_logits is not None:
                teacher_logits = torch.cat(teacher_logits, dim=0).to(logits_classifier)
                mse_loss = self.mse_criterion(logits_classifier, teacher_logits)
                output = output + 1.0 *  mse_loss

                        
        return output
