# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
BERT4Rec
################################################

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
    In CIKM 2019.

Reference code:
    The authors' tensorflow implementation https://github.com/FeiSun/BERT4Rec

"""

import random

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder,MLPLayers, SequenceAttLayer, ContextSeqEmbLayer
from recbole.utils import InputType
from torch.nn.init import xavier_normal_, constant_
from recbole.model.loss import IPSBCELoss


class BST(SequentialRecommender):

    def __init__(self, config, dataset):
        super(BST, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']


        self.embedding_size = config['embedding_size']
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        self.loss_type = config['loss_type']

        # load dataset info
        self.mask_token = 0
        #self.cls_token = 1

        self.types = ['user', 'item']
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        num_item_feature = len(self.item_feat.interaction.keys())

        num_user_feature = len(self.user_feat.interaction.keys())

        self.mask_user_length = int(self.mask_ratio * self.max_user_seq_length)
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        self.user_project = nn.Linear(num_user_feature*self.embedding_size,self.hidden_size)
        self.item_project = nn.Linear(num_item_feature*self.embedding_size,self.hidden_size)

        # define layers and loss
        #self.item_embedding = nn.Embedding(self.n_items + 2, self.hidden_size, padding_idx=0)  # for n_item is <MASK> token n_item+1 is the <CLS> token
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.embedding_layer = ContextSeqEmbLayer(dataset, self.embedding_size, self.pooling_mode, self.device)

        self.item_position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)
        self.user_position_embedding = nn.Embedding(self.max_user_seq_length+1, self.hidden_size)

        self.item_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.user_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.dnn_mlp_layers = MLPLayers(self.mlp_hidden_size, activation='Dice', dropout=self.dropout_prob, bn=True)
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()

        if self.loss_type == 'BCE':
            self.loss = nn.BCELoss()
        elif self.loss_type == 'IPSBCE':
            self.loss = IPSBCELoss(config['IPS_clip'])
        else:
            raise NotImplementedError('only support the BCE of IPSBCE loss')


        sparse_embedding, dense_embedding = self.embedding_layer(torch.arange(0,self.n_users), torch.arange(0,self.n_items))
        # concat the sparse embedding and float embedding
        feature_table = {}
        for type in self.types:
            feature_table[type] = []
            if sparse_embedding[type] is not None:
                feature_table[type].append(sparse_embedding[type])
            if dense_embedding[type] is not None:
                feature_table[type].append(dense_embedding[type])

            feature_table[type] = torch.cat(feature_table[type], dim=-2)
            table_shape = feature_table[type].shape
            feat_num, embedding_size = table_shape[-2], table_shape[-1]
            feature_table[type] = feature_table[type].view(table_shape[:-2] + (feat_num * embedding_size,))

        self.user_features,self.item_features = feature_table['user'],feature_table['item']

       # self.item_features,self.item_

        # we only need compute the loss at the masked position
        # try:
        #     assert self.loss_type in ['BPR', 'CE']
        # except AssertionError:
        #     raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

    def get_attention_mask(self, seq, seq_len):
        """Generate bidirectional attention mask for multi-head attention."""

        #attention_mask = (item_seq > 0).long() #[B,L]
        attention_mask = torch.zeros_like(seq,device=seq.device)
        for b,att in enumerate(attention_mask):
            attention_mask[b,0:seq_len[b]] = 1.0

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, seq, mask_length,seq_len, next_items):
        """
        Mask item sequence for training.
        """
        device = seq.device
        batch_size = seq.size(0)

        sequence_instances = seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_sequences = []
        pos_values = []
        #neg_items = []
        masked_index = []
        for m, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            #valid_length = seq_len[m].item()
            masked_sequence = instance.copy()
            pos = []
            # neg_item = []
            index_ids = []
            t = 0
            for index_id, j in enumerate(instance):
                # padding is 0, the sequence is end
                t = index_id
                if j == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos.append(j)
                    #neg_item.append(self._neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    index_ids.append(index_id) ####skip the cls vector
            pos.append(next_items[m])
            masked_sequence.append(self.mask_token)
            index_ids.append(t)

            masked_sequences.append(masked_sequence)
            pos_values.append(self._padding_sequence(pos, mask_length))
            #neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, mask_length))

        # [B Len]

        masked_sequences = torch.tensor(masked_sequences, dtype=torch.long, device=device).view(batch_size, -1)
        #attention_mask = self.get_attention_mask(masked_sequences,seq_len)
        # [B mask_len]
        pos_values = torch.tensor(pos_values, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        #neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_sequences, pos_values, masked_index

    # def reconstruct_test_data(self, seq, seq_len):
    #     """
    #     Add mask token at the last position according to the lengths of item_seq
    #     """
    #     masked_list = []
    #     #padding = torch.zeros(seq.size(0), dtype=torch.long, device=seq.device)  # [B]
    #     #seq = torch.cat((padding.unsqueeze(-1),seq), dim=-1)  # [B max_len+1]
    #     for batch_id, last_position in enumerate(seq_len):
    #         mask = [1]*last_position.item() + [0] * (seq.size(1)-last_position)
    #         masked_list.append(mask)
    #     return seq, torch.FloatTensor(masked_list,device = seq.device)

    def forward(self, user, item_seq, next_items, user_seq, item_seq_len, user_seq_len):
        item_seq_next = torch.cat((item_seq, next_items.unsqueeze(1)), dim=-1)
        user_seq_next = torch.cat((user_seq, user.unsqueeze(1)), dim=-1)

        sparse_embedding, dense_embedding = self.embedding_layer(user_seq_next, item_seq_next)
        # concat the sparse embedding and float embedding
        feature_table = {}
        for type in self.types:
            feature_table[type] = []
            if sparse_embedding[type] is not None:
                feature_table[type].append(sparse_embedding[type])
            if dense_embedding[type] is not None:
                feature_table[type].append(dense_embedding[type])

            feature_table[type] = torch.cat(feature_table[type], dim=-2)
            table_shape = feature_table[type].shape
            feat_num, embedding_size = table_shape[-2], table_shape[-1]
            feature_table[type] = feature_table[type].view(table_shape[:-2] + (feat_num * embedding_size,))

        user_feat_list,target_user_feat_emb = self.user_project(feature_table['user']).split([self.max_user_seq_length+1, 1], dim=1)
        item_feat_list, target_item_feat_emb = self.item_project(feature_table['item']).split([self.max_seq_length+1, 1], dim=1)

        position_ids = torch.arange(self.max_seq_length+1, dtype=torch.long, device=item_seq_next.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.item_position_embedding(position_ids)

        input_emb = item_feat_list + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq,item_seq_len)
        #print(extended_attention_mask[0:2])
        item_output = self.item_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_output[-1]

        position_ids = torch.arange(self.max_user_seq_length+1, dtype=torch.long, device=user_seq_next.device)
        position_ids = position_ids.unsqueeze(0).expand_as(user_seq)
        position_embedding = self.user_position_embedding(position_ids)

        input_emb = user_feat_list + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(user_seq, user_seq_len)
        user_output = self.user_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        user_output = user_output[-1]

        #exit(0)
        return item_output,target_item_feat_emb,user_output,target_user_feat_emb  # [B H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user_seq = interaction[self.USER_SEQ]

        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_seq_len = interaction[self.USER_SEQ_LEN]

        next_items = interaction[self.POS_ITEM_ID]

        label = interaction[self.LABEL_FIELD]



        masked_item_seq, pos_items, masked_item_index = self.reconstruct_train_data(item_seq,self.mask_item_length, item_seq_len, next_items)
        masked_user_seq, pos_users, masked_user_index = self.reconstruct_train_data(user_seq,self.mask_user_length, user_seq_len,  user)
        # print(masked_item_seq.shape)
        # print(masked_item_seq[0:2])
        # print(pos_items[0:2])
        # print(masked_item_index[0:2])

        item_output,target_item_feat_emb,user_output,target_user_feat_emb = self.forward(user=user, item_seq=masked_item_seq, next_items=next_items, user_seq=masked_user_seq,
                                                                                         item_seq_len=item_seq_len+1, user_seq_len=user_seq_len+1)


        pred_item_index_map = self.multi_hot_embed(masked_item_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_item_index_map = pred_item_index_map.view(masked_item_index.size(0), masked_item_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        # print(pred_item_index_map.shape)
        # print(item_output.shape)
        seq_output = torch.bmm(pred_item_index_map, item_output)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        test_item_emb = self.item_project(self.item_features).detach()  # [item_num H]

        item_logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        item_targets = (masked_item_index > 0).float().view(-1)  # [B*mask_len]



        pred_user_index_map = self.multi_hot_embed(masked_user_index, masked_user_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_user_index_map = pred_user_index_map.view(masked_user_index.size(0), masked_user_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_user_index_map, user_output)  # [B mask_len H]

        test_user_emb = self.user_project(self.user_features).detach()  # [item_num H]
        user_logits = torch.matmul(seq_output, test_user_emb.transpose(0, 1))  # [B mask_len item_num]
        user_targets = (masked_user_index > 0).float().view(-1)  # [B*mask_len]



        # print(masked_user_seq.shape)
        # print(target_item_feat_emb.shape)
        item_emb = target_item_feat_emb[:,0,:]
        #item_emb = torch.cat((torch.mean(user_output*((masked_user_seq>0).float().view(masked_user_seq.size(0),-1,1)),dim=1,keepdim=False),target_item_feat_emb[:,0,:]),dim=-1)

        #user_emb = target_user_feat_emb[:,0,:]
        user_emb = torch.cat((torch.mean(item_output*((masked_item_seq>0).float().view(masked_item_seq.size(0),-1,1)),dim=1,keepdim=False),target_user_feat_emb[:,0,:]),dim=-1)

        #in_r = torch.cat([user_emb, item_emb, user_emb * item_emb], dim=-1)
        in_r = torch.cat([user_emb, item_emb], dim=-1)
        out_r = self.dnn_mlp_layers(in_r)
        preds = self.dnn_predict_layers(out_r)
        preds = self.sigmoid(preds).squeeze(1)

        item_mlm_loss = torch.sum(loss_fct(item_logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * item_targets) \
                        / torch.sum(item_targets)

        user_mlm_loss = torch.sum(loss_fct(user_logits.view(-1, test_user_emb.size(0)), pos_users.view(-1)) * user_targets) \
                        / torch.sum(user_targets)

        if self.loss_type == "BCE":
            loss = self.loss(preds, label)
        else:
            #p_item_score = torch.sum((torch.mean(item_output*((masked_item_seq>0).float().view(masked_item_seq.size(0),-1,1)),dim=1,keepdim=False))*target_item_feat_emb[:,0,:],dim=1,keepdim=True)
            #p_user_score = torch.sum((torch.mean(user_output*((masked_user_seq>0).float().view(masked_user_seq.size(0),-1,1)),dim=1,keepdim=False))*target_user_feat_emb[:,0,:],dim=1,keepdim=True)

            p_item_score = torch.matmul((torch.mean(item_output*((masked_item_seq>0).float().view(masked_item_seq.size(0),-1,1)),dim=1,keepdim=False)),test_item_emb.transpose(0, 1))
            p_item_score = torch.softmax(p_item_score,dim=-1)
            p_item_score = p_item_score.gather(dim=1,index=next_items.unsqueeze(-1))

            p_user_score = torch.matmul((torch.mean(user_output*((masked_user_seq>0).float().view(masked_user_seq.size(0),-1,1)),dim=1,keepdim=False)),test_user_emb.transpose(0, 1))
            p_user_score = torch.softmax(p_user_score,dim=-1)
            p_user_score = p_user_score.gather(dim=1,index=user.unsqueeze(-1))

            # print(p_item_score)
            # print(p_user_score)

            loss = self.loss(preds,label,1-(1-p_item_score)*(1-p_user_score))
           # loss = self.loss(preds,label,torch.sigmoid(p_item_score)*torch.sigmoid(p_user_score))

        total_loss = loss

        #return preds.squeeze(1)

        return total_loss


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        user_seq = interaction[self.USER_SEQ]
        user = interaction[self.USER_ID]

        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_seq_len = interaction[self.USER_SEQ_LEN]

        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]

        padding = torch.zeros(user_seq.size(0), dtype=torch.long, device=user_seq.device)  # [B]
        user_seq = torch.cat((user_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        # item_seq,mask_item = self.reconstruct_test_data(item_seq, item_seq_len)
        # user_seq,mask_user = self.reconstruct_test_data(user_seq, user_seq_len)
        #mask_user = (user_seq > 0).float()
        #mask_item = (item_seq > 0).float()

        item_output,target_item_feat_emb,user_output,target_user_feat_emb = self.forward(user=user, item_seq=item_seq, next_items=test_item, user_seq=user_seq,
                                                                                         item_seq_len=item_seq_len,user_seq_len=user_seq_len)

        #item_emb = torch.cat((torch.mean(user_output*((user_seq>0).float().view(user_seq.size(0),-1,1)),dim=1,keepdim=False),target_item_feat_emb[:,0,:]),dim=-1)
        item_emb = target_item_feat_emb[:,0,:]
        #user_emb = target_user_feat_emb[:,0,:]
        user_emb = torch.cat((torch.mean(item_output*((item_seq>0).float().view(item_seq.size(0),-1,1)),dim=1,keepdim=False),target_user_feat_emb[:,0,:]),dim=-1)

        #in_r = torch.cat([user_emb, item_emb, user_emb * item_emb], dim=-1)
        in_r = torch.cat([user_emb, item_emb], dim=-1)
        out_r = self.dnn_mlp_layers(in_r)
        preds = self.dnn_predict_layers(out_r)
        scores = self.sigmoid(preds).squeeze(1)
        #scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    # def full_sort_predict(self, interaction):
    #     item_seq = interaction[self.ITEM_SEQ]
    #     item_seq_len = interaction[self.ITEM_SEQ_LEN]
    #     item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
    #     seq_output = self.forward(item_seq)
    #     seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
    #     test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
    #     scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
    #     return scores
