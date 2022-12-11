# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16, 2021/7/1, 2021/7/11
# @Author : Yushuo Chen, Xingyu Pan, Yupeng Hou
# @Email  : chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

import numpy as np
import torch

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        max_item_list_len (int): Max length of historical item list.
        item_list_length_field (str): Field name for item lists' length.
    """

    def __init__(self, config):
        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
        self.item_list_length_field = config['ITEM_LIST_LENGTH_FIELD']

        self.max_user_list_len = config['MAX_USER_LIST_LENGTH']
        self.user_list_length_field = config['USER_LIST_LENGTH_FIELD']

        

        #self.item_label_list_field = config['ITEM_LIST_LABEL_FIELD']
        #self.user_label_list_field = config['USER_LIST_LABEL_FIELD']

        super().__init__(config)
        if config['benchmark_filename'] is not None:
            self._benchmark_presets()

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
           then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config['benchmark_filename'] is not None:
            return
        self.logger.debug('Augmentation for sequential recommendation.')
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config['LIST_SUFFIX']
        # print(self.field2type)
        # exit(0)
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)
            else:
                setattr(self, f'{field}_list_field', field + list_suffix)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_user_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_user_list_len

                self.set_field_property(field + list_suffix, list_ftype, FeatureSource.INTERACTION, list_len)

        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        self.set_field_property(self.user_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

        self.set_field_property(self.item_label_list_field, FeatureType.TOKEN_SEQ, FeatureSource.INTERACTION, 1)
        self.set_field_property(self.user_label_list_field, FeatureType.TOKEN_SEQ, FeatureSource.INTERACTION, 1)

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug('data_augmentation')

        self._aug_presets()

        self._check_field('iid_field','uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']

        #############start to load the user sequence from item###################
        self.sort(by=[self.iid_field, self.time_field], ascending=True)
        item2uid_list = [[] for i in range(self.item_num)]
        item2ulable_list = [[] for i in range(self.item_num)]
        item2uitem_list = [[] for i in range(self.item_num)]


        #print(self.inter_feat)
        for i, (iid,uid,label,time) in enumerate(zip(self.inter_feat[self.iid_field].numpy(),self.inter_feat[self.uid_field].numpy(),
                                                self.inter_feat[self.label_field].numpy(),self.inter_feat[self.time_field].numpy())):
            item2uid_list[iid].append(uid)
            item2ulable_list[iid].append(label)
            item2uitem_list[iid].append(time)
        # print(item2uitem_list[-5:])
        # print(item2ulable_list[-5:])
        # print(item2uid_list[-5:])
        # exit(0)
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length, user_list, user_list_length = [], [], [], [], [], []

        user_seq_label_list = []
        item_seq_label_index = []

        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                item_seq_label_index.append(slice(seq_start, i))

                target_index.append(i)
                item_list_length.append(i - seq_start)

                iid = self.inter_feat[self.iid_field][i]
                time = self.inter_feat[self.time_field][i]
                po_uid = item2uitem_list[iid].index(time)
                min_puid = max(0,po_uid-self.max_user_list_len)


                user_list.append(item2uid_list[iid][min_puid:po_uid])
                user_list_length.append(po_uid-min_puid)

                user_seq_label_list.append(item2ulable_list[iid][min_puid:po_uid])

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_seq_label_index = np.array(item_seq_label_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
            self.user_list_length_field: torch.tensor(user_list_length)

        }

        #print(self.inter_feat)
        #exit(0)
        shape = (new_length, self.field2seqlen[getattr(self, f'{self.iid_field}_list_field')])
        #print(shape)
        item_label_field = getattr(self,f'item_label_list_field')
        new_dict[item_label_field] = torch.zeros(shape, dtype=torch.int64)
        for i, (index, length) in enumerate(zip(item_seq_label_index, item_list_length)):
            
            new_dict[item_label_field][i][:length] = self.inter_feat[self.label_field][index]

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                #print(list_field)
                list_len = self.field2seqlen[list_field]
                #print(self.field2seqlen)
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

            else:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.field2seqlen[list_field]
                #print(self.field2seqlen)
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                for i, (seq, length) in enumerate(zip(user_list, user_list_length)):
                    new_dict[list_field][i][:length] = torch.tensor(seq,dtype=dtype)



                user_label_field = getattr(self,f'user_label_list_field')
                new_dict[user_label_field] = torch.zeros(shape, dtype=torch.int64)
                for i, (label_seq, length) in enumerate(zip(user_seq_label_list, user_list_length)):
                    new_dict[user_label_field][i][:length] = torch.tensor(label_seq,dtype=torch.int64)
                    # print(length)
                    # print(value[index])
                    # if i>10:
                    #     exit(0)

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data
        # print(self.inter_feat[getattr(self, f'user_id_list_field')][0:10])
        # print(self.inter_feat[self.user_list_length_field][0:10])
        # exit(0)

    def _benchmark_presets(self):
        list_suffix = self.config['LIST_SUFFIX']
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        self.set_field_property(self.user_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        self.inter_feat[self.item_list_length_field] = self.inter_feat[self.item_id_list_field].agg(len)
        self.inter_feat[self.user_list_length_field] = self.inter_feat[self.user_id_list_field].agg(len)

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')

        l1_idx = (self.inter_feat[self.item_list_length_field] == 1)
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config['LIST_SUFFIX']
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat([self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]])
            elif (not field.endswith(list_suffix)) and (field != self.item_list_length_field):
                new_dict[field] = torch.cat([self.inter_feat[field], l1_inter_dict[field]])
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(local_inter_feat, self.uid_field, self.iid_field, form, value_field)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config['eval_args']['order']
        if ordering_args != 'TO':
            raise ValueError(f'The ordering args for sequential recommendation has to be \'TO\'')

        return super().build()
