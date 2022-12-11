import numpy as np
import pandas as pd
import os
#cols = ['user_id:token', 'item_id:token','rating:float', 'timestamp:float']
def SplitTDMData(Dataset,threshold,item_val,user_val,max_user_list_len=30,max_item_list_len = 50,
                 valid_ratio=0.1,test_ratio=0.1,user_id = "user_id:token",
                 item_id = "item_id:token",label_id = "rating:float",time_id = "timestamp:float"):
    #Dataset = "Amazon_Beauty"
    inter_file = os.path.join("datasets",Dataset,Dataset+".inter")


    frames = pd.read_csv(inter_file,delimiter='\t',dtype={"item_id:token":str,"user_id:token":str},usecols=[user_id,item_id,label_id,time_id])
    print(frames.head())
    # if Dataset == 'yelp':
    #     review_field,uid_field,iid_field,label_field,_,_,_,time_field = frames.columns
    # else:
    frames.columns = ["user_id:token","item_id:token","label:float","timestamp:float"]
    uid_field,iid_field,label_field,time_field = frames.columns


    frames.drop_duplicates(subset=[iid_field,uid_field],keep='first',inplace=True)
    print(uid_field,iid_field,label_field,time_field)
    item_num, user_num = len(frames[iid_field].unique()),len(frames[uid_field].unique())
    print("origin:----------item number: %d user number:%d total interactions:%d"%(item_num,user_num,len(frames)))

    frames = frames[frames[label_field]>=threshold]

    ###filter items
    itemLen = frames.groupby(iid_field).size()  # groupby itemID and get size of each item
    remain_items =  (itemLen[itemLen >= item_val].index).values
    frames = frames[frames[iid_field].isin(remain_items)]
    print("processing item val")
    frames.sort_values(by=[iid_field], ascending=True,inplace=True)
    print(frames.head())
    userLen = frames.groupby(uid_field).size()

    remain_users =  (userLen[userLen >= user_val].index).values
    frames = frames[frames[uid_field].isin(remain_users)]

    #frames.reset_index(drop=True, inplace=True)
    #frames.drop_duplicates(subset=[iid_field,uid_field],keep='first',inplace=True)
    item_num, user_num = len(frames[iid_field].unique()),len(frames[uid_field].unique())
    times = len(frames[time_field].unique())
    frames.sort_values(by=[uid_field], ascending=True,inplace=True)
    print("processing user val")
    print(frames.head())
    print("item number: %d user number:%d time number:%d total interactions:%d"%(item_num,user_num,times,len(frames)))


    frames.sort_values(by=[iid_field, time_field], ascending=True,inplace=True)
    item2uid_list = {i:[] for i in frames[iid_field].values}
    item2uidtime_list = {i:[] for i in frames[iid_field].values}


    #print(self.inter_feat)
    for i, (iid,uid,time) in enumerate(zip(frames[iid_field].values,frames[uid_field].values,frames[time_field].values)):
        item2uid_list[iid].append(uid)
        item2uidtime_list[iid].append(time)

    frames.sort_values(by=[uid_field, time_field], ascending=True,inplace=True)

    last_uid = None
    uid_list, item_list, target_index, item_list_length, user_list, user_list_length, label_list, time_list = [], [], [], [], [], [], [],[]
    seq_start = 0
    observe_probability = {}
    for i, uid in enumerate(frames[uid_field].values):
        time = frames.iloc[i,3]
        iid = frames.iloc[i,1]
        po_uid = item2uidtime_list[iid].index(time)
        # print(last_uid)
        # print(uid)
        if last_uid != uid:
            last_uid = uid
            seq_start = i
        else:
            if i - seq_start > max_item_list_len:
                seq_start += 1
            if po_uid != 0:
                #print("why?")
                if (uid,iid) not in observe_probability:
                    observe_probability[(uid,iid)] = 0
                observe_probability[(uid,iid)] += 1
                time_list.append(time)
                uid_list.append(uid)
                item_list.append(' '.join(list(frames.iloc[slice(seq_start, i),1])))
                target_index.append(iid)
                item_list_length.append(i - seq_start)
                label_list.append(1)
                #po_uid = item2uid_list[iid].index(uid)
                min_puid = max(0,po_uid-max_user_list_len)
                user_list.append(' '.join(item2uid_list[iid][min_puid:po_uid]))
                user_list_length.append(po_uid-min_puid)

    new_datas = pd.DataFrame({iid_field:target_index,uid_field:uid_list,"label:float":label_list,
                              "item_id_list:token_seq":item_list,"user_id_list:token_seq":user_list,"timestamp:float":time_list})
    time_field = "timestamp:float"
    new_datas.sort_values(by=[time_field], ascending=True,inplace=True)
    print(new_datas.head())

    rows, cols = new_datas.shape

    # item_inverse_probability_list = {i:0 for i in new_datas[iid_field].unique()}
    # max_prob = 0
    # for i in new_datas[iid_field].unique():
    #     for u in new_datas[uid_field].unique():
    #         if (u,i) in observe_probability:
    #             item_inverse_probability_list[i] += observe_probability[(u,i)]
    #     if max_prob < item_inverse_probability_list[i]:
    #         max_prob = item_inverse_probability_list[i]
    #         # else:
    #         #     item_inverse_probability_list[i].append(0)
    # # for u, i in zip(frames[uid_field].unique().values,frames[iid_field].unique().values):
    # #     item_inverse_probability_list[i].append(observe_probability[(u,i)])
    # item_inverse_probability = {i:item_inverse_probability_list[i]/max_prob for i in frames[iid_field].unique()}
    # print(item_inverse_probability)
    split_index_1 = int(rows * valid_ratio)
    split_index_2 = int(rows * test_ratio)
    #数据分割
    train_index = len(new_datas)-split_index_1-split_index_2
    train_time = new_datas.iloc[train_index, -1]
    for i in range(train_index+1,len(new_datas)):
        if new_datas.iloc[i,-1] == train_time:
            train_index += 1



    #data_test:pd.DataFrame = new_datas.iloc[valid_index+1:, :]
    #data_validate:pd.DataFrame = new_datas.iloc[train_index+1:valid_index+1, :]
    data_train:pd.DataFrame = new_datas.iloc[0:train_index+1,:]
    data_test:pd.DataFrame = new_datas.iloc[train_index+1:,:]

    resample_rate = 0.5
    train_items = data_train[iid_field].unique()
    train_users = data_train[uid_field].unique()

    # data_validate = data_validate[data_validate[iid_field].isin(train_items)]
    # data_validate = data_validate[data_validate[uid_field].isin(train_users)]
    # data_validate.sort_values(by=[time_field], ascending=True,inplace=True)

    data_test = data_test[data_test[iid_field].isin(train_items)]
    data_test = data_test[data_test[uid_field].isin(train_users)]
    data_test.sort_values(by=[time_field], ascending=True,inplace=True)

    item_key, item_freq = np.unique(data_test[iid_field], return_counts=True)
    #print(item_key)
    #print(item_freq)
    pscore = (item_freq.max() / item_freq) ** 0.005
    _range = pscore.max() - pscore.min()
    pscore = (pscore - pscore.min()) / _range

    item_map_pscore = pd.Series(
        data=pscore, index=item_key
    )
    total_length = len(data_test)
    total_index = np.arange(total_length)
    test_index = []
    #print(len(data_test))
    #print(item_map_pscore)
    for i, next in enumerate(data_test[iid_field]):
        next_pscore = item_map_pscore[next]
        x = np.random.uniform(0, 1)
        if x < next_pscore:
            test_index.append(i)
    #print(test_index)
    data_test = data_test[np.in1d(total_index, test_index)]
    data_test.sort_values(by=[time_field], ascending=True,inplace=True)
    data_test.reset_index(drop=True, inplace=True)
    print("test len:",len(data_test))

    valid_index = int(len(data_test)*(valid_ratio/(valid_ratio+test_ratio)))
    valid_time = data_test.iloc[valid_index, -1]
    for i in range(valid_index+1,len(new_datas)):
        if new_datas.iloc[i,-1] == valid_time:
            valid_index += 1
    val = data_test.iloc[:valid_index+1,:]
    test = data_test.iloc[valid_index+1:,:]
    # print(data_test.head())
    # exit(0)
    # #数据保存
    test.to_csv(os.path.join(Dataset,Dataset+".test.inter"), index=False,sep='\t')
    val.to_csv(os.path.join(Dataset,Dataset+".valid.inter"), index=False,sep='\t')
    data_train.to_csv(os.path.join(Dataset,Dataset+".train.inter"), index=False,sep='\t')
    print(Dataset+" split complete!")
    print("train:%d valid:%d test:%d"%(len(data_train),len(val),len(test)))
    item_num, user_num = len(data_train[iid_field].unique()),len(data_train[uid_field].unique())
    print("after:----------item number: %d user number:%d total interactions:%d"%(item_num,user_num,len(new_datas)))

if __name__ == "__main__":
    SplitTDMData("Amazon_Beauty",threshold=4,item_val=5,user_val=5,valid_ratio=0.2,test_ratio=0.3)
    SplitTDMData("Amazon_Digital_Music",threshold=4,item_val=5,user_val=5,valid_ratio=0.2,test_ratio=0.3)
    SplitTDMData("mind_small",threshold=1,item_val=5,user_val=5,valid_ratio=0.2,test_ratio=0.3,label_id="label:float")

