from Dataset_Traj import Dataset_Traj, Dataset_Traj_no_padding
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
import json
import os

data_dict = {'Traj':Dataset_Traj_no_padding}

def data_provider(args):
    Data = data_dict[args.data]

    if args.flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid



    if args.data == 'Traj':
        data_set = Data(
            root_path = args.root_path,
            flag = args.flag,
            city = args.city
        )
    

    data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

    return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dataset test")

    parser.add_argument('--data', type=str, required=True, default='Traj', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/yuxie/public/Dataset/bj_raw_data/bj/traj_bj_11.csv', help='root path of the data file')
    parser.add_argument('--city', type=str, default='bj', help='city of the dataset')
    parser.add_argument('--embedding_model',type=str,default='HHGCLV3',help='road_embedding_model')
    parser.add_argument('--flag', type=str,default='train')
    parser.add_argument('--batch_size', type=int,default=1)
    parser.add_argument('--num_workers',type=int,default=0)

    args = parser.parse_args()


    root_path = '/home/yuxie/LLM_SFT_Traj/data/Traj_bj'

    
    train_data_list = []
    train_dataloader = data_provider(args)
    for i, batch in tqdm(enumerate(train_dataloader), desc="Process Train dataset".format(0), total=len(train_dataloader)):
        temp_train_data_dict = {}
        path, traj_minute_indexs, traj_week_indexs, traj_len, traj_eta, traj_eta_list = batch

        temp_train_data_dict['path'] = path.tolist()[0]
        temp_train_data_dict['traj_minute_indexs'] = traj_minute_indexs.tolist()[0]
        temp_train_data_dict['traj_week_indexs'] = traj_week_indexs.tolist()[0]
        temp_train_data_dict['traj_len'] = traj_len.tolist()
        temp_train_data_dict['traj_eta'] = traj_eta.tolist()
        temp_train_data_dict['traj_eta_list'] = traj_eta_list.tolist()

        train_data_list.append(temp_train_data_dict)


    with open(os.path.join(root_path, 'train_no_padding.json'), 'w') as outfile:
        json.dump(train_data_list, outfile)

    del train_data_list

    '''

    val_data_list = []
    val_dataloader = data_provider(args)
    for i, batch in tqdm(enumerate(val_dataloader), desc="Process val dataset".format(0), total=len(val_dataloader)):
        temp_val_data_dict = {}
        path, traj_minute_indexs, traj_week_indexs, traj_len, traj_eta, traj_eta_list = batch

        temp_val_data_dict['path'] = path.tolist()[0]
        temp_val_data_dict['traj_minute_indexs'] = traj_minute_indexs.tolist()[0]
        temp_val_data_dict['traj_week_indexs'] = traj_week_indexs.tolist()[0]
        temp_val_data_dict['traj_len'] = traj_len.tolist()
        temp_val_data_dict['traj_eta'] = traj_eta.tolist()
        temp_val_data_dict['traj_eta_list'] = traj_eta_list.tolist()

        val_data_list.append(temp_val_data_dict)

    with open(os.path.join(root_path, 'val_no_padding.json'), 'w') as outfile:
        json.dump(val_data_list, outfile)

    del val_data_list

    test_data_list = []
    test_dataloader = data_provider(args)
    for i, batch in tqdm(enumerate(test_dataloader), desc="Process test dataset".format(0), total=len(test_dataloader)):
        temp_test_data_dict = {}
        path, traj_minute_indexs, traj_week_indexs, traj_len, traj_eta, traj_eta_list = batch

        temp_test_data_dict['path'] = path.tolist()[0]
        temp_test_data_dict['traj_minute_indexs'] = traj_minute_indexs.tolist()[0]
        temp_test_data_dict['traj_week_indexs'] = traj_week_indexs.tolist()[0]
        temp_test_data_dict['traj_len'] = traj_len.tolist()
        temp_test_data_dict['traj_eta'] = traj_eta.tolist()
        temp_test_data_dict['traj_eta_list'] = traj_eta_list.tolist()

        test_data_list.append(temp_test_data_dict)

    with open(os.path.join(root_path, 'test_no_padding.json'), 'w') as outfile:
        json.dump(test_data_list, outfile)
    del test_data_list
    '''

    
