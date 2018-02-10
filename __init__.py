import os
import torch
from dataset import DataSet
from model_manager import ModelManager


if __name__ == "__main__":
    emb_dim_t = int(input('please input emb_dim_t: 8 16 32 64 etc'))
    torch.manual_seed(3)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationCuda/' if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationCuda/') else 'LocationCuda/'
    dataset_name = 'foursquare'
    path = root_path + 'small/' + dataset_name + '/'
    opt = {'path': path,
           'u_vocab_file': path+ 'u.txt',
           'v_vocab_file': path + 'v.txt',
           't_vocab_file': path + 't.txt',
           'train_data_file': path + 'train.txt',
           'test_data_file': path + 'test.txt',
           'coor_nor_file': path + 'coor_nor.txt',
           'distance_file' : path + 'distance.txt',
           'train_log_file': path + 'log.txt',
           'id_offset': 1,
           'n_epoch': 80,
           'batch_size': 50,
           'data_worker': 1,
           'load_model': False,
           'emb_dim_d': 16, # for distance embedding  #best 16
           'emb_dim_v': 32,  #origin 32
           'emb_dim_t': emb_dim_t, #origin 8
           'emb_dim_u': 32,# !!!jiayi  copy from v3
           'hidden_dim': 16, #origin 16
           'save_gap': 20,
           'dropout': 0.5,
           'epoch': 80
           }
    dataset = DataSet(opt)
    manager = ModelManager(opt)
    model_type = 'birnnt'
    manager.build_model(model_type, dataset)
    print "evaluate"
    manager.evaluate(model_type, dataset)
