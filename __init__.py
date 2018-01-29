import os
import torch
from dataset import DataSet
from model_manager import ModelManager


if __name__ == "__main__":
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
           'train_log_file': path + 'log.txt',
           'id_offset': 1,
           'n_epoch': 80,
           'batch_size': 50,
           'data_worker': 1,
           'load_model': False,
           'emb_dim_v': 32,
           'emb_dim_t': 8,
           'hidden_dim': 16,
           'save_gap': 20,
           'dropout': 0.5,
           'epoch': 80
           }
    dataset = DataSet(opt)
    manager = ModelManager(opt)
    model_type = 'birnn'
    manager.build_model(model_type, dataset)
    print "evaluate"
    manager.evaluate(model_type, dataset)