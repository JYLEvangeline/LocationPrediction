import torch
from utils import format_list_to_string
from birnn import BiRNN
from trainer import Trainer
from evaluator import Evaluator
from birnnt import BiRNNT
#from SERM import SERM
from STRNN import STRNN

class ModelManager:
    def __init__(self, opt):
        self.opt = opt
        self.model_path = opt['path'] + 'model/'

    def init_model(self, model_type, u_size, v_size, t_size,distance):
        if model_type == 'birnn':
            return BiRNN(v_size, self.opt['emb_dim_v'], self.opt['hidden_dim'])
        elif model_type == 'birnnt':
            return BiRNNT(v_size, t_size, u_size, self.opt['emb_dim_v'], self.opt['emb_dim_t'],self.opt['emb_dim_u'],self.opt['emb_dim_d'],self.opt['hidden_dim'],distance)
        elif model_type == 'serm':
            w_size = 1
            #return SERM(u_size, v_size, t_size, w_size)
        elif model_type == 'strnn':
            return STRNN(u_size, v_size)


    def build_model(self, model_type, dataset):
        print 'build_model'
        model = self.init_model(model_type, dataset.u_vocab.size(), dataset.v_vocab.size(), dataset.t_vocab.size(),dataset.distance)
        if self.opt['load_model']:
            self.load_model(model, model_type, self.opt['epoch'])
            train_time = 0.0
            return model, train_time
        trainer = Trainer(model, self.opt, model_type)
        train_time = trainer.train(dataset.train_loader, self)
        print 'train_time: ', train_time
        return model, train_time

    def evaluate(self, model_type, dataset):
        print 'evaluate'
        model = self.init_model(model_type, dataset.u_vocab.size(), dataset.v_vocab.size(), dataset.t_vocab.size(),dataset.distance)
        self.load_model(model, model_type, self.opt['epoch'])
        evaluator = Evaluator(model, self.opt)
        evaluator.eval(dataset.test_loader)

    def get_model_name(self, model_type, epoch):
        emb_dim_v = self.opt['emb_dim_v']
        hidden_dim = self.opt['hidden_dim']
        batch_size = self.opt['batch_size']
        n_epoch = self.opt['n_epoch']
        dp = self.opt['dropout']
        attributes = [model_type, 'DH', hidden_dim, 'DV', emb_dim_v, 'B', batch_size, 'NE', n_epoch, 'E', epoch,
                      'dp', dp]
        model_name = format_list_to_string(attributes, '_')
        return model_name + '.model'

    def load_model(self, model, model_type, epoch):
        model_name = self.get_model_name(model_type, epoch)
        file_name = self.model_path + model_name
        model.load_state_dict(torch.load(file_name))

    def save_model(self, model, model_type, epoch):
        model_name = self.get_model_name(model_type, epoch)
        file_name = self.model_path + model_name
        torch.save(model.state_dict(), file_name)
