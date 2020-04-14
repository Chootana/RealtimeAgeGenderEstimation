import torch


class Config():
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_base_path = args['data_base_path']        
        self.batch_size = args['batch_size']
        self.input_size = args['input_size']
        self.num_epochs = args['num_epochs']
        self.learning_rate = args['learning_rate']
        self.weight_decay = args['weight_decay']
        self.augmented = args['augmented']
        self.load_pretrained = args['load_pretrained']

        # lr_scheduler
        self.step_size = args['step_size']
        self.gamma = args['gamma']
