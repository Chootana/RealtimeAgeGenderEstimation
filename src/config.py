import torch


class Config():
    def __init__(self):
        self.batch_size = 50
        self.input_size = 64
        self.num_epochs = 5
        self.learning_rate = 0.0015
        self.weight_decay = 1e-4
        self.augmented = False
        self.load_pretrained = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # lr_scheduler
        self.step_size = 30
        self.gamma = 0.1
