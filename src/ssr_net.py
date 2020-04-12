import torch
from torch import nn
from torch.nn import init

import time


class Stream1(nn.Module):
    def __init__(self, in_channels, last_block=False):
        super(Stream1, self).__init__()
        modules = []
        modules.append(nn.Conv2d(in_channels, 32, 3, 1, 1))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU())
        if not last_block:
            modules.append(nn.AvgPool2d(2, 2))
        
        self.net_block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.net_block(x)
        return x

class Stream2(nn.Module):
    def __init__(self, in_channels, last_block=False):
        super(Stream2, self).__init__()
        modules = []
        modules.append(nn.Conv2d(in_channels, 16, 3, 1, 1))
        modules.append(nn.BatchNorm2d(16))
        modules.append(nn.Tanh())
        if not last_block:
            modules.append(nn.MaxPool2d(2, 2))
        
        self.net_block = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.net_block(x)
        return x
        
class FBStream1(nn.Module):
    def __init__(self, avg_pool_size, last_block=False):
        super(FBStream1, self).__init__()
    
        modules = []
        modules.append(nn.Conv2d(32, 10, 1, padding=0))
        modules.append(nn.ReLU())
        if not last_block:
            modules.append(nn.AvgPool2d(avg_pool_size, avg_pool_size))
        
        self.net_block = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.net_block(x)
        return x

class FBStream1PB(nn.Module):
    def __init__(self, fc_size, stage_num):
        super(FBStream1PB, self).__init__()
        
        modules = []
        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(fc_size, stage_num))
        modules.append(nn.ReLU())
        
        self.net_block = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.net_block(x)
        return x

class FBStream2(nn.Module):
    def __init__(self, avg_pool_size, last_block=False):
        super(FBStream2, self).__init__()
    
        modules = []
        modules.append(nn.Conv2d(16, 10, 1, padding=0))
        modules.append(nn.ReLU())
        if not last_block:
            modules.append(nn.AvgPool2d(avg_pool_size, avg_pool_size))
        
        self.net_block = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.net_block(x)
        return x

class FBStream2PB(nn.Module):
    def __init__(self, fc_size, stage_num):
        super(FBStream2PB, self).__init__()
        
        modules = []
        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(fc_size, stage_num))
        modules.append(nn.ReLU())
        
        self.net_block = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.net_block(x)
        return x  

class FBProb(nn.Module):
    def __init__(self, stage_num):
        super(FBProb, self).__init__()
        
        modules = []
        modules.append(nn.Linear(stage_num, 2*stage_num))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(2*stage_num, stage_num))
        modules.append(nn.ReLU())
        
        self.net_block = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.net_block(x)
        return x    

class FBIndexOffsets(nn.Module):
    def __init__(self, stage_num):
        super(FBIndexOffsets, self).__init__()
        
        modules = []
        modules.append(nn.Linear(stage_num, 2*stage_num))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(2*stage_num, stage_num))
        modules.append(nn.Tanh())
        
        self.net_block = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.net_block(x)
        return x     

class FBDeltaK(nn.Module):
    def __init__(self, fc_size):
        super(FBDeltaK, self).__init__()
        
        modules = []
        modules.append(nn.Linear(fc_size, 1))
        modules.append(nn.Tanh())
        
        self.net_block = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.net_block(x)
        return x    

class SSRNet(nn.Module):
    def __init__(self, stage_num=[3, 3, 3], image_size=64,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(SSRNet, self).__init__()
        
        assert len(stage_num)==3, 'len(stage_num): {} != 3'.format(len(stage_num))
        
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        
        self.stream1_stage3 = Stream1(in_channels=3)
        self.stream1_stage2 = Stream1(in_channels=32)
        self.stream1_stage1_1 = Stream1(in_channels=32)
        self.stream1_stage1_2 = Stream1(in_channels=32, last_block=True)

        self.stream2_stage3 = Stream2(in_channels=3)
        self.stream2_stage2 = Stream2(in_channels=16)
        self.stream2_stage1_1 = Stream2(in_channels=16)
        self.stream2_stage1_2 = Stream2(in_channels=16, last_block=True)
        
        # FB stream1
        self.fusion_block_stream1_stage3 = FBStream1(avg_pool_size=8)
        self.fusion_block_stream1_stage3_prediction_block = FBStream1PB(fc_size=10*4*4, stage_num=self.stage_num[2])
        
        self.fusion_block_stream1_stage2 = FBStream1(avg_pool_size=4)
        self.fusion_block_stream1_stage2_prediction_block = FBStream1PB(fc_size=10*4*4, stage_num=self.stage_num[1])
        
        self.fusion_block_stream1_stage1 = FBStream1(avg_pool_size=2, last_block=True)
        self.fusion_block_stream1_stage1_prediction_block = FBStream1PB(fc_size=10*8*8, stage_num=self.stage_num[0])
        
        # FB stream2
        self.fusion_block_stream2_stage3 = FBStream2(avg_pool_size=8)
        self.fusion_block_stream2_stage3_prediction_block = FBStream2PB(fc_size=10*4*4, stage_num=self.stage_num[2])
    
        self.fusion_block_stream2_stage2 = FBStream2(avg_pool_size=4)
        self.fusion_block_stream2_stage2_prediction_block = FBStream2PB(fc_size=10*4*4, stage_num=self.stage_num[1])
    
        self.fusion_block_stream2_stage1 = FBStream2(avg_pool_size=2, last_block=True)
        self.fusion_block_stream2_stage1_prediction_block = FBStream2PB(fc_size=10*8*8, stage_num=self.stage_num[0])
        
        self.stage3_prob = FBProb(stage_num=self.stage_num[2])
        self.stage3_index_offsets = FBIndexOffsets(stage_num=self.stage_num[2])
        self.stage3_delta_k = FBDeltaK(fc_size=10*4*4)
        
        self.stage2_prob = FBProb(stage_num=self.stage_num[1])
        self.stage2_index_offsets = FBIndexOffsets(stage_num=self.stage_num[1])
        self.stage2_delta_k = FBDeltaK(fc_size=10*4*4)
        
        self.stage1_prob = FBProb(stage_num=self.stage_num[0])
        self.stage1_index_offsets = FBIndexOffsets(stage_num=self.stage_num[0])
        self.stage1_delta_k = FBDeltaK(fc_size=10*8*8)

        # init parameters
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, image):
        
        # Stream1
        feature_stream1_stage3 = self.stream1_stage3(image)
        feature_stream1_stage2 = self.stream1_stage2(feature_stream1_stage3)
        feature_stream1_stage1 = self.stream1_stage1_2(self.stream1_stage1_1((feature_stream1_stage2)))
        
        # Stream2
        feature_stream2_stage3 = self.stream2_stage3(image)
        feature_stream2_stage2 = self.stream2_stage2(feature_stream2_stage3)
        feature_stream2_stage1 = self.stream2_stage1_2(self.stream2_stage1_1((feature_stream2_stage2)))
        
        # Stream1 fusion block before PB
        feature_stream1_stage3_before_PB = self.fusion_block_stream1_stage3(feature_stream1_stage3)
        feature_stream1_stage2_before_PB = self.fusion_block_stream1_stage2(feature_stream1_stage2)
        feature_stream1_stage1_before_PB = self.fusion_block_stream1_stage1(feature_stream1_stage1)
        
        # Stream2 fusion block before PB
        feature_stream2_stage3_before_PB = self.fusion_block_stream2_stage3(feature_stream2_stage3)
        feature_stream2_stage2_before_PB = self.fusion_block_stream2_stage2(feature_stream2_stage2)
        feature_stream2_stage1_before_PB = self.fusion_block_stream2_stage1(feature_stream2_stage1)

        # Stream1 [Batch, Cout, Hout, Wout] -> [Batch, -1]        
        embedding_stream1_stage3_before_PB = feature_stream1_stage3_before_PB.view(feature_stream1_stage3_before_PB.size(0), -1)
        embedding_stream1_stage2_before_PB = feature_stream1_stage2_before_PB.view(feature_stream1_stage2_before_PB.size(0), -1)
        embedding_stream1_stage1_before_PB = feature_stream1_stage1_before_PB.view(feature_stream1_stage1_before_PB.size(0), -1)

        # Stream2 [Batch, Cout, Hout, Wout] -> [Batch, -1]    
        embedding_stream2_stage3_before_PB = feature_stream2_stage3_before_PB.view(feature_stream2_stage3_before_PB.size(0), -1)
        embedding_stream2_stage2_before_PB = feature_stream2_stage2_before_PB.view(feature_stream2_stage2_before_PB.size(0), -1)
        embedding_stream2_stage1_before_PB = feature_stream2_stage1_before_PB.view(feature_stream2_stage1_before_PB.size(0), -1)
        
        # delta_k
        stage3_delta_k = self.stage3_delta_k(torch.mul(embedding_stream1_stage3_before_PB, embedding_stream2_stage3_before_PB))
        stage2_delta_k = self.stage2_delta_k(torch.mul(embedding_stream1_stage2_before_PB, embedding_stream2_stage2_before_PB))
        stage1_delta_k = self.stage1_delta_k(torch.mul(embedding_stream1_stage1_before_PB, embedding_stream2_stage1_before_PB))

        # fusion PB1 * PB2
        embedding_stage3_after_PB = torch.mul(self.fusion_block_stream1_stage3_prediction_block(embedding_stream1_stage3_before_PB),
                                              self.fusion_block_stream2_stage3_prediction_block(embedding_stream2_stage3_before_PB))
        embedding_stage2_after_PB = torch.mul(self.fusion_block_stream1_stage2_prediction_block(embedding_stream1_stage2_before_PB),
                                              self.fusion_block_stream2_stage2_prediction_block(embedding_stream2_stage2_before_PB))
        embedding_stage1_after_PB = torch.mul(self.fusion_block_stream1_stage1_prediction_block(embedding_stream1_stage1_before_PB),
                                              self.fusion_block_stream2_stage1_prediction_block(embedding_stream2_stage1_before_PB))
        
        # prob
        prob_stage3 = self.stage3_prob(embedding_stage3_after_PB)
        prob_stage2 = self.stage2_prob(embedding_stage2_after_PB)
        prob_stage1 = self.stage1_prob(embedding_stage1_after_PB)
        
        # index_offsets
        index_offset_stage3 = self.stage3_index_offsets(embedding_stage3_after_PB)
        index_offset_stage2 = self.stage2_index_offsets(embedding_stage2_after_PB)
        index_offset_stage1 = self.stage1_index_offsets(embedding_stage1_after_PB)
        
        # -----regression-----
        
        stage1_regress = prob_stage1[:, 0] * 0

        for index in range(self.stage_num[0]):
            stage1_regress += (index + self.lambda_index * index_offset_stage1[:, index]) * prob_stage1[:, index]
        stage1_regress = torch.unsqueeze(stage1_regress, 1)
        stage1_regress /= (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k))


        stage2_regress = prob_stage2[:, 0] * 0
                
        for index in range(self.stage_num[1]):
            stage2_regress += (index + self.lambda_index * index_offset_stage2[:, index]) * prob_stage2[:, index]
        stage2_regress = torch.unsqueeze(stage2_regress, 1)
        stage2_regress /= (
            (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k)) *
            (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k))
            )

        
        stage3_regress = prob_stage3[:, 0] * 0
        
        for index in range(self.stage_num[2]):
            stage3_regress = stage3_regress + (index + self.lambda_index * index_offset_stage3[:, index]) * prob_stage3[:, index]
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        stage3_regress /= (
            (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k)) * 
            (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)) * 
            (self.stage_num[2] * (1 + self.lambda_delta * stage3_delta_k))
            )
        
        regress_class = (stage1_regress + stage2_regress + stage3_regress) * self.class_range
        regress_class = torch.squeeze(regress_class, 1)
        return regress_class


def demo_test():
    device = 'cpu'
    net = SSRNet()
    net = net.to(device)
    net.eval()
    x = torch.randn(1, 3, 64, 64).to(device)
    test_numbers_ = 1000
    a_time = time.time()
    for i in range(test_numbers_):
        y = net(x)
    cost_time = time.time() - a_time
    print("time costs:{} s, average_time:{} s\n".format(cost_time, cost_time / test_numbers_))


if __name__ == "__main__":
    demo_test()
