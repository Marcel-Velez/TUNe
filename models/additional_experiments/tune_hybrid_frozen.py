import torch
import torch.nn as nn

from .. import ConvBlock, SampleCNN, Identity

from clmr_utils import load_encoder_checkpoint
ENCODER_CHECKPOINT = "./clmr/models/model_weight/clmr_checkpoint_10000.pt"
N_CLASSES = 50
REPRESENTATION_SIZE = 512

class TuneHybridFrozen(nn.Module):
    def __init__(self, in_channels=10, n_classes=1):
        super(TuneHybridFrozen, self).__init__()

        padding = 0
        stride = 3
        self.pool = nn.MaxPool1d(3, stride=3)
        
        self.conv1 = ConvBlock(1            , int(in_channels),  kernel=3, stride=1, padding=1)
        self.conv2 = ConvBlock(int(in_channels), in_channels*2,  kernel=3, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels*2,    in_channels*4,  kernel=3, stride=1, padding=1)
        self.conv4 = ConvBlock(in_channels*4,    in_channels*8,  kernel=3, stride=1, padding=1)
        self.conv5 = ConvBlock(in_channels*8,    in_channels*16, kernel=3, stride=1, padding=1)
                
        self.transposedConv6 = nn.ConvTranspose1d(in_channels*16, in_channels*8, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv7 = nn.ConvTranspose1d(in_channels*8,  in_channels*4, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv8 = nn.ConvTranspose1d(in_channels*4,  in_channels*2, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv9 = nn.ConvTranspose1d(in_channels*2,  in_channels*1, kernel_size=3, stride=stride, padding=padding)

        self.conv6 = ConvBlock(in_channels*16,     in_channels*8, kernel=3, stride=1, padding=1)
        self.conv7 = ConvBlock(in_channels*8,      in_channels*4, kernel=3, stride=1, padding=1)
        self.conv8 = ConvBlock(in_channels*4,      in_channels*2, kernel=3, stride=1, padding=1)
        self.conv9 = ConvBlock(int(in_channels*2), 1, kernel=3, stride=1, padding=1)


        # go down again

        self.model_down = SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=0,
            out_dim=50,
        )

        state_dict = load_encoder_checkpoint(ENCODER_CHECKPOINT, N_CLASSES) # fill in path to directory

        state_dict['fc.weight'] = torch.randn((N_CLASSES, REPRESENTATION_SIZE))

        self.model_down.load_state_dict(state_dict)
        self.model_down.fc = Identity()
        self.model_down.requires_grad_(False)

        self.fc = nn.Linear(512, n_classes)
        torch.nn.init.xavier_uniform(self.fc.weight)


    def forward(self, x):

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        # expansive
        u6 = self.transposedConv6(c5)
        u6 = torch.cat((u6, c4), axis=1) # sum to 10
        c6 = self.conv6(u6)


        u7 = self.transposedConv7(c6)
        u7 = torch.cat((u7, c3), axis=1) # sum to 10
        c7 = self.conv7(u7)

        u8 = self.transposedConv8(c7)
        u8 = torch.cat((u8, c2), axis=1) # sum to 10
        c8 = self.conv8(u8)

        u9 = self.transposedConv9(c8)
        u9 = torch.cat((u9, c1), axis=1) # sum to 10
        c9 = self.conv9(u9)

        output = self.model_down(c9)
            
        output = output.unsqueeze(2)
        # and way down we go 
        
        output = self.fc(output.permute(0,2,1))

        return output.view(output.shape[0],-1)





