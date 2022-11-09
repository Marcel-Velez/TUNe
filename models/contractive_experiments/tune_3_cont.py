import torch
import torch.nn as nn

from .. import ConvBlock


class Tune3Cont(nn.Module):
    def __init__(self, in_channels=4, n_classes=1):
        super(Tune3Cont, self).__init__()

        padding = 0
        stride = 3
        self.pool = nn.MaxPool1d(3, stride=3)
        
        self.conv1 = ConvBlock(1            , int(in_channels),  kernel=3, stride=1, padding=1)
        self.conv2 = ConvBlock(int(in_channels), in_channels*2,  kernel=3, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels*2,    in_channels*4,  kernel=3, stride=1, padding=1)
        self.conv4 = ConvBlock(in_channels*4,    in_channels*8,  kernel=3, stride=1, padding=1)
        self.conv5 = ConvBlock(in_channels*8,    in_channels*16, kernel=3, stride=1, padding=1)

        # extra contractive path layers
        self.deeperDown1 = ConvBlock(in_channels*16,    in_channels*32, kernel=3, stride=1, padding=1)
        self.deeperDown2 = ConvBlock(in_channels*32,    in_channels*64, kernel=3, stride=1, padding=1)
        self.deeperDown3 = ConvBlock(in_channels*64,    in_channels*128, kernel=3, stride=1, padding=1)
                
        self.transposedDeeper3 = nn.ConvTranspose1d(in_channels*128, in_channels*64, kernel_size=3, stride=stride, padding=padding)
        self.transposedDeeper2 = nn.ConvTranspose1d(in_channels*64, in_channels*32, kernel_size=3, stride=stride, padding=padding)
        self.transposedDeeper1 = nn.ConvTranspose1d(in_channels*32, in_channels*16, kernel_size=3, stride=stride, padding=padding)


        # regular upsample convolution layers
        self.transposedConv6 = nn.ConvTranspose1d(in_channels*16, in_channels*8, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv7 = nn.ConvTranspose1d(in_channels*8,  in_channels*4, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv8 = nn.ConvTranspose1d(in_channels*4,  in_channels*2, kernel_size=3, stride=stride, padding=padding)
        self.transposedConv9 = nn.ConvTranspose1d(in_channels*2,  in_channels*1, kernel_size=3, stride=stride, padding=padding)

        # extra convolutions after deeper contractive path
        self.deeperUp3 = ConvBlock(in_channels*128,     in_channels*64, kernel=3, stride=1, padding=1)
        self.deeperUp2 = ConvBlock(in_channels*64,     in_channels*32, kernel=3, stride=1, padding=1)
        self.deeperUp1 = ConvBlock(in_channels*32,     in_channels*16, kernel=3, stride=1, padding=1)

        # regular convolutions after transposed
        self.conv6 = ConvBlock(in_channels*16,     in_channels*8, kernel=3, stride=1, padding=1)
        self.conv7 = ConvBlock(in_channels*8,      in_channels*4, kernel=3, stride=1, padding=1)
        self.conv8 = ConvBlock(in_channels*4,      in_channels*2, kernel=3, stride=1, padding=1)
        self.conv9 = ConvBlock(int(in_channels*2), in_channels*1, kernel=3, stride=1, padding=1)


        # tail part
        self.convDown1 = ConvBlock(int(in_channels*1), in_channels*2,  kernel=3, stride=1, padding=1)
        self.convDown2 = ConvBlock(in_channels*2,      in_channels*4,  kernel=3, stride=1, padding=1)
        self.convDown3 = ConvBlock(in_channels*4,      in_channels*8,  kernel=3, stride=1, padding=1)
        self.convDown4 = ConvBlock(in_channels*8,      512, kernel=3, stride=1, padding=1)


        self.output_avg = nn.AvgPool1d(729)

        self.fc = nn.Linear(512, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)

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

        # deeper contractvie layers 
        p5 = self.pool(c5)

        cdeepdown1 = self.deeperDown1(p5)
        pdeepdown1 = self.pool(cdeepdown1)

        cdeepdown2 = self.deeperDown2(pdeepdown1)
        pdeepdown2 = self.pool(cdeepdown2)

        cdeepdown3 = self.deeperDown3(pdeepdown2)

        udeep3 = self.transposedDeeper3(cdeepdown3)
        udeep3 = torch.cat((udeep3, cdeepdown2), axis=1)
        cdeepup3 = self.deeperUp3(udeep3)

        udeep2 = self.transposedDeeper2(cdeepup3)
        udeep2 = torch.cat((udeep2, cdeepdown1), axis=1)
        cdeepup2 = self.deeperUp2(udeep2)

        udeep1 = self.transposedDeeper1(cdeepup2)
        udeep1 = torch.cat((udeep1, c5), axis=1)
        cdeepup1 = self.deeperUp1(udeep1)


        # expansive
        u6 = self.transposedConv6(cdeepup1)
        u6 = torch.cat((u6, c4), axis=1)
        c6 = self.conv6(u6)

        u7 = self.transposedConv7(c6)
        u7 = torch.cat((u7, c3), axis=1)
        c7 = self.conv7(u7)

        u8 = self.transposedConv8(c7)
        u8 = torch.cat((u8, c2), axis=1)
        c8 = self.conv8(u8)

        u9 = self.transposedConv9(c8)
        u9 = torch.cat((u9, c1), axis=1)
        c9 = self.conv9(u9)

        p9 = self.pool(c9)
        
        # tail
        c10 = self.convDown1(p9)
        p10 = self.pool(c10)
        
        c11 = self.convDown2(p10)
        p11 = self.pool(c11)

        c12 = self.convDown3(p11)
        p12 = self.pool(c12)

        c13 = self.convDown4(p12)

        output = self.output_avg(c13)
        output = self.fc(output.permute(0,2,1))

        return output.view(output.shape[0],-1)





