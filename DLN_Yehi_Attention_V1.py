import torch
import torch.nn as nn
# from torch.nn.parameter import Parameter

##---------- My new algorithm by adding attentino mechanizm ----------##
# DLN (Lighting Network for Low-Light Image Enhancement)
### paper and code at https://github.com/WangLiwen1994/DLN/tree/master
##--------- End ----------##


######---------- Start ShuffleAttention ----------####
class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

##---------- End ShuffleAttention ----------


##---------- Selective Kernel Feature Fusion (SKFF) ----------
from collections import OrderedDict
class SKAttention(nn.Module):

    def __init__(self, channel,kernels=[3],reduction=16,group=1,L=32):   #kernels=[1,3,5,7]
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V

##---------- End Selective Kernel Feature Fusion (SKFF) ----------


class LightenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(LightenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code+offset
        out = self.conv_Decoder(code_lighten)
        return out

class DarkenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code-offset
        out = self.conv_Decoder(code_lighten)
        return out

class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y


class LBP(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size,output_size)
        self.conv1_1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x=self.fusion(x)
        hr = self.conv1_1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1_1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2_1(hr)
        return hr_weight + h_residue


class DLNV1(nn.Module):
    def __init__(self, input_dim=3, dim=64, num_classes=4):
        super(DLNV1, self).__init__()
        inNet_dim = input_dim + 1
        # 1:brightness
        self.feat1 = ConvBlock(inNet_dim, 2 * dim, 3, 1, 1)
        self.feat2 = ConvBlock(2 * dim, dim, 3, 1, 1)

        self.feat_out_1 = LBP(input_size=dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.feat_out_2 = LBP(input_size=2 * dim, output_size=dim, kernel_size=3, stride=1,
                              padding=1)
        self.feat_out_3 = LBP(input_size=3 * dim, output_size=dim, kernel_size=3, stride=1,
                              padding=1)

        self.feature = ConvBlock(input_size=4 * dim, output_size=dim, kernel_size=3, stride=1, padding=1)

        # Shuffle Attention
        self.intermediate_Attention = ShuffleAttention(channel=dim)

        # Last stage SK Attention
        self.final_toAttention = SKAttention(channel=dim)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


        self.out = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.flat = nn.Flatten()

        self.out_Final_1 = None
        self.out_Final_1_Relu  = nn.ReLU()

        self.out_Final = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, x_ori, tar=None):
        # data gate
        x = (x_ori - 0.5) * 2
        x_bright, _ = torch.max(x_ori, dim=1, keepdim=True)

        x_in = torch.cat((x, x_bright), 1)

        # feature extraction
        feature = self.feat1(x_in)
        feature_1_in = self.feat2(feature)
        feature_1_out = self.feat_out_1(feature_1_in)

        feature_2_in = torch.cat([feature_1_in, feature_1_out], dim=1)
        feature_2_out = self.feat_out_2(feature_2_in)

        feature_3_in = torch.cat([feature_1_in, feature_1_out, feature_2_out], dim=1)
        feature_3_out = self.feat_out_3(feature_3_in)

        # Shuffle Attention
        feature_1_in = self.intermediate_Attention(feature_1_in)
        feature_1_out = self.intermediate_Attention(feature_1_out)
        feature_2_out = self.intermediate_Attention(feature_2_out)
        feature_3_out = self.intermediate_Attention(feature_3_out)

        feature_in = torch.cat([feature_1_in, feature_1_out, feature_2_out, feature_3_out], dim=1)
        feature_out = self.feature(feature_in)
        # pred = self.out(feature_out) + x_ori

        # Multi-scale selective feature fusion
        attention_result = self.final_toAttention(feature_out)


        ##### Reconstruct #####
        # pred = self.out(attention_result) + x_ori
        pred = self.out(attention_result)


        ### Classifiaction Part
        _, c, w, h = pred.shape
        # print(c,w,h)
        # input_images = input_images.view(-1, c * w * h)
        pred = self.flat(pred)

        # input_images = self.hidden_layer(input_images)
        if self.out_Final_1 is None:
            self.hidden_layer = nn.Linear(c*w*h, 128)
        pred = self.hidden_layer(pred)
        pred = self.out_Final_1_Relu(pred)

        # pred = self.out_Final_1_Relu(self.hidden_layer(pred))
        pred = self.out_Final(pred)

        return pred


############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out
    

#
#
# if __name__ == "__main__":
#     model = DLNV1().cuda()
#     print(model)
#     from torchsummary import summary
#     # summary(model, (3, 64, 64))
#
#     print('Finish First Check')

# #################
#     from thop import profile
#
#     # input = torch.ones(1, 3, 400, 592, dtype=torch.float, requires_grad=False).cuda()
    input = torch.ones(1, 3, 100, 100, dtype=torch.float, requires_grad=False)
#
    model = DLNV1()
    out = model(input)
#     flops, params = profile(model, inputs=(input,))
#
    print('input shape:', input.shape)
#     print('parameters:', params / 1e6)
#     print('flops', flops / 1e9)
    print('output shape', out.shape)
    print('output', out)

    

