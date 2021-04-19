import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet_dialated as resnet
from models import ASPP
from models.GRN import GRN
from models.mRN import mRN
# from models.mRN1 import mRN
import math
# The Code of baseline network is referenced from https://github.com/icoz69/CaNet
# The code of training & testing is referenced from https://github.com/xiaomengyc/SG-One

class OneModel(nn.Module):
    def __init__(self, args):
        self.inplanes = 64
        self.num_pro = 3
        super(OneModel, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv1_dsn6 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2_dsn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_dsn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_dsn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv44_dsn6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_dsn6 = nn.Conv2d(256, 2, kernel_size=1)
        
        # self.conv1_dsn6 = nn.Conv2d(128, 64, kernel_size=1)
        # self.conv2_dsn6 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.conv3_dsn6 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.conv4_dsn6 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.conv5_dsn6 = nn.Conv2d(64, 1, kernel_size=1)      
        self.conv5_dsn6_up = nn.ConvTranspose2d(
            2, 2, kernel_size=64, stride=32)

        self.conv5_dsn6_5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.conv1_dsn5 = nn.Conv2d(512, 64, kernel_size=1)
        # self.conv1_dsn5 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2_dsn5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn5 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.sum_dsn5_up = nn.ConvTranspose2d(2, 2, kernel_size=32, stride=16)

        self.sum_dsn5_4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.sum_dsn5_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=4)
        self.conv1_dsn4 = nn.Conv2d(512, 64, kernel_size=1)
        # self.conv1_dsn4 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2_dsn4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn4 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.sum_dsn4_up = nn.ConvTranspose2d(2, 2, kernel_size=16, stride=8)

        self.sum_dsn4_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.conv1_dsn3 = nn.Conv2d(256, 64, kernel_size=1)
        # self.conv1_dsn3 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2_dsn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.sum_dsn3_up = nn.ConvTranspose2d(2, 2, kernel_size=8, stride=4)

        self.sum_dsn3_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.sum_dsn3_1 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=4)
        self.conv1_dsn2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2_dsn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn2 = nn.Conv2d(64, 2, kernel_size=3, padding=1)      
        # self.conv1_dsn2 = nn.Conv2d(32, 32, kernel_size=1)
        # self.conv2_dsn2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.conv3_dsn2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.conv4_dsn2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        self.sum_dsn2_up = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)

        self.conv1_dsn1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn1 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        # self.conv1_dsn1 = nn.Conv2d(18, 18, kernel_size=1)
        # self.conv2_dsn1 = nn.Conv2d(18, 18, kernel_size=3, padding=1)
        # self.conv3_dsn1 = nn.Conv2d(18, 18, kernel_size=3, padding=1)
        # self.conv4_dsn1 = nn.Conv2d(18, 1, kernel_size=3, padding=1)
        self.batchnorm1=nn.BatchNorm2d(256)
        self.batchnorm2=nn.BatchNorm2d(64)
        self.batch_size = args.batch_size
        # self.layer6 = ASPP.PSPnet()
        self.model_res = resnet.Res50_Deeplab(pretrained=True,stop_layer='layer4')
    def forward(self, query_rgb, support_rgb, support_mask):
        # resout_query = self.model_res(query_rgb)
        # resout_support = self.model_res(support_rgb)
        # print(resout_query[2].shape )
        x1_size = query_rgb.size()
        x1 = F.relu(self.conv1_1(query_rgb))
        conv1_2_query  = F.relu(self.conv1_2(x1))
        # print(conv1_2_query)
        x1 = F.max_pool2d(conv1_2_query, kernel_size=2, stride=2)
       
        x1 = F.relu(self.conv2_1(x1))
        conv2_2_query = F.relu(self.conv2_2(x1))
        x1 = F.max_pool2d(conv2_2_query, kernel_size=2, stride=2)
        x1 = F.relu(self.conv3_1(x1))
        x1 = F.relu(self.conv3_2(x1))
        conv3_3_query = F.relu(self.conv3_3(x1))
        x1 = F.max_pool2d(conv3_3_query, kernel_size=2, stride=2)
        x1 = F.relu(self.conv4_1(x1))
        x1 = F.relu(self.conv4_2(x1))
        conv4_3_query = F.relu(self.conv4_3(x1))
        x1 = F.max_pool2d(conv4_3_query, kernel_size=3, stride=2, padding=1)
        x1 = F.relu(self.conv5_1(x1))
        x1 = F.relu(self.conv5_2(x1))
        conv5_3_query = F.relu(self.conv5_3(x1))
        x1 = F.max_pool2d(conv5_3_query, kernel_size=3, stride=2, padding=1)


        x2_size = support_rgb.size()
        x2 = F.relu(self.conv1_1(support_rgb))
        conv1_2_support  = F.relu(self.conv1_2(x2))
        x2 = F.max_pool2d(conv1_2_support, kernel_size=2, stride=2)
        x2 = F.relu(self.conv2_1(x2))
        conv2_2_support = F.relu(self.conv2_2(x2))
        x2 = F.max_pool2d(conv2_2_support, kernel_size=2, stride=2)
        x2 = F.relu(self.conv3_1(x2))
        x2 = F.relu(self.conv3_2(x2))
        conv3_3_support = F.relu(self.conv3_3(x2))
        x2 = F.max_pool2d(conv3_3_support, kernel_size=2, stride=2)
        x2 = F.relu(self.conv4_1(x2))
        x2 = F.relu(self.conv4_2(x2))
        conv4_3_support = F.relu(self.conv4_3(x2))
        x2 = F.max_pool2d(conv4_3_support, kernel_size=3, stride=2, padding=1)
        x2 = F.relu(self.conv5_1(x2))
        x2 = F.relu(self.conv5_2(x2))
        conv5_3_support = F.relu(self.conv5_3(x2))
        x2 = F.max_pool2d(conv5_3_support, kernel_size=3, stride=2, padding=1)
        
        _,C6,_,_ = x1.shape
        mRN6 = mRN(C6).cuda()
        # m6 = nn.MaxPool2d(32,stride=32)
        # mask6 = m6(support_mask)
        
        relation6 = mRN6(x2,support_mask,x1)
        
        x = self.conv1_dsn6(relation6)
        xm = F.relu(self.conv2_dsn6(x))
        # x = self.batchnorm1(x)
        x = x+F.relu(self.conv3_dsn6(xm))
        xm = F.relu(self.conv4_dsn6(x))
        x = x+F.relu(self.conv44_dsn6(xm))
        conv5_dsn6 = self.conv5_dsn6(x)
        
        # conv5_dsn61  = F.softmax(conv5_dsn6 , dim=1)
        # values, conv5_dsn6_1 = torch.max(conv5_dsn61, dim=1,keepdim=True)

        upscore_dsn6 = self.crop(self.conv5_dsn6_up(conv5_dsn6), x1_size)

        x = self.conv5_dsn6_5(conv5_dsn6)
        crop1_dsn5 = self.crop(x, conv5_3_support.size())
        
        b,c,h,w = crop1_dsn5.shape
        # crop1_dsn5_1  = crop1_dsn5[:,1,:,:].view(b,1,h,w)
        crop1_dsn5_2  = crop1_dsn5[:,0,:,:].view(b,1,h,w)
        x = torch.sigmoid(crop1_dsn5_2)
        # x = -1*(torch.sigmoid(crop1_dsn5))+1
        x = x.expand_as(conv5_3_query).mul(conv5_3_query)
        
        _,C5,_,_ = x.shape
        mRN5 = mRN(C5).cuda()
        relation5 = mRN5(conv5_3_support,support_mask,conv5_3_query)
        
        x = self.conv1_dsn5(relation5)
        xm = F.relu(self.conv2_dsn5(x))
        # x = self.batchnorm2(x)
        x = x+F.relu(self.conv3_dsn5(xm))
        conv4_dsn5 = self.conv4_dsn5(x)
        # x = conv4_dsn5 + crop1_dsn5
        x = conv4_dsn5 
        # xf = F.softmax(x , dim=1)
        # values, x_1 = torch.max(xf, dim=1,keepdim=True)
        upscore_dsn5 = x
        # upscore_dsn5 = self.crop(self.sum_dsn5_up(x), x1_size)

        # x = self.sum_dsn5_4(x)
        # crop1_dsn4 = self.crop(x, conv4_3_support.size())
        # x = -1*(torch.sigmoid(crop1_dsn4))+1
        # x = x.expand_as(conv4_3_query).mul(conv4_3_query)
        # _,C4,_,_ = x.shape
        # GRN4 = GRN(C4).cuda()
        # m4 = nn.MaxPool2d(8,stride=8)
        # mask4 = m4(support_mask)
        # Graph4 = GRN4(conv4_3_support,mask4,conv4_3_query)
        
        # x = self.conv1_dsn4(Graph4)
        # x = F.relu(self.conv2_dsn4(x))
        # x = F.relu(self.conv3_dsn4(x))
        # conv4_dsn4 = self.conv4_dsn4(x)
        # x = conv4_dsn4 + crop1_dsn4
        # upscore_dsn4 = self.crop(self.sum_dsn4_up(x), x1_size)

        x = self.sum_dsn5_3(x)
        # x = self.sum_dsn6_3(conv5_dsn6)
        crop1_dsn3 = self.crop(x, conv3_3_query.size())
        
        b,c,h,w = crop1_dsn3.shape
        # # crop1_dsn3_1 = crop1_dsn3[:,1,:,:].view(b,1,h,w)
        crop1_dsn3_2 = crop1_dsn3[:,0,:,:].view(b,1,h,w)
        # x = -1*(torch.sigmoid(crop1_dsn3))+1
        x = torch.sigmoid(crop1_dsn3_2)
        x = x.expand_as(conv3_3_query).mul(conv3_3_query)
        _,C3,_,_ = x.shape
        mRN3 = mRN(C3).cuda()
        # m3 = nn.MaxPool2d(4,stride=4)
        # mask3 = m3(support_mask)
        relation3 = mRN3(conv3_3_support,support_mask,conv3_3_query)
        
        x = self.conv1_dsn3(relation3)
        xm = F.relu(self.conv2_dsn3(x))
        # x = self.batchnorm2(x)
        x = x+F.relu(self.conv3_dsn3(xm))
        conv4_dsn3 = self.conv4_dsn3(x)
        x = conv4_dsn3 + crop1_dsn3
        
        # x1  = F.softmax(x , dim=1)
        # values, x_1 = torch.max(x1, dim=1,keepdim=True)
        
        upscore_dsn3 = self.crop(self.sum_dsn3_up(x), x1_size)

        # x = self.sum_dsn3_2(x)
        # crop1_dsn2 = self.crop(x, conv2_2_query.size())
        # x = -1*(torch.sigmoid(crop1_dsn2))+1
        # x = x.expand_as(conv2_2_support).mul(conv2_2_query)
        # _,C2,_,_ = x.shape
        # GRN2 = GRN(C2).cuda()
        # m2 = nn.MaxPool2d(2,stride=2)
        # mask2 = m2(support_mask)
        # Graph2 = GRN2(conv2_2_support,mask2,conv2_2_query)
        
        # x = self.conv1_dsn2(Graph2)
        # x = F.relu(self.conv2_dsn2(x))
        # x = F.relu(self.conv3_dsn2(x))
        # conv4_dsn2 = self.conv4_dsn2(x)
        # x = conv4_dsn2 + crop1_dsn2
        # upscore_dsn2 = self.crop(self.sum_dsn2_up(x), x1_size)

        upscore_dsn2 = upscore_dsn3
        upscore_dsn2 = self.crop(upscore_dsn2 , conv1_2_query.size())
        
        b,c,h,w = upscore_dsn2.shape
        # # upscore_dsn2_1 = upscore_dsn2[:,1,:,:].view(b,1,h,w)
        upscore_dsn2_2 = upscore_dsn2[:,0,:,:].view(b,1,h,w)
        # x = -1*(torch.sigmoid(upscore_dsn2))+1
        x = torch.sigmoid(upscore_dsn2_2)
        x = x.expand_as(conv1_2_query).mul(conv1_2_query)
        _,C1,_,_ = x.shape
        mRN1 = mRN(C1).cuda()
        mask1 = support_mask
        relation1 = mRN1(conv1_2_support,mask1,conv1_2_query)
        
        x = self.conv1_dsn1(relation1)
        xm = F.relu(self.conv2_dsn1(x))
        # x = self.batchnorm2(x)
        x = x+F.relu(self.conv3_dsn1(xm))
        conv4_dsn1 = self.conv4_dsn1(x)
        
        x = conv4_dsn1 + upscore_dsn2
        
        # x1  = F.softmax(x , dim=1)
        # values, x_1 = torch.max(x1, dim=1,keepdim=True)
        
        upscore_dsn1 = self.crop(x, x1_size)
        # return torch.sigmoid(upscore_dsn1), torch.sigmoid(upscore_dsn2), torch.sigmoid(upscore_dsn3), torch.sigmoid(upscore_dsn4), torch.sigmoid(upscore_dsn5), torch.sigmoid(upscore_dsn6)
        # return torch.sigmoid(upscore_dsn1), torch.sigmoid(upscore_dsn3),  torch.sigmoid(upscore_dsn5), torch.sigmoid(upscore_dsn6)
        return upscore_dsn1, upscore_dsn3,  upscore_dsn5, upscore_dsn6

    def get_loss(self, logits, query_label, idx):
        bce_logits_func = nn.CrossEntropyLoss()
        # bce_logits_func = nn.MSELoss()
        # out1, out2, out3, out4, out5, out6 = logits
        out1, out3, out5, out6 = logits
        b, c, w, h = query_label.size()
        # out1= F.upsample(out1, size=(w, h), mode='bilinear')
        # out2= F.upsample(out2, size=(w, h), mode='bilinear')
        # out3= F.upsample(out3, size=(w, h), mode='bilinear')
        # out4= F.upsample(out4, size=(w, h), mode='bilinear')
        # out5= F.upsample(out5, size=(w, h), mode='bilinear')
        # out6= F.upsample(out6, size=(w, h), mode='bilinear')
        
        out1= F.upsample(out1, size=(w, h), mode='bilinear')
        out3= F.upsample(out3, size=(w, h), mode='bilinear')
        out5= F.upsample(out5, size=(w, h), mode='bilinear')
        out6= F.upsample(out6, size=(w, h), mode='bilinear')
        # add
        query_label = query_label.view(b, -1)
        bb, cc, _, _ = out1.size()
        
        out1 = out1.view(b, cc, w * h)
        # out1 = F.softmax(out1,dim =1)
        # out1_0 = out1[:,0,:]
        # out1_1 = out1[:,1,:]
        out3 = out3.view(b, cc, w * h)
        # out3 = F.softmax(out3,dim=1)
        # out3_0 = out3[:,0,:]
        # out3_1 = out3[:,1,:]
        out5 = out5.view(b, cc, w * h)
        # out5 = F.softmax(out5,dim=1)
        # out5_0 = out5[:,0,:]
        # out5_1 = out5[:,1,:]
        out6 = out6.view(b, cc, w * h)
        # out6 = F.softmax(out6,dim=1)
        # out6_0 = out6[:,0,:]
        # out6_1 = out6[:,1,:]
        
        for i in range(b):
            sum_label = query_label.float().sum(dim=1)
            if sum_label[i]==0:
                query_label[i,:]=1
        
        # out_softmax1 = F.softmax(out1, dim=1)
        # values1, pred1 = torch.max(out_softmax1, dim=1)
        # out_softmax3 = F.softmax(out3, dim=1)
        # values3, pred3 = torch.max(out_softmax3, dim=1)
        # out_softmax5 = F.softmax(out5, dim=1)
        # values5, pred5 = torch.max(out_softmax5, dim=1)
        # out_softmax6 = F.softmax(out6, dim=1)
        # values6, pred6 = torch.max(out_softmax6, dim=1)
        
        loss1 = bce_logits_func(out1, query_label.long())
        loss3 = bce_logits_func(out3, query_label.long())
        loss5 = bce_logits_func(out5, query_label.long())
        loss6 = bce_logits_func(out6, query_label.long())

        # # print(query_label.float().sum(dim=1))
        # out1 = out1.view(b, w*h)
        # # max1 = torch.max(out1)
        # # if max1<0.5:
        # #     c1 = 1-max1
        # #     out1 = out1+c1
        # out3 = out3.view(b, w*h)
        # # max3 = torch.max(out3)
        # # if max3<0.5:
        # #     c3 = 1-max3
        # #     out3 = out3+c3
        # out5 = out5.view(b, w*h)
        # # max5 = torch.max(out5)
        # # if max5<0.5:
        # #     c5 = 1-max5
        # #     out5 = out5+c5
        # out6 = out6.view(b, w*h)
        # # max6 = torch.max(out6)
        # # if max6<0.5:
        # #     c6 = 1-max6
        # #     out6 = out6+c6
        
        # loss1 = bce_logits_func(out1, query_label.float())
        # loss3 = bce_logits_func(out3, query_label.float())
        # loss5 = bce_logits_func(out5, query_label.float())
        # loss6 = bce_logits_func(out6, query_label.float())
        
        # loss10 = bce_logits_func(out1_0, (1-query_label).float())
        # loss30 = bce_logits_func(out3_0, (1-query_label).float())
        # loss50 = bce_logits_func(out5_0, (1-query_label).float())
        # loss60 = bce_logits_func(out6_0, (1-query_label).float())

        # loss11 = bce_logits_func(out1_1, query_label.float())
        # loss31 = bce_logits_func(out3_1, query_label.float())
        # loss51 = bce_logits_func(out5_1, query_label.float())
        # loss61 = bce_logits_func(out6_1, query_label.float())
        
        loss = loss1+loss3+loss5+loss6
        # loss = loss10+loss30+loss50+loss60+loss11+loss31+loss51+loss61
        return loss5, loss1, loss3

    def get_pred(self, logits, query_image):
        # out1, out2, out3, out4, out5, out6 = logits
        out1,out3,out5, out6 = logits
        b,c,w, h = query_image.size()
        # out1 = (out1+out3+out5+out6)/4
        out1 = F.upsample(out5, size=(w, h), mode='bilinear')
        # print(out1)
        out_softmax = F.softmax(out1, dim=1)
        # max1 = torch.max(out1)
        # if max1<0.5:
        #     c1 = 1-max1
        #     out1 = out1+c1
        values, pred = torch.max(out_softmax, dim=1)
        # print(pred)
        # print(out_softmax[0,0,0,0])
        # print(out_softmax[0,1,0,0])
        # print(out1)
        # print(out1.view(b,w,h).sum(dim=[1,2]))
        # pred = torch.where(out1 > 0.5, 1, 0)
        # pred = out1.view(b,w,h).gt(0.5)
        # pred = out1.view(b,w,h)
        # print(pred)
        # print(pred.eq(0.5).sum(dim=[1,2]))
        # print(out_softmax)
        # print(pred.sum(dim=[1,2]))
        # outB, outA_pos, outB_side1, outB_side = logits
        # w, h = query_image.size()[-2:]
        # outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
        # out_softmax = F.softmax(outB_side, dim=1)
        # values, pred = torch.max(out_softmax, dim=1)
        
        return out_softmax, pred
    
    def crop(self, upsampled, x_size):
        c = (upsampled.size()[2] - x_size[2]) // 2
        _c = x_size[2] - upsampled.size()[2] + c
        assert(c >= 0)
        if(c == _c == 0):
            return upsampled
        return upsampled[:, :, c:_c, c:_c]

