import os
import tensorflow as tf
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

#------------image load---------------
real_image_path = "./dataset/face2comics_v1.0.0_by_Sxela/comics/"
condition_image_path = "./dataset/face2comics_v1.0.0_by_Sxela/face/"
real_image_pathList = os.listdir(real_image_path)
condition_image_pathList = os.listdir(condition_image_path)
image_index = 0
def get_images_batch(batch_size):
    global image_index
    if image_index+batch_size >= 10000:
        image_index = 0
    real_image = torch.full((0,3,512,512),0)
    condition_image = torch.full((0,3,512,512),0)
    for image in real_image_pathList[image_index:image_index+batch_size]:
        img = torchvision.io.read_image(real_image_path+image)
        img = img/255.
        img = torch.unsqueeze(img,dim=0)
        real_image = torch.cat((real_image,img))
    for image in condition_image_pathList[image_index:image_index+batch_size]:
        img = torchvision.io.read_image(condition_image_path+image)
        img = img/255.
        img = torch.unsqueeze(img,dim=0)
        condition_image = torch.cat((condition_image,img))
    image_index += batch_size
    return real_image,condition_image

#-------------image save------------------
def save_images(image_rows,num_images,epoch,cur_step):
    for i in range(3):
        image_shifted = image_rows[i]
        image = image_shifted.detach()
        image = image.permute(0,2,3,1)
        for j in range(num_images):
            plt.subplot(3,num_images,i*num_images+j+1)
            plt.imshow(image[j])
            plt.axis('off')
    plt.savefig(f"./generated images/Epoch{epoch+1}_Step{cur_step}.jpg")

def save_loss(step,gen_loss,disc_loss):
    step = torch.Tensor(step)
    gen_loss = torch.Tensor(gen_loss)
    disc_loss = torch.Tensor(disc_loss)
    loss = torch.stack((gen_loss,disc_loss))
    loss = torch.permute(loss,(1,0))
    plt.plot(step,loss)
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.legend(['generator','discriminator'],loc='upper left')
    plt.savefig(f"./loss_images/pix2pix_loss_chart.jpg")
    
#------------network structure-------------------
def crop(image,new_shape):
    middle_height = image.shape[2]//2
    middle_width = image.shape[3]//2
    starting_height = middle_height-round(new_shape[2]/2)
    final_height = starting_height+new_shape[2]
    starting_width = middle_width-round(new_shape[3]/2)
    final_width = starting_width+new_shape[3]
    cropped_image = image[:,:,starting_height:final_height,starting_width:final_width]
    return cropped_image

class ContractingBlock(nn.Module):
    def __init__(self,input_channels,use_dropout=False,use_bn=True):
        super(ContractingBlock,self).__init__()
        self.conv1 = nn.Conv2d(input_channels,input_channels*2,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(input_channels*2,input_channels*2,kernel_size=3,padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels*2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self,x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self,input_channels,use_dropout=False,use_bn=True):
        super(ExpandingBlock,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv1 = nn.Conv2d(input_channels,input_channels//2,kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels,input_channels//2,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(input_channels//2,input_channels//2,kernel_size=2,padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels//2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self,x,skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x,x.shape)
        x = torch.cat([x,skip_con_x],axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.deopout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(FeatureMapBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels,kernel_size=1)

    def forward(self,x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self,input_channels,output_channels,hidden_channels=32):
        super(UNet,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels*2,use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels*4,use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.contract5 = ContractingBlock(hidden_channels*8)
        self.contract5 = ContractingBlock(hidden_channels*16)
        self.contract6 = ContractingBlock(hidden_channels*32)
        self.expand0 = ExpandingBlock(hidden_channels*64)
        self.expand1 = ExpandingBlock(hidden_channels*32)
        self.expand2 = ExpandingBlock(hidden_channels*16)
        self.expand3 = ExpandingBlock(hidden_channels*8)
        self.expand4 = ExpandingBlock(hidden_channels*4)
        self.expand5 = ExpandingBlock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels,output_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6,x5)
        x8 = self.expand1(x7,x4)
        x9 = self.expand2(x8,x3)
        x10 = self.expand3(x9,x2)
        x11 = self.expand4(x10,x1)
        x12 = self.expand5(x11,x0)
        xn = self.downfeature(x12)
        return self.sigmoid(xn)

class Discriminator(nn.Module):
    def __init__(self,input_channels,hidden_channels=8):
        super(Discriminator,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16,1,kernel_size=1)

    def forward(self,x,y):
        x = torch.cat([x,y],axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn

#---------------params initialize------------
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200

n_epochs = 20
input_dim = 3
real_dim = 3
display_step = 200
batch_size = 8
lr = 0.0002
target_shape = 256
device = 'cuda'

gen = UNet(input_dim,real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(),lr=lr)
disc = Discriminator(input_dim+real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(),lr=lr)

def weights_init(m):
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight,0.0,0.02)
    if isinstance(m,nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight,0.0,0.02)
        torch.nn.init.constant_(m.bias,0)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

#----------loss--------------
def get_gen_loss(gen,disc,real,condition,adv_criterion,recon_criterion,lambda_recon):
    fake_images = gen(condition)
    fake_preds = disc(fake_images,real)
    adv_loss = adv_criterion(fake_preds,torch.ones_like(fake_preds))
    rec_loss = recon_criterion(fake_images,real)
    gen_loss = adv_loss+(rec_loss*lambda_recon)
    return gen_loss

#----------------train-------------
def train():
    generator_loss_record = []
    discriminator_loss_record = []
    cur_step_record = []
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    batch_per_epoch = 9960//batch_size
    cur_step = 0
    for epoch in range(n_epochs):
        print("Epoch: ",epoch)
        for i in range(batch_per_epoch):
            condition,real = get_images_batch(batch_size=batch_size)
            condition = condition.to(device)
            real = real.to(device)
            disc_opt.zero_grad()
            with torch.no_grad():
                fake = gen(condition)
            disc_fake_hat = disc(fake.detach(),condition)
            disc_fake_loss = adv_criterion(disc_fake_hat,torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real,condition)
            disc_real_loss = adv_criterion(disc_real_hat,torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss+disc_real_loss)/2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen,disc,real,condition,adv_criterion,recon_criterion,lambda_recon)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item()/display_step
            mean_generator_loss += gen_loss.item()/display_step
            discriminator_loss_record.append(disc_loss.item())
            generator_loss_record.append(gen_loss.item())
            cur_step_record.append(cur_step)
            if (cur_step+1)%display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                save_images([condition,real,fake],batch_size,epoch)
            cur_step += 1

        if (epoch+1)%2 == 0:
            torch.save({'gen':gen.state_dict(),
                        'gen_opt':gen_opt.state_dict(),
                        'disc':disc.state_dict(),
                        'disc_opt':disc_opt.state_dict()
                        },f"./model/pix2pix_{epoch}.pth")
            
    save_loss(cur_step_record,generator_loss_record,discriminator_loss_record)
train()
            
















        
