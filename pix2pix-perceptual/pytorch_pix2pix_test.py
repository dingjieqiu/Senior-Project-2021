import torch, network, argparse, os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import util
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='face2comics',  help='')
parser.add_argument('--test_subfolder', required=False, default='test',  help='')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')
opt = parser.parse_args()
print(opt)

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_loader = util.data_load('data/' + opt.dataset, opt.test_subfolder, transform, batch_size=1, shuffle=False)

if not os.path.isdir(opt.dataset + '_results/test_results'):
    os.mkdir(opt.dataset + '_results/test_results')

G = network.generator(opt.ngf)
G.cuda()
G.load_state_dict(torch.load(opt.dataset + '_results/' + opt.dataset + '_generator_param.pkl'))
"""
D = network.discriminator(opt.ngf)
D.cuda()
D.load_state_dict(torch.load(opt.dataset + '_results/' + opt.dataset + '_discriminator_param.pkl'))
"""

# network
n = 0
print('test start!')
PSNR_avg = 0
SSIM_avg = 0
for x_, _ in test_loader:
    if opt.inverse_order:
        y_ = x_[:, :, :, :x_.size()[2]]
        x_ = x_[:, :, :, x_.size()[2]:]
    else:
        y_ = x_[:, :, :, x_.size()[2]:]
        x_ = x_[:, :, :, :x_.size()[2]]

    if x_.size()[2] != opt.input_size:
        x_ = util.imgs_resize(x_, opt.input_size)
        y_ = util.imgs_resize(y_, opt.input_size)
    with torch.no_grad():
        test_image = G(x_.cuda())
        #real_prediction = D(x_.cuda(),y_.cuda())
        #fake_prediction = D(x_.cuda(),test_image.cuda())

    img = (test_image.cpu().numpy().squeeze(0).transpose((1,2,0))+1)/2
    img_y = (y_.cpu().numpy().squeeze(0).transpose((1,2,0))+1)/2

    #print(test_image)
    #print(y_)
    PSNR = peak_signal_noise_ratio(img, img_y)
    SSIM = structural_similarity(img, img_y, channel_axis=2)
    PSNR_avg += PSNR
    SSIM_avg += SSIM
    #time.sleep(5)
    
    """
    img = test_image
    plt.imshow(img)
    plt.show()
    img = y_
    
    plt.imshow(img)
    plt.show()
    print(torch.mean(real_prediction))
    print(torch.mean(fake_prediction))
    """
    """
    s = test_loader.dataset.imgs[n][0]
    ind = s[22:]
    path = opt.dataset + '_results/test_results/' + ind
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(path)
    """

    n += 1

print(PSNR_avg/n)
print(SSIM_avg/n)
print('%d images generation complete!' % n)
