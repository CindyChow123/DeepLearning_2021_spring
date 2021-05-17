import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self,latent_dim,out_chan=1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.out_chann = out_chan
        self.fm = 64
        # latent_dim *1 *1
        self.add_module('conv1',nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim,self.fm*8,4,1,0,bias=False),
            nn.BatchNorm2d(self.fm*8),
            nn.ReLU(True)
        ))
        # 512 *4 *4
        self.add_module('conv2', nn.Sequential(
            nn.ConvTranspose2d(self.fm*8, self.fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm * 4),
            nn.ReLU(True)
        ))
        # 256 *8 *8
        self.add_module('conv3', nn.Sequential(
            nn.ConvTranspose2d(self.fm*4, self.fm*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm*2),
            nn.ReLU(True)
        ))
        # 128 *16 *16
        self.add_module('conv4', nn.Sequential(
            nn.ConvTranspose2d(self.fm*2, self.fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm),
            nn.ReLU(True)
        ))
        # 64 *28 28
        self.add_module('conv5', nn.Sequential(
            nn.ConvTranspose2d(self.fm, self.out_chann, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        ))
        # 3 *

    def forward(self, z):
        # Generate images from z
        z = self.__getattr__('conv1')(z)
        z = self.__getattr__('conv2')(z)
        z = self.__getattr__('conv3')(z)
        z = self.__getattr__('conv4')(z)
        z = self.__getattr__('conv5')(z)
        return z


class Discriminator(nn.Module):
    def __init__(self,in_chan):
        super(Discriminator, self).__init__()
        self.in_chan = in_chan
        self.fm = 64
        # 1*28*28
        self.add_module('conv1',nn.Sequential(
            nn.Conv2d(self.in_chan,self.fm,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True)
        ))
        # 1*14*14
        self.add_module('conv2',nn.Sequential(
            nn.Conv2d(self.fm,self.fm*2,4,2,1,bias=False),
            nn.BatchNorm2d(self.fm*2),
            nn.LeakyReLU(0.2,inplace=True)
        ))
        self.add_module('conv3', nn.Sequential(
            nn.Conv2d(self.fm*2, self.fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        self.add_module('conv4', nn.Sequential(
            nn.Conv2d(self.fm*4, self.fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm * 8),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        # 1*7*7
        self.add_module('conv5', nn.Sequential(
            nn.Conv2d(self.fm*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ))

    def forward(self, img):
        # return discriminator score for img
        img = self.__getattr__('conv1')(img)
        img = self.__getattr__('conv2')(img)
        img = self.__getattr__('conv3')(img)
        img = self.__getattr__('conv4')(img)
        score = self.__getattr__('conv5')(img)
        return score



def train(dataloader, discriminator, generator, optimizer_G, optimizer_D,criterion,device,args):
    # with tqdm(total=60000 * args.n_epochs) as pbar:
    G_losses = []
    D_losses = []
    for epoch in range(args.n_epochs):

        for i, (imgs,_) in enumerate(dataloader):
            # ground truth image
            gt = imgs.to(device)

            # Train Generator: minimize log(1-D(G(z)) = maximize log(D(G(z))
            # ---------------
            generator.zero_grad()
            # to be more real to fool discriminator
            label = torch.ones(imgs.shape[0],device=device)
            noise = torch.randn(imgs.shape[0], args.latent_dim, 1, 1, device=device)
            fake = generator(noise)
            score = discriminator(fake).view(-1)
            # loss
            lossG = criterion(score,label)
            lossG.backward()
            # a = score.mean().item()
            optimizer_G.step()


            # Train Discriminator: maximize log(D(x)) + log(1-D(G(z)))
            # -------------------
            optimizer_D.zero_grad()
            # log(D(x))
            score_gt = discriminator(gt).view(-1)
            # loss
            lossD_gt =  criterion(score_gt,label)
            lossD_gt.backward() # accumulate gradients for part 1
            # log(1-D(G(z))), cut gradient update of fake for this branch
            label = torch.zeros(imgs.shape[0],device=device)
            score_detach = discriminator(fake.detach()).view(-1)
            lossD_fake = criterion(score_detach,label)
            lossD_fake.backward()
            lossD = lossD_gt + lossD_fake

            optimizer_D.step()

            # Print stats
            # -----------
            if i % 1000 == 0:
                print('epoch: %d ,batch: %d\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch+args.start_epoch,i,lossD,lossG))
                G_losses.append(lossG.item())
                D_losses.append(lossD.item())
            # Save Images
            # -----------
            batches_done = (epoch+args.start_epoch) * len(dataloader) + i
            if i == 3000:
                # You can use the functsion save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(fake[:4],
                           'images/{}.png'.format(batches_done),nrow=2,
                            normalize=True)
                torch.save(generator.state_dict(), "face_generator2.pt")
                torch.save(discriminator.state_dict(), "face_dis2.pt")
                np.save('G_loss.npy', np.array(G_losses))
                np.save('D_loss.npy', np.array(D_losses))
                # pbar.update(imgs.shape[0])
    return G_losses,D_losses


def main(args):
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    # trans = transforms.Compose([
    #     transforms.Resize([64,64]),
    #     transforms.CenterCrop([64,64]),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    dataloader = torch.utils.data.DataLoader(datasets.ImageFolder(root=args.img_dir,
                           transform=transforms.Compose([
                               transforms.Resize([64,64]),
                               transforms.CenterCrop([64,64]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])),batch_size=args.batch_size,shuffle=True)

    # load data
    # dataloader = torch.utils.data.DataLoader(
    #     CELEBA(annotation_file=args.annotation_file,img_dir=args.img_dir,transform=trans),
    #     batch_size=args.batch_size, shuffle=True)

    # device
    device = torch.device("cuda" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim,out_chan=3)
    discriminator = Discriminator(in_chan=3)
    netD = discriminator.to(device)
    netG = generator.to(device)
    if (device.type == 'cuda'):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    if args.start_epoch == 0:
        netD.apply(weights_init)
        netG.apply(weights_init)
    else:
        netD.load_state_dict(torch.load('face_dis2.pt'))
        netG.load_state_dict(torch.load('face_generator2.pt'))
        print('Load successfully!')

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr,betas=(0.5,0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr,betas=(0.5,0.999))

    criterion = nn.BCELoss()

    # Start training
    G_losses,D_losses = train(dataloader,netD, netG, optimizer_G, optimizer_D,criterion,device,args)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "face_generator2.pt")
    torch.save(discriminator.state_dict(), "face_dis2.pt")
    np.save('G_loss.npy',np.array(G_losses))
    np.save('D_loss.npy',np.array(D_losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str,default='./data/identity_CelebA.txt')
    parser.add_argument('--img_dir', type=str,default='./data')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--ngpu', type=int, default=1,help='the number of cuda to use, if 0, then cpu')
    parser.add_argument('--start_epoch',type=int,default=5,help="resume last time's training")
    args = parser.parse_args()

    main(args)
