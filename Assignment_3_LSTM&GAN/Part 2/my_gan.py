import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


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
            nn.ConvTranspose2d(self.latent_dim,self.fm*4,4,1,0,bias=False),
            nn.BatchNorm2d(self.fm*4),
            nn.ReLU(True)
        ))
        # 256 *4 *4
        self.add_module('conv2', nn.Sequential(
            nn.ConvTranspose2d(self.fm*4, self.fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm * 2),
            nn.ReLU(True)
        ))
        # 128 *8 *8
        self.add_module('conv3', nn.Sequential(
            nn.ConvTranspose2d(self.fm*2, self.fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm),
            nn.ReLU(True)
        ))
        # 64 *16 *16
        self.add_module('conv4', nn.Sequential(
            nn.ConvTranspose2d(self.fm, self.out_chann, kernel_size=2, stride=2, padding=2, bias=False),
            nn.Tanh()
        ))
        # 1 *28 28

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

    def forward(self, z):
        # Generate images from z
        z = self.__getattr__('conv1')(z)
        z = self.__getattr__('conv2')(z)
        z = self.__getattr__('conv3')(z)
        z = self.__getattr__('conv4')(z)
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
        # 1*7*7
        self.add_module('conv3', nn.Sequential(
            nn.Conv2d(self.fm*2, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        ))

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

    def forward(self, img):
        # return discriminator score for img
        img = self.__getattr__('conv1')(img)
        img = self.__getattr__('conv2')(img)
        score = self.__getattr__('conv3')(img)
        return score



def train(dataloader, discriminator, generator, optimizer_G, optimizer_D,criterion,device,args):
    # with tqdm(total=60000 * args.n_epochs) as pbar:
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
            if i % 100 == 0:
                print('epoch: %d ,batch: %d\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch,i,lossD,lossG))
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(fake[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

                # pbar.update(imgs.shape[0])


def main(args):
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim)
    discriminator = Discriminator(in_chan=1)
    netD = discriminator.to(device)
    netG = generator.to(device)
    if (device.type == 'cuda'):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    netD.apply(weights_init)
    netG.apply(weights_init)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr,betas=(0.5,0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr,betas=(0.5,0.999))

    criterion = nn.BCELoss()

    # Start training
    train(dataloader,netD, netG, optimizer_G, optimizer_D,criterion,device,args)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--ngpu', type=int, default=1,help='the number of cuda to use, if 0, then cpu')
    args = parser.parse_args()

    main(args)
