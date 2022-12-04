import argparse
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

class DigifaceContrastiveDataset(Dataset):
    def __init__(self, image_paths, transform, n_views=2):
        self.image_paths = image_paths
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        return [self.transform(image) for i in range(self.n_views)]

class SimCLRModel():

    def __init__(self, convnet, hidden_dim, temperature):
        self.convnet = convnet
        self.convnet.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 4*hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.temperature = temperature

    def info_nce_loss(self, batch, mode='train'):
        imgs = batch
        imgs = torch.cat(imgs, dim=0)

        feats = self.convnet(imgs)
        cos_sim = torch.nn.functional.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        if mode=='train':
            return nll
        else:
            comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                                  cos_sim.masked_fill(pos_mask, -9e15)],
                                 dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            # return nll, (sim_argsort < 5).float().mean()
            return nll, (sim_argsort == 0).float().mean()

def pretrain_digiface(args):

    fldrlist = []
    for fldrname in os.listdir('digiface/'):
        fldrlist.append(os.path.join('digiface/', fldrname))

    fldrlist, fldrlist_valid = train_test_split(fldrlist, test_size=0.2, random_state=0)

    imglist = []
    for fldrname in fldrlist:
        for filename in os.listdir(fldrname):
            imglist.append(os.path.join(fldrname, filename))
    imglist_valid = []
    for fldrname in fldrlist_valid:
        for filename in os.listdir(fldrname):
            imglist_valid.append(os.path.join(fldrname, filename))

    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(size=224),
                                              transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=9),
                                              transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = DigifaceContrastiveDataset(imglist, contrast_transforms, 2)
    valid_dataset = DigifaceContrastiveDataset(imglist_valid, contrast_transforms, 2)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    print("Total Number of Train Batches : ", len(train_loader))
    model = SimCLRModel(torch.nn.DataParallel(models.resnet50()), 128, args.temperature)
    model.convnet.cuda()
    # model.convnet.to(device)
    optimizer = torch.optim.AdamW(model.convnet.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/50)

    best_acc = -1
    for epoch in range(args.epochs):
        print("Epoch ", epoch)

        ttl_loss = 0.
        model.convnet.train()
        for i, data in tqdm(enumerate(train_loader, 0)):
            # if i>50:
            #     break
            optimizer.zero_grad()

            loss = model.info_nce_loss([ele.cuda() for ele in data])
            # loss = model.info_nce_loss([ele.to(device) for ele in data])
            ttl_loss += loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

        print("Loss ", (ttl_loss/i).cpu().detach().numpy())

        model.convnet.eval()
        ttl_acc = 0.
        for i, data in enumerate(valid_loader, 0):
            # if i>50:
            #     break
            loss, acc = model.info_nce_loss([ele.cuda() for ele in data], mode='val')
            ttl_acc += acc

        ttl_acc = (ttl_acc/i).cpu().detach().numpy()
        print("Validation Acc (Top 5) ", ttl_acc*100)
        if ttl_acc > best_acc:
            best_acc = ttl_acc
            torch.save(model.convnet, args.savefldr + 'digiface_pretrained.pth')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
parser.add_argument("--epochs", type=int, default=5, help="Number of Training Epochs")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate for Training")
parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for CE Loss")
parser.add_argument("--gpus", default=None, help="GPU Device ID to use")
parser.add_argument("--savefldr", default="models/", help="Folder to save checkpoints")

args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
pretrain_digiface(args)
