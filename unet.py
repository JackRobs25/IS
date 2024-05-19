import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.functional import relu
from scipy import ndimage
import time
from skimage.segmentation import slic
from skimage.util import img_as_float

SP = True

in_channels = 9 if SP else 3

############################################################################################### Online SP generation stuff
def create_adj_matrix(segments_slic, kernel_size, n_sp, width, length):
    G = np.zeros((n_sp, n_sp)) # represents neighbouring relationship between superpixels
    for seg in np.unique(segments_slic):
        mask = segments_slic == seg
        
        xy = np.where(mask)
        max_x, min_x = np.max(xy[0]), np.min(xy[0])
        max_y, min_y = np.max(xy[1]), np.min(xy[1])
        min_x = max(0, min_x - kernel_size) 
        min_y = max(0, min_y - kernel_size)
        max_x = min(width, max_x + kernel_size)
        max_y = max(length, max_y + kernel_size)
        region_of_interest = mask[min_x:max_x, min_y:max_y]
        
        dilated = ndimage.binary_dilation(region_of_interest)
        diff = dilated - region_of_interest.astype(int)
        neig = np.unique(segments_slic[min_x:max_x, min_y:max_y][diff != 0])
        G[seg, neig] = 1
    return G


def generate_colored_G(G):
    # count degree of all node.
    degree =[]
    for i in range(len(G)):
        degree.append(sum(G[i]))

    # instantiate the possible color
    colorDict = {}
    for i in range(len(G)):
        colorDict[i]=[0,1,2,3,4,5]


    # sort the node depends on the degree
    sortedNode=[]
    indeks = []

    # use selection sort
    for i in range(len(degree)):
        _max = 0
        j = 0
        for j in range(len(degree)):
            if j not in indeks:
                if degree[j] > _max:
                    _max = degree[j]
                    idx = j
        indeks.append(idx)
        sortedNode.append(idx)

    # The main process
    solution={}
    for n in sortedNode: # starting from the node of highest degree
        setTheColor = colorDict[n] # setTheColor = list of available colors
        solution[n] = setTheColor[0] # assign the color for current node to be the first available color
        adjacentNode = G[n] # G[t_[n]] is the corresponding row of the adj matrix for node n
        for j in range(len(adjacentNode)): # for each node in the graph
            if adjacentNode[j]==1 and (setTheColor[0] in colorDict[j]): # if its a neighbour and it currently has the color just used by node n
                colorDict[j].remove(setTheColor[0]) # remove that color from the list so that we don't end up having some colored neighbours
    return solution


def sp_encode(img):
    width, length, _ = np.shape(img)
    segments_slic = slic(img, n_segments=300, compactness=10, sigma=1,
                        start_label=1) - 1 

    # # # Find the adjacency matrix for that superpixelation
    n_sp = len(np.unique(segments_slic))
    kernel_size = 3

    G = create_adj_matrix(segments_slic, kernel_size, n_sp, width, length)
    
    # # # Use that adjacency matrix to generate a "colored graph" - CSP problem
    solution = generate_colored_G(G)

    # Vectorised code
    segments_slic_1d = segments_slic.flatten()
    color_pix = np.array([solution[segment] for segment in segments_slic_1d])
    color_pix = color_pix.reshape(segments_slic.shape)

    out = F.one_hot(torch.tensor(color_pix)).permute(2, 0, 1)

    return out
###############################################################################################



class UNet(nn.Module):
    def __init__(self, input_channels):
        super(UNet, self).__init__()
        # Encoder
        self.e11 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # x here is a batch of images (with the super pixel channels)
        # to then use the superpixel channels and cat then to later layers we can...
        # sp = x[:,:6,width,height]
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        # cat sp to xp1 (I think, check UNet architecture)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))

        # Decoder
        
        xu1 = self.upconv1(xe32)
        diff_h = xe22.size()[2] - xu1.size()[2]
        diff_w = xe22.size()[3] - xu1.size()[3]
        xu1 = F.pad(xu1, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))
        xu11 = torch.cat([xu1, xe22], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        diff_h = xe12.size()[2] - xu2.size()[2]
        diff_w = xe12.size()[3] - xu2.size()[3]
        xu2 = F.pad(xu2, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))
        xu22 = torch.cat([xu2, xe12], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))


        # Output layer
        out = self.outconv(xd22)

        return out

import os

# print(os.listdir("/home/jroberts2/Carvana/carvana-image-masking-challenge/"))

import zipfile
import shutil

DATASET_DIR = '/home/jroberts2/Carvana/carvana-image-masking-challenge/'
WORKING_DIR = '/home/jroberts2/Carvana/working/'

# # ######################################################
# # # code for superpixelation moving
# # Move sp_train to WORKING_DIR
# # need an empty sp_val directory in working and sp_tensors full of 5088 tensors 
# shutil.move(DATASET_DIR + 'sp_train', WORKING_DIR)
# # it will still use the same train_masks 
# print("moved!")
    
# train_dir = WORKING_DIR + 'sp_train/'
# val_dir = WORKING_DIR + 'sp_val/'
# for file in sorted(os.listdir(train_dir))[3200:]:
#     shutil.move(train_dir + file, val_dir)
# print("sp_val set up!")
# print(len(os.listdir(WORKING_DIR + 'sp_train/')))
# print(len(os.listdir(WORKING_DIR + 'sp_val/')))
# ######################################################



# if len(os.listdir(WORKING_DIR)) <= 1:
#     # Move train and train_masks directories to WORKING_DIR
#     shutil.move(DATASET_DIR + 'train', WORKING_DIR)
#     shutil.move(DATASET_DIR + 'train_masks', WORKING_DIR)

#     print(
#         len(os.listdir(WORKING_DIR + 'train')),
#         len(os.listdir(WORKING_DIR + 'train_masks'))
#     )
# print(len(os.listdir(WORKING_DIR + 'train/')))
# print(len(os.listdir(WORKING_DIR + 'train_masks/')))

# train_dir = WORKING_DIR + 'train/'
# val_dir = WORKING_DIR + 'val/'
# for file in sorted(os.listdir(train_dir))[3200:]:
#     shutil.move(train_dir + file, val_dir)

# masks_dir = WORKING_DIR + 'train_masks/'
# val_masks_dir = WORKING_DIR + 'val_masks/'
# for file in sorted(os.listdir(masks_dir))[3200:]:
#     shutil.move(masks_dir + file, val_masks_dir)

# print(len(os.listdir(WORKING_DIR + 'train/')))
# print(len(os.listdir(WORKING_DIR + 'train_masks/')))
# print(len(os.listdir(WORKING_DIR + 'val/')))
# print(len(os.listdir(WORKING_DIR + 'val_masks/')))

# # # Show first 10 files in train and train_masks directories
# print(sorted(os.listdir(WORKING_DIR + 'train'))[:10])
# print(sorted(os.listdir(WORKING_DIR + 'train_masks'))[:10])
# print(sorted(os.listdir(WORKING_DIR + 'val'))[:10])
# print(sorted(os.listdir(WORKING_DIR + 'val_masks'))[:10])

from torch.utils.data import Dataset

# write function to add SP channels to image

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None, sp_transform=None, SP=False):
        self.SP = SP
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.sp_transform = sp_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.SP is True: 
            sp_tensor = sp_encode(image)
            if sp_tensor.shape[0] == 5:
                # add a channel of 0's so all tensors have 9 channels at the end
                zs = np.zeros((1, sp_tensor.shape[1], sp_tensor.shape[2]))
                sp_tensor = torch.tensor(np.concatenate((sp_tensor, zs), axis=0))
            # stick tensor to original 3 channel rgb image
            img = torch.tensor(image).permute(2,0,1)
            sp_tensor = sp_tensor.unsqueeze(0)  
            img = img.unsqueeze(0)  
            combined_tensor = torch.cat((sp_tensor, img), dim=1)  
            image = combined_tensor.squeeze(0)  
            image = image.permute(1,2,0)
            image = np.array(image, dtype='uint8') #uint 8 to allow for resizing
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        
        if SP:
            augmentations = self.sp_transform(image=image, mask=mask)
        else:
            augmentations = self.img_transform(image=image, mask=mask)
        image = augmentations['image']
        mask = augmentations['mask']

            
            
            
        return image, mask
    
# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 1
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '/home/jroberts2/Carvana/working/train/'
TRAIN_MASK_DIR = '/home/jroberts2/Carvana/working/train_masks/'
VAL_IMG_DIR = '/home/jroberts2/Carvana/working/val/'
VAL_MASK_DIR = '/home/jroberts2/Carvana/working/val_masks/'

import albumentations as A
from albumentations.pytorch import ToTensorV2

################################################### SP TRUE
sp_train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), # SP stuff doesn't like the resize operation for some reason
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


sp_test_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
###################################################

################################################### for SP FALSE
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(), 
    ]
)
###################################################

train_ds = CarvanaDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, img_transform=train_transform, sp_transform=sp_train_transform,SP=SP) # change transform and SP

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

test_ds = CarvanaDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, img_transform=test_transform, sp_transform=sp_test_transform, SP=SP) # change SP

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

# Initialize UNet model
model = UNet(in_channels) # change parameters if using sp's

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.to(DEVICE)
print("Device = ", DEVICE)

# from torchsummary import summary
# summary(model, (3, 256, 256))

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in  loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(f'Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}')
    model.train()
    
import torchvision

def save_predictions_as_imgs(
        loader, model, folder=WORKING_DIR+'saved_images', device='cuda'
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f'{folder}/truth_{idx}.png'
        )
        
from tqdm import tqdm
scaler = torch.cuda.amp.GradScaler()

images_folder = 'saved_images_sp' if SP else 'saved_images'

print("Device = ", DEVICE)

for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

          # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # print("A.", predictions.shape)
            # if SP:
            #     predictions = predictions.unsqueeze(2)
            #     print("B.", predictions.shape)
            loss = criterion(predictions, targets)


        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    check_accuracy(test_loader, model, device=DEVICE)
save_predictions_as_imgs(test_loader, model, folder=WORKING_DIR+images_folder, device=DEVICE)

print('Finished Training')


epoch_losses = []
model.eval()
loop = tqdm(test_loader)
for batch_idx, (data, targets) in enumerate(loop):
      data = data.to(device=DEVICE)
      targets = targets.float().unsqueeze(1).to(device=DEVICE)
      preds = model(data)
      loss = criterion(preds, targets)
      epoch_losses.append(loss.detach().cpu().numpy())

print(f"average losses: {np.mean(epoch_losses)}")
