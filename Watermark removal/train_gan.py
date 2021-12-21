import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import torchvision
from torch.utils.tensorboard import SummaryWriter

import datasets_j as datasets
import utils
import loss_func
import cyclegan_model as models
import patchGAN

# Hyperparameters
IN_CHANNELS = 3
OUT_CHANNELS = 3
LEARNING_RATE_GEN = 1e-6
LEARNING_RATE_DIS = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = 1000
LOAD_MODEL = True
LAMBDA_VGG = 100
LAMBDA_L1 = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformation
TRAIN_MEAN = [0.5109, 0.4903, 0.4257]
TRAIN_STD = [0.2509, 0.2338, 0.1879]
image_normalize = transforms.Compose(
    [
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
    ]
)
image_denormalize = utils.NormalizeInverse(mean=TRAIN_MEAN, std=TRAIN_STD)

# Dataset & DataLoader
dataset = datasets.Watermark_Dataset(os.getcwd(),
                                     random_crop=True,
                                     transform=image_normalize)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model init
generator = models.cycleGan().to(DEVICE)
discriminator = patchGAN.Discriminator().to(DEVICE)

# Loss and optimizer
criterion = loss_func.VGGLoss(device=DEVICE)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999),)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_DIS, betas=(0.5, 0.999),)
scaler_gen = torch.cuda.amp.GradScaler()
scaler_dis = torch.cuda.amp.GradScaler()
mse = nn.MSELoss()
l1 = nn.L1Loss()

# Load model
if LOAD_MODEL:
    utils.load_checkpoint(torch.load("generator_gan.pth.tar"), generator, optimizer_gen)
    utils.load_checkpoint(torch.load("discriminator_gan.pth.tar"), discriminator, optimizer_dis)
    #utils.initialize_weights(discriminator)

# Weight initialization
if not LOAD_MODEL:
    utils.initialize_weights(generator)
    utils.initialize_weights(discriminator)

# Tensorboard
runs = os.listdir(os.path.join(os.getcwd(), 'runs'))
runs = [int(i.replace('watermark','')) for i in runs if i.replace('watermark','') != '']
writer = SummaryWriter(f"runs/watermark"+str(max(runs)+1))

# Train
step = 0
generator.train()
discriminator.train()

for epoch in range(NUM_EPOCHS):
    #losses = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for batch_idx, (data, label) in loop:
        data = data.to(device=DEVICE)
        label = label.to(device=DEVICE)
        noise = torch.randn(data.shape).to(device=DEVICE)
        noise *= 0.1

        skip = False
        if BATCH_SIZE == 1:
            with torch.no_grad():
                if l1(image_denormalize(data), label) < 0.005:
                    skip = True
        if skip:
            continue

        # discriminator forward
        with torch.cuda.amp.autocast():
            # adversarial loss
            pred_label = generator(data)
            dis_real = discriminator(label+noise)
            loss_dis_real = mse(dis_real, torch.ones_like(dis_real)*0.9)
            dis_fake = discriminator(pred_label.detach())
            loss_dis_fake = mse(dis_fake, torch.zeros_like(dis_fake)+0.1)
            dis_loss = loss_dis_real + loss_dis_fake

        # discriminator backward
        optimizer_dis.zero_grad()
        scaler_dis.scale(dis_loss).backward()
        scaler_dis.step(optimizer_dis)
        scaler_dis.update()

        # generator forward
        with torch.cuda.amp.autocast():
            # adversarial loss
            gen_fake = discriminator(pred_label)
            gen_adversarial_loss = mse(gen_fake, torch.ones_like(gen_fake))

            # identity VGG loss
            gen_identity_loss = criterion(pred_label, label)

            # identity l1 loss
            gen_l1_loss = l1(pred_label, label)

            # combine both
            gen_loss = gen_adversarial_loss + LAMBDA_VGG * gen_identity_loss + LAMBDA_L1 * gen_l1_loss

        #losses.append(loss.item())
        optimizer_gen.zero_grad()
        scaler_gen.scale(gen_loss).backward()
        scaler_gen.step(optimizer_gen)
        scaler_gen.update()

        # update progress bar
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(dis_loss=dis_loss.item(), gen_loss=gen_identity_loss.item())

        # tensorboard
        writer.add_scalar('Training discriminator loss', dis_loss, global_step=step)
        writer.add_scalar('Training generator loss', gen_loss, global_step=step)

        # update step
        step += 1

        # Print losses occasionally and print to tensorboard
        if step % 2 == 0:
            with torch.no_grad():
                writer.add_image("Output",
                                 torch.cat((image_denormalize(data), pred_label, label), 3).squeeze(0),
                                 global_step=step)

    # LR scheduler
    # mean_loss = sum(losses)/len(losses)
    # scheduler.step(mean_loss)

    # save model
    checkpoint_gen = {
        "state_dict": generator.state_dict(),
        "optimizer": optimizer_gen.state_dict(),
    }
    checkpoint_dis = {
        "state_dict": discriminator.state_dict(),
        "optimizer": optimizer_dis.state_dict(),
    }



# save checkpoint
utils.save_checkpoint(checkpoint_gen, filename='generator_gan.pth.tar')
utils.save_checkpoint(checkpoint_dis, filename='discriminator_gan.pth.tar')

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                       factor=0.1,
#                                                       patience=5,
#                                                       verbose=True)