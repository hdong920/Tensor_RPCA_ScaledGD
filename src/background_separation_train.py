import numpy as np
import random
import os
import tensorly as tl
import json
import torch
import skvideo.io
from training_functions import process_grads
from model import TensorRPCANet
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tl.set_backend('pytorch')

data_dir = 'data/bmc/' # data location
output_path = 'outputs/background_separation/'

# Partition the 9 videos into training and validation
train_vids = [1, 2, 3, 4, 5, 6] 
val_vids = [7, 8, 9]


skip = [1, 2, 3] # skip these dims
max_frames = 100 # Max number of frames per video
downsample = 2 # downsampling factor along the H and W dims of the video
frame_rank = 1  # low rank

frame_to_save = 42 # create a snapshot of this frame

config = dict()

# config['training_type'] = 'supervised' # supervised learning
config['training_type'] = 'self_supervised_cold' # self-supervised learning from scratch

config["epochs"] = 15
config['fine_tuning_epochs'] = 0 

config['T'] = 150 # number of iterative updates of RPCA

# Network Initialization
config["z0_init"] = 0
config["z1_init"] = 0
config["eta_init"] = 0.1
config["decay_init"] = 0.8

config['log_interval'] = 5

# Optimization Parameters
config['lr'] = 0.05
config['softplus_factor'] = 0.01
config['grad_clip'] = 100
config['scheduler_decay'] = 0.5
config['patience'] = 2

config['eps'] = 1e-7

metrics_path = f'{output_path}{config["training_type"]}_metrics.json'

Ys = []
S_stars = []
masks = []
for i in range(1, 10):
    vid_data_dir = f'{data_dir}Video_00{i}/img/'
    vid_output_path = f'{output_path}Video_00{i}/'
    video = []
    video_mask = []
    for filename in sorted(os.listdir(vid_data_dir)):
        if filename[-3:] != "bmp":
            continue
        f = os.path.join(vid_data_dir, filename)
        image = Image.open(f)
        if "Img" in filename:
            video.append(np.array(image))
            if f"Mask{filename[3:]}" not in os.listdir(vid_data_dir):
                print(filename)

        if "Mask" in filename:
            video_mask.append(np.array(image))
    Y = torch.Tensor(np.array(video))[:min(max_frames, len(video)), ::downsample, ::downsample] / 255
    video_mask = torch.Tensor(np.array(video_mask))[:min(max_frames, len(video)), ::downsample, ::downsample].unsqueeze(-1) / 255
    S_star = Y * video_mask
    X_star = Y * (1-video_mask)
    skvideo.io.vwrite(f'{vid_output_path}Y.mp4', Y.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite(f'{vid_output_path}X_star.mp4', X_star.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite(f'{vid_output_path}S_star.mp4', S_star.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
    Ys.append(Y)
    masks.append(video_mask)
    S_stars.append(S_star)


model = TensorRPCANet(config["z0_init"], config["z1_init"], config["eta_init"], config["decay_init"], device, config['softplus_factor'], skip=skip)

optimizer = torch.optim.Adam([
            {'params': model.z0, 'lr': config['lr'] },
            {'params': model.z1, 'lr': config['lr'] },
            {'params': model.eta, 'lr': config['lr']},
            {'params': model.decay, 'lr': config['lr'] },
        ],)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['scheduler_steps'], gamma=config['scheduler_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_decay'], patience=config['patience'])

metrics = dict()
metrics['train_loss_traj'] = []
metrics['train_reconstruction_loss_traj'] = []
metrics['train_X_loss_traj'] = []
metrics['val_loss_traj'] = []
metrics['val_reconstruction_loss_traj'] = []
metrics['val_X_loss_traj'] = []
metrics['z0_traj'] = [model.z0.item()]
metrics['z1_traj'] = [model.z1.item()]
metrics['eta_traj'] = [model.eta.item()]
metrics['decay_traj'] = [model.decay.item()]


print("Beginning Training")
for epoch in range(config['epochs']):
    model.train()
    random.shuffle(train_vids)
    total_train_loss = 0
    for i in train_vids:
        
        optimizer.zero_grad()
        Y = Ys[i-1].to(device)
        S_star = S_stars[i-1].to(device)
        mask = masks[i-1].to(device)
        rank = [frame_rank, Y.shape[1], Y.shape[2], 3]
        X, S = model(Y, rank, config["T"], epsilon=config["eps"])
        if config['training_type'] == 'supervised':
            loss = ((Y-X) * (1-mask)).norm()**2 / (Y * (1-mask)).norm()**2
        elif config['training_type'] == 'self_supervised_cold':
            loss = (Y-X).norm(p=1) / (Y).norm()**2

        total_train_loss += loss.item()
        loss.backward()
        process_grads(model, config)
        
        optimizer.step()

        metrics['train_loss_traj'].append(loss.item())
        metrics['train_reconstruction_loss_traj'].append(((Y - X - S).norm()**2 / Y.norm()**2).item())
        metrics['train_X_loss_traj'].append((((Y-X) * (1-mask)).norm()**2 / (Y * (1-mask)).norm()**2).item())
        metrics['z0_traj'].append(model.z0.item())
        metrics['z1_traj'].append(model.z1.item())
        metrics['eta_traj'].append(model.eta.item())
        metrics['decay_traj'].append(model.decay.item())

        if epoch % config['log_interval'] == config['log_interval'] - 1:
            vid_output_path = f'{output_path}Video_00{i}/'
            skvideo.io.vwrite(f"{vid_output_path}X_{config['training_type']}_train.mp4", X.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
            skvideo.io.vwrite(f"{vid_output_path}S_{config['training_type']}_train.mp4", S.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
            plt.imshow(Y[frame_to_save].cpu().detach().numpy() )
            plt.axis('off')
            plt.savefig(f"{vid_output_path}Y{frame_to_save}_{config['training_type']}_train.eps", bbox_inches='tight')
            plt.close()
            plt.imshow(X[frame_to_save].cpu().detach().numpy() )
            plt.axis('off')
            plt.savefig(f"{vid_output_path}X{frame_to_save}_{config['training_type']}_train.eps", bbox_inches='tight')
            plt.close()
            plt.imshow(S[frame_to_save].cpu().detach().numpy() )
            plt.axis('off')
            plt.savefig(f"{vid_output_path}S{frame_to_save}_{config['training_type']}_train.eps", bbox_inches='tight')
            plt.close()
    
    scheduler.step(total_train_loss / len(train_vids))
        
    with open(metrics_path, 'w') as fp:
        json.dump(metrics, fp)
    print("EPOCH ", epoch)
    print("Train Mean loss: ", np.mean(metrics['train_loss_traj'][-len(train_vids):]))
    print("Train Mean X loss: ", np.mean(metrics['train_X_loss_traj'][-len(train_vids):]))
    print("Train Mean reconstruction loss: ", np.mean(metrics['train_reconstruction_loss_traj'][-len(train_vids):]))
                
    model.eval()
    with torch.no_grad():
        for i in val_vids:
            Y = Ys[i-1].clone().to(device)
            S_star = S_stars[i-1].clone().to(device)
            mask = masks[i-1].clone().to(device)
            rank = [frame_rank, Y.shape[1], Y.shape[2], 3]
            X, S = model(Y, rank, config["T"], epsilon=config["eps"])
            if config['training_type'] == 'supervised':
                loss = ((Y-X) * (1-mask)).norm()**2 / (Y * (1-mask)).norm()**2
            elif config['training_type'] == 'self_supervised_cold':
                loss = (Y-X).norm(p=1) / (Y).norm()**2
            metrics['val_loss_traj'].append(loss.item())
            metrics['val_reconstruction_loss_traj'].append(((Y - X - S).norm()**2 / Y.norm()**2).item())
            metrics['val_X_loss_traj'].append((((Y-X) * (1-mask)).norm()**2 / (Y * (1-mask)).norm()**2).item())
            if epoch % config['log_interval'] == config['log_interval'] - 1:
                vid_output_path = f'{output_path}Video_00{i}/'
                skvideo.io.vwrite(f"{vid_output_path}X_{config['training_type']}_val.mp4", X.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
                skvideo.io.vwrite(f"{vid_output_path}S_{config['training_type']}_val.mp4", S.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})

                plt.imshow(Y[frame_to_save].cpu().detach().numpy() )
                plt.axis('off')
                plt.savefig(f"{vid_output_path}Y{frame_to_save}_{config['training_type']}_val.eps", bbox_inches='tight')
                plt.close()
                plt.imshow(X[frame_to_save].cpu().detach().numpy() )
                plt.axis('off')
                plt.savefig(f"{vid_output_path}X{frame_to_save}_{config['training_type']}_val.eps", bbox_inches='tight')
                plt.close()
                plt.imshow(S[frame_to_save].cpu().detach().numpy() )
                plt.axis('off')
                plt.savefig(f"{vid_output_path}S{frame_to_save}_{config['training_type']}_val.eps", bbox_inches='tight')
                plt.close()
            print(f"Val {i} loss: ", metrics['val_loss_traj'][-1])
            print(f"Val {i} X loss: ", metrics['val_X_loss_traj'][-1])
            print(f"Val {i} reconstruction loss: ", metrics['val_reconstruction_loss_traj'][-1])


z0 = model.z0.item()
z1 = model.z1.item()
eta = model.eta.item()
decay = model.decay.item()

for i in val_vids:
    del model
    print(f"Fine tuning {i}:")
    model = TensorRPCANet(z0, z1, eta, decay, device, config['softplus_factor'], skip=skip)

    optimizer = torch.optim.Adam([
            {'params': model.z0, 'lr': config['lr'] / 2},
            {'params': model.z1, 'lr': config['lr'] / 2},
            {'params': model.eta, 'lr': config['lr'] / 2},
            {'params': model.decay, 'lr': config['lr'] / 2},
        ],)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_decay'], patience=config['patience'])
    
    # model.train()
    Y = Ys[i-1].clone().to(device)
    mask = masks[i-1].clone().to(device)
    rank = [frame_rank, Y.shape[1], Y.shape[2], 3]
    with torch.no_grad():
        X, S = model(Y, rank, config["T"], epsilon=config["eps"])
        original_X_loss = (((Y-X) * (1-mask)).norm()**2 / (Y * (1-mask)).norm()**2).item()
        print("ORIGINAL LOSS: ", original_X_loss)
    
    for epoch in range(config['fine_tuning_epochs']):
        X, S = model(Y, rank, config["T"], epsilon=config["eps"])
        loss = (Y-X).norm(p=1) / (Y).norm()**2
        loss.backward()
        process_grads(model, config)
        optimizer.step()
        print("loss: ", loss.item())
        print("X loss: ", (((Y-X) * (1-mask)).norm()**2 / (Y * (1-mask)).norm()**2).item())
        print("reconstruction loss: ", ((Y - X - S).norm()**2 / Y.norm()**2).item())
    vid_output_path = f'{output_path}Video_00{i}/'
    skvideo.io.vwrite(f"{vid_output_path}X_{config['training_type']}_fine_tuned.mp4", X.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
    skvideo.io.vwrite(f"{vid_output_path}S_{config['training_type']}_fine_tuned.mp4", S.cpu().detach().numpy() * 255, outputdict={"-pix_fmt": "yuv420p"})
    