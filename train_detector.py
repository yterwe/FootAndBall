# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Train FootAndBall detector on ISSIA-CNR Soccer and SoccerPlayerDetection dataset
#

import tqdm
import argparse
import pickle
import numpy as np
import os
import time

import torch
import torch.optim as optim

from network import footandball
from data.data_reader import make_dataloaders
from network.ssd_loss import SSDLoss
from misc.config import Params
from misc import utils
import matplotlib.pyplot as plt
import metric

MODEL_FOLDER = 'models'

def plot_training_stats(training_stats, model_name):
    metrics = ['loss', 'loss_ball_c', 'loss_player_c', 'loss_player_l', 
               'val_ap_ball', 'val_ap_player', 'val_map']
    for metric in metrics:
        plt.figure()
        for phase in training_stats:
            values = [epoch_stats[metric] for epoch_stats in training_stats[phase]]
            if len(values) == 0:
                continue
        plt.plot(values, label=phase)
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f'{model_name}_{metric}.png'
        plt.savefig(filename)
        print(f'Saved: {filename}')
        plt.close()

def train_model(model, optimizer, scheduler, num_epochs, dataloaders, device, model_name):
    # Weight for components of the loss function.
    # Ball-related loss and player-related loss are mean losses (loss per one positive example)
    alpha_l_player = 0.01
    alpha_c_player = 1.
    alpha_c_ball = 5.

    # Normalize weights
    total = alpha_l_player + alpha_c_player + alpha_c_ball
    alpha_l_player /= total
    alpha_c_player /= total
    alpha_c_ball /= total

    # Loss function
    criterion = SSDLoss(neg_pos_ratio=3)

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    training_stats = {'train': [], 'val': []}

    print('Training...')
    for epoch in tqdm.tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            batch_stats = {'loss': [], 'loss_ball_c': [], 'loss_player_c': [], 'loss_player_l': []}

            count_batches = 0
            # Iterate over data.
            for ndx, (images, boxes, labels) in enumerate(dataloaders[phase]):
                images = images.to(device)
                h, w = images.shape[-2], images.shape[-1]
                gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
                gt_maps = [e.to(device) for e in gt_maps]
                count_batches += 1

                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(images)
                    # Backpropagation
                    optimizer.zero_grad()
                    loss_l_player, loss_c_player, loss_c_ball = criterion(predictions, gt_maps)

                    loss = alpha_l_player * loss_l_player + alpha_c_player * loss_c_player + alpha_c_ball * loss_c_ball

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                count_batches += 1
                batch_stats['loss'].append(loss.item())
                batch_stats['loss_ball_c'].append(loss_c_ball.item())
                batch_stats['loss_player_c'].append(loss_c_player.item())
                batch_stats['loss_player_l'].append(loss_l_player.item())

            # Average stats per batch
            avg_batch_stats = {}
            for e in batch_stats:
                avg_batch_stats[e] = np.mean(batch_stats[e])

            training_stats[phase].append(avg_batch_stats)

            # Optionally evaluate mAP during validation
            if phase == 'val':
                all_detections = []
                all_groundtruths = []
            
                for images, boxes, labels in dataloaders['val']:
                    images = images.to(device)
                    with torch.no_grad():
                        outputs = model(images)
                    for i in range(len(images)):
                        preds = {
                            "boxes": outputs[i]["boxes"],
                            "scores": outputs[i]["scores"],
                            "labels": outputs[i]["labels"]
                        }
                        gts = {
                            "boxes": boxes[i],
                            "labels": labels[i]
                        }
                        all_detections.append(preds)
                        all_groundtruths.append(gts)
            
                ap_results = metric.compute_ap_map(all_detections, all_groundtruths)
            
                print(f'[Validation AP] Ball AP: {ap_results.get(0, 0.0):.4f}, '
                      f'Player AP: {ap_results.get(1, 0.0):.4f}, '
                      f'mAP: {ap_results.get("mAP", 0.0):.4f}')
            
                # Add to stats so you can plot/save later if needed
                avg_batch_stats['val_ap_ball'] = ap_results.get(0, 0.0)
                avg_batch_stats['val_ap_player'] = ap_results.get(1, 0.0)
                avg_batch_stats['val_map'] = ap_results.get("mAP", 0.0)


            
            s = '{} Avg. loss total / ball conf. / player conf. / player loc.: {:.4f} / {:.4f} / {:.4f} / {:.4f}'
            print(s.format(phase, avg_batch_stats['loss'], avg_batch_stats['loss_ball_c'],
                           avg_batch_stats['loss_player_c'], avg_batch_stats['loss_player_l']))

        # Scheduler step
        scheduler.step()
        print('')

    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)

    with open('training_stats_{}.pickle'.format(model_name), 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_training_stats(training_stats, model_name)

    return training_stats


def train(params: Params):
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    assert os.path.exists(MODEL_FOLDER), ' Cannot create folder to save trained model: {}'.format(MODEL_FOLDER)

    dataloaders = make_dataloaders(params)
    print('Training set: Dataset size: {}'.format(len(dataloaders['train'].dataset)))
    if 'val' in dataloaders:
        print('Validation set: Dataset size: {}'.format(len(dataloaders['val'].dataset)))

    # Create model
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = footandball.model_factory(params.model, 'train')
    model.print_summary(show_architecture=True)
    model = model.to(device)

    model_name = 'model_' + time.strftime("%Y%m%d_%H%M")
    print('Model name: {}'.format(model_name))

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)
    train_model(model, optimizer, scheduler, params.epochs, dataloaders, device, model_name)


if __name__ == '__main__':
    print('Train FoootAndBall detector on ISSIA dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))

    params = Params(args.config)
    params.print()

    train(params)


'''
NOTES:
In camera 5 some of the player bboxes are moved by a few pixels from the true position.
When evaluating mean precision use smaller IoU ratio, otherwise detection results are poor.
Alternatively add some margin to ISSIA ground truth bboxes.
'''
