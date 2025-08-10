from torch.utils.data import DataLoader
import torch
import logging
from utils.helper import SaveHandler, AverageMeter, extract_patches, reassemble_patches
from utils.trainer import Trainer
from models.model import vgg19
from datasets.dataset import Crowd
import numpy as np
import os
import time
import random

import torch.nn.functional as F
from utils.ssim_loss import cal_avg_ms_ssim
import torch.nn as nn
from utils.evaluation import eval_game, eval_relative



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_normalized_map(density_map):
    B, C, H, W = density_map.size()
    mu_sum = density_map.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_normed = density_map / (mu_sum + 1e-6)
    return mu_normed


class Reg_Trainer(Trainer):
    def setup(self):
        args = self.args
        if args.seed != -1:
            setup_seed(args.seed)
            print('Random seed is set as {}'.format(args.seed))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_device(0)
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('Using {} gpus'.format(self.device_count))
        else:
            raise Exception('GPU is not available')

        self.d_ratio = args.downsample_ratio

        self.datasets = {x: Crowd(args,
                                  method=x) for x in ['train', 'val', 'test']}

        logging.info('Number of images in the training set: {}, validation set:{}, test set: {}'.format(
            len(self.datasets['train']), len(self.datasets['val']), len(self.datasets['test'])))

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(args.batch_size if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False)
                                          ) for x in ['train', 'val', 'test']}

        self.model = vgg19(args)
        self.model.to(self.device)
        self.model.depth_feat.pipeline.to(self.device)
        self.count = torch.Tensor([0, 1, 2, 3, 4, 5]).float().unsqueeze(0).to(self.device)
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': args.lr},
            {'params': [p for n, p in self.model.depth_feat.pipeline.unet.named_parameters() if p.requires_grad], 'lr': args.lr}],
            weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if args.density_lora_path is None:
                raise Exception('Please provide your pretrained Density lora')
            else:
                self.model.depth_feat.pipeline.unet.from_pretrained(args.density_lora_path)
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                raise Exception('Not supported')

        self.best_game0 = np.inf
        self.best_game3 = np.inf
        self.save_list = SaveHandler(num=args.max_num, save_dir=self.save_dir)


    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            logging.info('-' * 50 + "Epoch:{}/{}".format(epoch, args.epochs - 1) + '-' * 50)
            self.epoch = epoch
            self.train_epoch()
            if self.epoch >= args.start_val and self.epoch % args.val_epoch == 0:
                game0_is_best, game3_is_best = self.val_epoch()
                if epoch >= args.start_val and (game0_is_best or game3_is_best):
                    self.test_epoch()

    def train_epoch(self):
        epoch_reg_loss = AverageMeter()
        epoch_seg_loss = AverageMeter()
        epoch_dist_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()

        epoch_start = time.time()
        self.model.train()
        self.model.depth_feat.setup_train()

        for step, (thermal_inputs, thermal_d_inputs, den_map, token_ids, pt_map) in enumerate(self.dataloaders['train']):
            thermal_inputs = thermal_inputs.to(self.device)
            thermal_d_inputs = thermal_d_inputs.to(self.device)
            gt_den_map = den_map.to(self.device)
            token_ids = token_ids.to(self.device)
            pt_map = pt_map.to(self.device)

            with torch.set_grad_enabled(True):
                pred_den, pred_sim = self.model(thermal_inputs, thermal_d_inputs, token_ids)
                N = thermal_inputs.shape[0]
                reg_loss = get_reg_loss(pred_den, gt_den_map)
                labels = pt_map.view(-1)
                pos_mask = labels != 0
                neg_mask = labels == 0
                keep_neg = neg_mask.nonzero().squeeze()
                subset = keep_neg[torch.randperm(len(keep_neg))[:int(len(keep_neg) * 0.1)]]
                final_mask = pos_mask | torch.zeros_like(pos_mask).bool().scatter(0, subset, True)
                sim_loss = F.cross_entropy(pred_sim.view(-1, pred_sim.shape[-1])[final_mask], labels[final_mask])

                epoch_reg_loss.update(reg_loss.item(), N)
                epoch_seg_loss.update(sim_loss.item(), N)

                loss = reg_loss + sim_loss

                gt_counts = torch.sum(gt_den_map.view(N, -1), dim=1).detach().cpu().numpy()
                pred_counts = torch.sum(pred_den.view(N, -1), dim=1).detach().cpu().numpy()
                diff = pred_counts - gt_counts

                epoch_mae.update(np.mean(np.abs(diff)).item(), N)
                epoch_mse.update(np.mean(diff * diff), N)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        logging.info(
            'Epoch {} Train, reg:{:.4f}, seg:{:.4f}, dist:{:.4f} mae:{:.2f}, mse:{:.2f}, Cost: {:.1f} sec '
            .format(self.epoch, epoch_reg_loss.getAvg(), epoch_seg_loss.getAvg(), epoch_dist_loss.getAvg(), epoch_mae.getAvg(),
                    np.sqrt(epoch_mse.getAvg()), (time.time() - epoch_start)))

        if self.epoch % 5 == 0:
            save_epoch_dir = os.path.join(self.save_dir, "{}_ckpt".format(self.epoch))
            os.makedirs(save_epoch_dir, exist_ok=True)
            model_state_dict = self.model.state_dict()
            save_path = os.path.join(save_epoch_dir, "{}_ckpt.tar".format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dict,
            }, save_path)
            den_unet_dir = os.path.join(save_epoch_dir, "unet".format(self.epoch))
            os.makedirs(den_unet_dir, exist_ok=True)
            self.model.depth_feat.pipeline.unet.save_pretrained(den_unet_dir)
            self.save_list.append(save_epoch_dir)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        self.model.depth_feat.setup_eval()
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0
        for thermal_inputs, thermal_d_inputs, target, name, token_ids in self.dataloaders['val']:
            thermal_inputs = thermal_inputs.to(self.device)
            thermal_d_inputs = thermal_d_inputs.to(self.device)
            target = target.to(self.device)
            token_ids = token_ids.to(self.device)

            cropped_t, num_h, num_w = extract_patches(thermal_inputs, patch_size=384, stride=384)
            cropped_t_d, num_h, num_w = extract_patches(thermal_d_inputs, patch_size=384, stride=384)
            outputs = []
            with torch.set_grad_enabled(False):
                num_chunks = (cropped_t.size(0) + 3) // 4
                for i in range(num_chunks):
                    start_idx = i * 4
                    end_idx = min((i + 1) * 4, cropped_t.size(0))
                    outputs_partial = self.model(cropped_t[start_idx:end_idx], cropped_t_d[start_idx:end_idx], token_ids.repeat((end_idx-start_idx), 1, 1))[0]
                    outputs.append(outputs_partial)

                pred_den = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, thermal_inputs.size(2),
                                              thermal_inputs.size(3),
                                              patch_size=self.args.crop_size, stride=self.args.crop_size)

                for L in range(4):
                    abs_error, square_error = eval_game(pred_den, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(pred_den, target)
                total_relative_error += relative_error

        N = len(self.dataloaders['val'])
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N
        logging.info('Epoch {} Val{}, '
                     'GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f} Re {relative:.4f}, Time cost {time_cost:.1f}s'
                     .format(self.epoch, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0],
                             relative=total_relative_error, time_cost=time.time() - epoch_start
                             )
                     )

        model_state_dict = self.model.state_dict()

        game0_is_best = game[0] < self.best_game0
        game3_is_best = game[3] < self.best_game3

        if game[0] < self.best_game0 or game[3] < self.best_game3:
            self.best_game3 = min(game[3], self.best_game3)
            self.best_game0 = min(game[0], self.best_game0)
            logging.info("*** Best Test GAME0 {:.3f} GAME3 {:.3f} model epoch {}".format(self.best_game0,
                                                                                         self.best_game3,
                                                                                         self.epoch))
            save_best_dir = os.path.join(self.save_dir, 'best_model_{}'.format(self.epoch))
            os.makedirs(save_best_dir, exist_ok=True)
            torch.save(model_state_dict, os.path.join(save_best_dir, 'best_model.pth'.format(self.epoch)))
            best_unet_dir = os.path.join(save_best_dir, "unet".format(self.epoch))
            os.makedirs(best_unet_dir, exist_ok=True)
            self.model.depth_feat.pipeline.unet.save_pretrained(best_unet_dir)
        return game0_is_best, game3_is_best



def get_reg_loss(pred, gt, threshold=1e-3, level=3, window_size=3):
    mask = gt > threshold
    loss_ssim = cal_avg_ms_ssim(pred * mask, gt * mask, level=level,
                                window_size=window_size)
    mu_normed = get_normalized_map(pred)
    gt_mu_normed = get_normalized_map(gt)
    tv_loss = (nn.L1Loss(reduction='none')(mu_normed, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)
    return loss_ssim + 0.1 * tv_loss
