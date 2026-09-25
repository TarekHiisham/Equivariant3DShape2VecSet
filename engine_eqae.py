# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import e3nn.o3 as o3

import util.misc as misc
import util.lr_sched as lr_sched

num_sample = 512

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, labels, surface) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        R = o3.rand_matrix().to(dtype=torch.float32)
        D = model.irreps.D_from_matrix(R)

        R = R.to(device)
        D = D.to(device)

        points_rot = torch.einsum('ij, bnj -> bni', R, points)
        surface_rot = torch.einsum('ij, bnj -> bni', R, surface)

        with torch.amp.autocast('cuda', enabled=False):
            o = model(surface, points, return_latents=True)

            # Invariance results 
            o_rot = model(surface_rot, points_rot, return_latents=True)

            outputs = o['logits']
            outputs_rot = o_rot['logits']


            loss_vol = criterion(outputs[:, :num_sample], labels[:, :num_sample])
            loss_near = criterion(outputs[:, num_sample:], labels[:, num_sample:])
            loss = loss_vol + 0.1 * loss_near

            loss_vol_rot = criterion(outputs_rot[:, :num_sample], labels[:, :num_sample])
            loss_near_rot = criterion(outputs_rot[:, num_sample:], labels[:, num_sample:])
            loss_rot = loss_vol_rot + 0.1 * loss_near_rot

            loss = loss + loss_rot

        loss_value = loss.item()

        threshold = 0

        pred = torch.zeros_like(outputs[:, :num_sample])
        pred[outputs[:, :num_sample]>=threshold] = 1

        accuracy = (pred==labels[:, :num_sample]).float().sum(dim=1) / labels[:, :num_sample].shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels[:, :num_sample]).sum(dim=1)
        union = (pred + labels[:, :num_sample]).gt(0).sum(dim=1) + 1e-5
        iou = intersection * 1.0 / union
        iou = iou.mean()

        pred_rot = torch.zeros_like(outputs_rot[:, :num_sample])
        pred_rot[outputs_rot[:, :num_sample]>=threshold] = 1

        accuracy_rot = (pred_rot==labels[:, :num_sample]).float().sum(dim=1) / labels[:, :num_sample].shape[1]
        accuracy_rot = accuracy_rot.mean()
        intersection_rot = (pred_rot * labels[:, :num_sample]).sum(dim=1)
        union_rot = (pred_rot + labels[:, :num_sample]).gt(0).sum(dim=1) + 1e-5
        iou_rot = intersection_rot * 1.0 / union_rot
        iou_rot = iou_rot.mean()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())

        metric_logger.update(loss_vol_rot=loss_vol_rot.item())
        metric_logger.update(loss_near_rot=loss_near_rot.item())

        metric_logger.update(iou=iou.item())
        metric_logger.update(iou_rot=iou_rot.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.BCELoss()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for points, labels, surface in metric_logger.log_every(data_loader, 50, header):

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        R = o3.rand_matrix().to(dtype=torch.float32)
        D = model.irreps.D_from_matrix(R)

        R = R.to(device)
        D = D.to(device)

        points_rot = torch.einsum('ij, bnj -> bni', R, points)
        surface_rot = torch.einsum('ij, bnj -> bni', R, surface)

        # compute output
        with torch.amp.autocast('cuda', enabled=True):

            o = model(surface, points, return_latents=True)
            # Invariance results 
            o_rot = model(surface_rot, points_rot, return_latents=True)
            
            outputs = o['logits']
            outputs_rot = o_rot['logits']

            loss = criterion(outputs, labels)
            loss_rot = criterion(outputs_rot, labels)

            loss = loss + loss_rot

        threshold = 0

        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        pred_rot = torch.zeros_like(outputs_rot)
        pred_rot[outputs_rot>=threshold] = 1

        accuracy_rot = (pred_rot==labels).float().sum(dim=1) / labels.shape[1]
        accuracy_rot = accuracy_rot.mean()
        intersection_rot = (pred_rot * labels).sum(dim=1)
        union_rot = (pred_rot + labels).gt(0).sum(dim=1)
        iou_rot = intersection_rot * 1.0 / union_rot + 1e-5
        iou_rot = iou_rot.mean()

        batch_size = points.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_rot=loss_rot.item())

        metric_logger.meters['iou'].update(iou.item(), n=batch_size)
        metric_logger.meters['iou_rot'].update(iou_rot.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* iou {iou.global_avg:.3f} iou_rot {iou_rot.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(iou=metric_logger.iou, iou_rot=metric_logger.iou_rot, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}