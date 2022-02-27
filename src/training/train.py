import os
import time
import json
from contextlib import nullcontext
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.distributed.nn

from .zero_shot import zero_shot_eval
from .cross_entropy import pairwise_cross_entropy
import bitsandbytes.functional as F

import sys
import pdb
import wandb

import logging

def is_master(args):
    return (not args.distributed) or args.rank == 0

def get_loss(model, images, texts, loss_img, loss_txt, args):
    assert not args.block_size or not args.sharded_loss, "loss can be either blocky or sharded, not both"
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate and not args.sharded_loss:
        # We gather tensors from all gpus to get more negatives to contrast with.
        all_image_features, all_text_features = all_gather_compressed(image_features, text_features)

        if args.block_size:
            return pairwise_cross_entropy(logit_scale * all_image_features, all_text_features, args.block_size)

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    elif args.distributed and args.aggregate and args.sharded_loss:
        gathered_image_features = torch.cat(dist.nn.all_gather(image_features.half())).to(text_features.dtype)
        gathered_text_features = torch.cat(dist.nn.all_gather(text_features.half())).to(text_features.dtype)
        logits_per_image = logit_scale * image_features @ gathered_text_features.t()
        logits_per_text = logit_scale * text_features @ gathered_image_features.t()
    else:
        if args.block_size:
            return pairwise_cross_entropy(logit_scale * image_features, text_features, args.block_size)

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    
    ground_truth = torch.arange(len(logits_per_image), dtype=torch.long, device=logits_per_image.device)

    if args.sharded_loss:
        ground_truth += dist.get_rank() * len(logits_per_image)
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2
    return total_loss


def all_gather_compressed(image_features, text_features):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    text_codes, (text_absmax, text_codebook) = F.quantize_blockwise(text_features)
    gathered_text_codes = [torch.zeros_like(text_codes) for _ in range(world_size)]
    handle_text1 = dist.all_gather(gathered_text_codes, text_codes, async_op=True)
    gathered_text_absmax = [torch.zeros_like(text_absmax) for _ in range(world_size)]
    handle_text2 = dist.all_gather(gathered_text_absmax, text_absmax, async_op=True)
    gathered_text_codebook = [torch.zeros_like(text_codebook) for _ in range(world_size)]
    handle_text3 = dist.all_gather(gathered_text_codebook, text_codebook, async_op=True)


    image_codes, (image_absmax, image_codebook) = F.quantize_blockwise(image_features)
    gathered_image_codes = [torch.zeros_like(image_codes) for _ in range(world_size)]
    handle_image1 = dist.all_gather(gathered_image_codes, image_codes, async_op=True)
    gathered_image_absmax = [torch.zeros_like(image_absmax) for _ in range(world_size)]
    handle_image2 = dist.all_gather(gathered_image_absmax, image_absmax, async_op=True)
    gathered_image_codebook = [torch.zeros_like(image_codebook) for _ in range(world_size)]
    handle_image3 = dist.all_gather(gathered_image_codebook, image_codebook, async_op=True)

    handle_text1.wait(), handle_text2.wait(), handle_text3.wait()
    gathered_text_features = [F.dequantize_blockwise(
        gathered_text_codes[i], (gathered_text_absmax[i], gathered_text_codebook[i])).to(text_features.dtype)
                              for i in range(world_size)]
    all_text_features = torch.cat(
        [text_features]
        + gathered_text_features[:rank]
        + gathered_text_features[rank + 1 :]
    )


    handle_image1.wait(), handle_image2.wait(), handle_image3.wait()
    gathered_image_features = [F.dequantize_blockwise(
        gathered_image_codes[i], (gathered_image_absmax[i], gathered_image_codebook[i])).to(image_features.dtype)
                              for i in range(world_size)]
    all_image_features = torch.cat(
        [image_features]
        + gathered_image_features[:rank]
        + gathered_image_features[rank + 1 :]
    )
    return all_image_features, all_text_features


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, texts = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast(), model.no_sync() if args.grad_compression == 'no_sync' else nullcontext():
                total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)

            scaler.scale(total_loss).backward()
            total_loss = total_loss.item()
            if 'power' in args.grad_compression and step == args.power_sgd_warmup - 1:
                torch.cuda.synchronize(images.device); torch.cuda.empty_cache()
            scaler.step(optimizer)
            scaler.update()

        else:
            with model.no_sync() if args.grad_compression == 'no_sync' else nullcontext():
                total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)
            if 'power' in args.grad_compression and step == args.power_sgd_warmup - 1:
                torch.cuda.synchronize(images.device); torch.cuda.empty_cache()
            total_loss.backward()
            total_loss = total_loss.item()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 1) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss:.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.item():.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss,
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return
    
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    dataloader = data['val'].dataloader

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in dataloader:
            images, texts = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                texts = texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        metrics = get_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale
        )
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )
        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
