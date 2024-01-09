import datetime
import logging
import os
import time
import gc
import torch
import torch.distributed as dist

from g2l.data import make_data_loader
from g2l.utils.comm import get_world_size, synchronize
from g2l.utils.metric_logger import MetricLogger
from g2l.engine.inference import inference
from ..utils.comm import is_main_process


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    loss = loss.item()
    return loss


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    param_dict,
    max_norm=5
):

    logger = logging.getLogger("g2l.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH+cfg.SOLVER.WARMUP_EPOCHS

    model.train()
    start_training_time = time.time()
    end = time.time()
    max_iteration = len(data_loader)
    writer_count = 0

    for epoch in range(arguments["epoch"], max_epoch + 1):
        rest_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch
        # data_loader.batch_sampler.sampler.set_epoch(epoch)
        if epoch <= cfg.SOLVER.FREEZE_BERT:
            for param in param_dict['bert']:
                param.requires_grad_(False)
        else:
            for param in param_dict['bert']:
                param.requires_grad_(True)
        logger.info("Start epoch {}. base_lr={:.1e}, bert_lr={:.1e}, bert.requires_grad={}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], str(param_dict['bert'][0].requires_grad)))
        if epoch <= cfg.SOLVER.ONLY_IOU:
            logger.info("Using all losses")
        else:
            logger.info("Using only bce loss")
        for iteration, (batches, idx) in enumerate(data_loader):
            writer_count += 1
            iteration += 1
            batches = batches.to(device)
            optimizer.zero_grad()
            contr_weight = cfg.MODEL.G2L.LOSS.CONTRASTIVE_WEIGHT
            bce_weight=cfg.MODEL.G2L.LOSS.BCE_WEIGHT
            geo_weight1 = cfg.MODEL.G2L.LOSS.HARD_GEO_W # for hard
            geo_weight2 = cfg.MODEL.G2L.LOSS.GEO_W # for hard
            sa_weight = cfg.MODEL.G2L.LOSS.SHAPLEY_W
            # as_cl_weight= cfg.MODEL.G2L.LOSS.AS_CL_WEIGHT

            loss_sa,loss_vid, loss_hard_sent, loss_sent,loss_iou = model(batches, cur_epoch=epoch,device=device,idx=idx)
            # loss_sa, loss_iou = model(batches, cur_epoch=epoch,device=device)
            loss_vid, loss_hard_sent, loss_sent , loss_iou, loss_sa= loss_vid * contr_weight, loss_hard_sent * contr_weight, loss_sent *contr_weight,loss_iou * bce_weight, loss_sa * sa_weight
            loss_hard_sent = loss_hard_sent * geo_weight1
            loss_sent = loss_sent * geo_weight2
            # loss_sa, loss_iou= loss_sa, loss_iou * bce_weight

            # loss_vid, loss_sent ,loss_dense_vid= loss_vid * contr_weight * as_cl_weight, loss_sent * contr_weight,loss_dense_vid* contr_weight * (1-as_cl_weight)
            loss = 0
            if epoch <= cfg.SOLVER.ONLY_IOU:
                loss += loss_iou
                # loss_cl=loss_sent + loss_vid+loss_dense_vid
                # loss = loss_iou+loss_cl/(loss_cl/loss_iou).detach()
                # print("loss += loss_sent + loss_vid+loss_dense_vid")

                loss += loss_sent + loss_vid+loss_hard_sent +loss_sa
            else:
                # loss += loss_iou + loss_sa
                loss += loss_iou
                loss += (loss_sent + loss_vid+loss_hard_sent +loss_sa) * 0.01
            # meters.update(loss_iou=loss_iou.detach(),all_loss=loss.detach())
            meters.update(loss_sa=loss_sa.detach(),loss_vid=loss_vid.detach(), loss_sent=loss_sent.detach(), loss_hard_sent=loss_hard_sent.detach(),loss_iou=loss_iou.detach(),all_loss=loss.detach())
            loss.backward()
            # max_norm=cfg.SOLVER.MAX_NORM
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            if cfg.SOLVER.LR_SCHEDULER!="multistep":
                scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (max_iteration - iteration + rest_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iteration:
                # logger.info("base_lr={:.1e}, bert_lr={:.1e}".format(optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], str(param_dict['bert'][0].requires_grad)))
                logger.info(
                    meters.delimiter.join(
                        [
                            "lr: {lr}",
                            "base_lr:{base_lr}",
                            "bert_lr:{bert_lr}"
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        lr=scheduler.get_last_lr()[0],
                        base_lr=optimizer.param_groups[0]["lr"], 
                        bert_lr=optimizer.param_groups[1]["lr"],
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            gc.collect()
        if cfg.SOLVER.LR_SCHEDULER=="multistep":
            scheduler.step()
        if checkpoint_period != -1 and epoch % checkpoint_period == 0:
            checkpointer.save(f"{cfg.MODEL.G2L.FEAT2D.NAME}_model_{epoch}e", **arguments)

        if data_loader_val is not None and test_period > 0 and epoch % test_period == 0 and epoch >= cfg.SOLVER.SKIP_TEST:
            synchronize()
            torch.cuda.empty_cache()
            result_dict = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            synchronize()
            model.train()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
