import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2


def init_dist(backend = 'nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none = True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ[ 'RANK' ])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend = backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type = str, help = 'Path to option YAML file.',
                        default = './options/train/LOLv2_real.yml')
    parser.add_argument('--launcher', choices = [ 'none', 'pytorch' ], default = 'none',
                        help = 'job launcher')
    parser.add_argument('--local_rank', type = int, default = 0)

    # TODO? 判断是否为消融实验
    parser.add_argument('--is_ablation',action = 'store_true')
    parser.add_argument('--datasetname',default = 'huawei',help = 'train datasetname...(huawei,lolv2_real,lolv2_syn,nikon)')

    # TODO?
    parser.add_argument('--down_type', type = str, default = 'ours', help = 'normal,akconv,mffa,pconv,wtconv,ours')
    parser.add_argument('--is_use_dalte', action = 'store_true')
    parser.add_argument('--is_use_freq', action = 'store_true')
    parser.add_argument('--fre_type', type = str, default = 'ours', help = 'dm,four,ours')

    # TODO?
    parser.add_argument('--pixel', type = float, default = 0.85, help = 'pixel weight')
    parser.add_argument('--ssim', type = float, default = 1., help = 'ssim weight')
    parser.add_argument('--vgg', type = float, default = 0.45, help = 'vgg weight')
    parser.add_argument('--color', type = float, default = 0.5, help = 'color weight')
    parser.add_argument('--gradient', type = float, default = 0.8, help = 'gradient weight')
    parser.add_argument('--hist', type = float, default = 0.75, help = 'hist weight')

    # TODO?
    parser.add_argument('--pixel_loss', action = 'store_true')
    parser.add_argument('--ssim_loss', action = 'store_true')
    parser.add_argument('--color_loss', action = 'store_true')
    parser.add_argument('--gradient_loss', action = 'store_true')
    parser.add_argument('--hist_loss', action = 'store_true')
    parser.add_argument('--vgg_loss', action = 'store_true')
    # ------------------------------------------------#
    #   TODO? 在这里加入消融实验可控条件
    # ------------------------------------------------#

    # TODO 1. 模块消融
    # TODO 2. 参数消融
    # TODO 3. 损失函数消融
    # TODO 4. 模块对比消融

    args = parser.parse_args()

    if args.is_ablation:
     sym = f"experiment_down_type_{args.down_type}_" \
          f"is_use_dalte_{args.is_use_dalte}_" \
          f"is_use_freq_{args.is_use_freq}_" \
          f"fre_type_{args.fre_type}_" \
          f"pixel_{args.pixel}_" \
          f"ssim_{args.ssim}_" \
          f"vgg_{args.vgg}_" \
          f"color_{args.color}_gradient_{args.gradient}_" \
          f"hist_{args.hist}_pixel_loss_{args.pixel_loss}_" \
          f"ssim_loss_{args.ssim_loss}_vgg_loss_{args.vgg_loss}_" \
          f"color_loss_{args.color_loss}_gradient_loss_{args.gradient_loss}_" \
          f"hist_loss_{args.hist_loss}"
    else:
        sym = f"{args.datasetname}_experiment_down_type_{args.down_type}_" \
             f"is_use_dalte_{args.is_use_dalte}_" \
             f"is_use_freq_{args.is_use_freq}_" \
             f"fre_type_{args.fre_type}_" \
             f"pixel_{args.pixel}_" \
             f"ssim_{args.ssim}_" \
             f"vgg_{args.vgg}_" \
             f"color_{args.color}_gradient_{args.gradient}_" \
             f"hist_{args.hist}_pixel_loss_{args.pixel_loss}_" \
             f"ssim_loss_{args.ssim_loss}_vgg_loss_{args.vgg_loss}_" \
             f"color_loss_{args.color_loss}_gradient_loss_{args.gradient_loss}_" \
             f"hist_loss_{args.hist_loss}"
    opt = option.parse(args.opt, is_train = True, sym = sym)
    # TODO? 数据集
    opt['datasetname'] = args.datasetname
    # TODO? 添加配置
    opt[ 'down_type' ] = args.down_type
    opt[ 'is_use_dalte' ] = args.is_use_dalte
    opt[ 'is_use_freq' ] = args.is_use_freq
    opt[ 'fre_type' ] = args.fre_type

    opt[ 'pixel' ] = args.pixel
    opt[ 'ssim' ] = args.ssim
    opt[ 'vgg' ] = args.vgg
    opt[ 'color' ] = args.color
    opt[ 'gradient' ] = args.gradient
    opt[ 'hist' ] = args.hist

    opt[ 'pixel_loss' ] = args.pixel_loss
    opt[ 'ssim_loss' ] = args.ssim_loss
    opt[ 'color_loss' ] = args.color_loss
    opt[ 'gradient_loss' ] = args.gradient_loss
    opt[ 'hist_loss' ] = args.hist_loss
    opt[ 'vgg_loss' ] = args.vgg_loss

    ###########################################
    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt[ 'dist' ] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt[ 'dist' ] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt[ 'path' ].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt[ 'path' ][ 'resume_state' ],
                                  map_location = lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state[ 'iter' ])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    # TODO?

    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt[ 'path' ][ 'experiments_root' ])  # rename experiment folder if exists

            util.mkdirs((path for key, path in opt[ 'path' ].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt[ 'path' ][ 'log' ], 'train_' + opt[ 'name' ], level = logging.INFO,
                          screen = True, tofile = True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt[ 'use_tb_logger' ] and 'debug' not in opt[ 'name' ]:
            version = float(torch.__version__[ 0:3 ])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir = './tb_logger/' + opt[ 'name' ])
    else:
        util.setup_logger('base', opt[ 'path' ][ 'log' ], 'train', level = logging.INFO, screen = True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt[ 'train' ][ 'manual_seed' ]
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt[ 'datasets' ].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)

            # import pdb; pdb.set_trace()

            train_size = int(math.ceil(len(train_set) / dataset_opt[ 'batch_size' ]))

            total_iters = int(opt[ 'train' ][ 'niter' ])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt[ 'dist' ]:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt[ 'name' ], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state[ 'epoch' ], resume_state[ 'iter' ]))

        start_epoch = resume_state[ 'epoch' ]
        current_step = resume_state[ 'iter' ]
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0

    best_psnr = 0.
    best_ssim = 0.
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt[ 'dist' ]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter = opt[ 'train' ][ 'warmup_iter' ])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt[ 'logger' ][ 'print_freq' ] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt[ 'use_tb_logger' ] and 'debug' not in opt[ 'name' ]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            #### validation
            if opt[ 'datasets' ].get('val', None) and current_step % opt[ 'train' ][ 'val_freq' ] == 0:
                if opt[ 'model' ] in [ 'sr', 'srgan' ] and rank <= 0:  # image restoration validation
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(os.path.basename(val_data[ 'LQ_path' ][ 0 ]))[ 0 ]
                        img_dir = os.path.join(opt[ 'path' ][ 'val_images' ], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals[ 'rlt' ])  # uint8
                        gt_img = util.tensor2img(visuals[ 'GT' ])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border([ sr_img, gt_img ], opt[ 'scale' ])
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        pbar.update('Test {}'.format(img_name))

                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    # tensorboard logger
                    if opt[ 'use_tb_logger' ] and 'debug' not in opt[ 'name' ]:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)
                else:  # video restoration validation
                    if opt[ 'dist' ]:
                        # multi-GPU testing
                        psnr_rlt = {}  # with border and center frames
                        if rank == 0:
                            pbar = util.ProgressBar(len(val_set))

                        random_index = random.randint(0, len(val_set) - 1)
                        for idx in range(rank, len(val_set), world_size):

                            if not (idx == random_index):
                                continue

                            val_data = val_set[ idx ]
                            val_data[ 'LQs' ].unsqueeze_(0)
                            val_data[ 'GT' ].unsqueeze_(0)
                            folder = val_data[ 'folder' ]
                            idx_d, max_idx = val_data[ 'idx' ].split('/')
                            idx_d, max_idx = int(idx_d), int(max_idx)
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[ folder ] = torch.zeros(max_idx, dtype = torch.float32,
                                                                 device = 'cuda')
                            # tmp = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                            model.feed_data(val_data)
                            model.test()
                            visuals = model.get_current_visuals()
                            sou_img = util.tensor2img(visuals[ 'LQ' ])
                            rlt_img = util.tensor2img(visuals[ 'rlt' ])  # uint8
                            rlt_imgs1 = util.tensor2img(visuals[ 'rlt_s1' ])  # uint8
                            gt_img = util.tensor2img(visuals[ 'GT' ])  # uint8

                            save_img = np.concatenate([ sou_img, rlt_img, rlt_imgs1, gt_img ], axis = 0)
                            im_path = os.path.join(opt[ 'path' ][ 'val_images' ], '%06d.png' % current_step)
                            cv2.imwrite(im_path, save_img.astype(np.uint8))

                            # calculate PSNR
                            psnr_rlt[ folder ][ idx_d ] = util.calculate_psnr(rlt_img, gt_img)
                        # # collect data
                        for _, v in psnr_rlt.items():
                            dist.reduce(v, 0)
                        dist.barrier()

                        if rank == 0:
                            psnr_rlt_avg = {}
                            psnr_total_avg = 0.
                            for k, v in psnr_rlt.items():
                                psnr_rlt_avg[ k ] = torch.mean(v).cpu().item()
                                psnr_total_avg += psnr_rlt_avg[ k ]
                            psnr_total_avg /= len(psnr_rlt)
                            log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                            for k, v in psnr_rlt_avg.items():
                                log_s += ' {}: {:.4e}'.format(k, v)
                            logger.info(log_s)
                            if opt[ 'use_tb_logger' ] and 'debug' not in opt[ 'name' ]:
                                tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                                for k, v in psnr_rlt_avg.items():
                                    tb_logger.add_scalar(k, v, current_step)
                    else:
                        # pbar = util.ProgressBar(len(val_loader))
                        psnr_rlt = {}  # with border and center frames
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        ssim_rlt = {}  # with border and center frames
                        ssim_rlt_avg = {}
                        ssim_total_avg = 0.
                        for val_data in val_loader:
                            folder = val_data[ 'folder' ][ 0 ]
                            idx_d = val_data[ 'idx' ]
                            # border = val_data['border'].item()
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[ folder ] = [ ]
                            if ssim_rlt.get(folder, None) is None:
                                ssim_rlt[ folder ] = [ ]

                            model.feed_data(val_data)
                            model.test()
                            visuals = model.get_current_visuals()
                            rlt_img = util.tensor2img(visuals[ 'rlt' ])  # uint8
                            gt_img = util.tensor2img(visuals[ 'GT' ])  # uint8

                            # calculate PSNR
                            psnr = util.calculate_psnr(rlt_img, gt_img)
                            psnr_rlt[ folder ].append(psnr)
                            ssim = util.calculate_ssim(rlt_img, gt_img)
                            ssim_rlt[ folder ].append(ssim)
                            # pbar.update('Test {} - {}'.format(folder, idx_d))
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[ k ] = sum(v) / len(v)
                            psnr_total_avg += psnr_rlt_avg[ k ]
                        for k, v in ssim_rlt.items():
                            ssim_rlt_avg[ k ] = sum(v) / len(v)
                            ssim_total_avg += ssim_rlt_avg[ k ]
                        psnr_total_avg /= len(psnr_rlt)
                        ssim_total_avg /= len(ssim_rlt)
                        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                        log_s1 = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                        new_best = False
                        if psnr_total_avg > best_psnr:
                            best_psnr = psnr_total_avg
                            log_s += ' best psnr : {:.4e}'.format(best_psnr)
                            new_best = True
                        if ssim_total_avg > best_ssim:
                            best_ssim = ssim_total_avg
                            log_s1 += ' best ssim : {:.4e}'.format(best_ssim)
                            new_best = True
                        # for k, v in psnr_rlt_avg.items():
                        #     log_s += ' {}: {:.4e}'.format(k, v)
                        logger.info(log_s)
                        logger.info(log_s1)
                        if opt[ 'use_tb_logger' ] and 'debug' not in opt[ 'name' ]:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)

                        if new_best and current_step % opt[ 'logger' ][ 'save_checkpoint_freq' ] == 0:
                            if rank <= 0:
                                logger.info('Saving models and training states.')
                                model.save(current_step)
                                model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        # tb_logger.close()


if __name__ == '__main__':
    main()
