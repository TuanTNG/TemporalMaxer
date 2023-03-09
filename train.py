# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

from libs.utils.logger import Logger


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    # if not os.path.exists(cfg['output_folder']):
    #     os.mkdir(cfg['output_folder'])
    # cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    # if len(args.output) == 0:
    #     ts = datetime.datetime.fromtimestamp(int(time.time()))
    #     ckpt_folder = os.path.join(
    #         cfg['output_folder'], cfg_filename + '_' + str(ts))
    # else:
    #     ckpt_folder = os.path.join(
    #         cfg['output_folder'], cfg_filename + '_' + str(args.output))
    ckpt_folder = args.save_ckpt_dir
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    # create val loader
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        raise NotImplementedError("Not support resuming yet")
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    best_mAP = 0
    max_decrease_eps = 3
    decrease_ep_cnt = 0
    best_ep = 0
    logger = Logger("training_mAP", os.path.join(ckpt_folder, 'logs'))

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # save ckpt once in a while
        save_states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        save_states['state_dict_ema'] = model_ema.module.state_dict()
        save_checkpoint(
            save_states,
            False,
            file_folder=ckpt_folder,
            file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
        )

        if epoch >= (max_epochs-5)//2:
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds'])
            mAP = valid_one_epoch(
                val_loader,
                model_ema.module,
                -1,
                evaluator=det_eval,
                output_file=None,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=None,
                print_freq=args.print_freq)
            logger.info(f'epoch: {epoch}, mAP: {mAP}')
            if mAP > best_mAP:
                best_mAP = mAP
                best_ep = epoch + 1

                print('saving best model')

                # save ckpt once in a while
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mAP': mAP
                }

                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='bestmodel.pth.tar'
                )
                decrease_ep_cnt -= 1
                decrease_ep_cnt = max(0, decrease_ep_cnt)
            else:
                decrease_ep_cnt += 1
            
            if decrease_ep_cnt >= max_decrease_eps:
                break
    
    print(f'best epoch {best_ep} with {best_mAP} mAP')

    # wrap up
    tb_writer.close()
    print("All done!")
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based TemporalMaxer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--save_ckpt_dir', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
