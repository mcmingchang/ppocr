from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import torch
import torch.distributed as dist
import torch.nn as nn

from ocr.data import build_dataloader
from ocr.modeling.architectures import build_model
from ocr.losses import build_loss
from ocr.optimizer import build_optimizer
from ocr.postprocess import build_post_process
from ocr.metrics import build_metric
from ocr.utils.save_load import load_model
from ocr.utils.utility import set_seed
from ocr.modeling.architectures import apply_to_static
import tools.program as program
dist.get_world_size()


def main(config, device, logger, vdl_writer):
    # init dist environment
    # if config['Global']['distributed']:
    #     dist.init_parallel_env()

    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head'][
                        'name'] == 'MultiHead':  # for multi head
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    # update SARLoss params
                    assert list(config['Loss']['loss_config_list'][-1].keys())[
                        0] == 'DistillationSARLoss'
                    config['Loss']['loss_config_list'][-1][
                        'DistillationSARLoss']['ignore_index'] = char_num + 1
                    out_channels_list = {}
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # for multi head
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            # update SARLoss params
            assert list(config['Loss']['loss_config_list'][1].keys())[
                0] == 'SARLoss'
            if config['Loss']['loss_config_list'][1]['SARLoss'] is None:
                config['Loss']['loss_config_list'][1]['SARLoss'] = {
                    'ignore_index': char_num + 1
                }
            else:
                config['Loss']['loss_config_list'][1]['SARLoss'][
                    'ignore_index'] = char_num + 1
            out_channels_list = {}
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

        if config['PostProcess']['name'] == 'SARLabelDecode':  # for SAR model
            config['Loss']['ignore_index'] = char_num - 1

    model = build_model(config['Architecture'])
    if config['Global']['distributed']:
        model = nn.DataParallel(model)

    model = apply_to_static(model, config, logger)

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        model=model)

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = load_model(config, model, optimizer,
                                     config['Architecture']["model_type"])
    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))

    use_amp = config["Global"].get("use_amp", False)
    if use_amp:
        scale_loss = config["Global"].get("scale_loss", 1.0)
        # use_dynamic_loss_scaling = config["Global"].get(
        #     "use_dynamic_loss_scaling", False)
        scaler = torch.cuda.amp.GradScaler(init_scale=scale_loss)
    else:
        scaler = None
    torch.cuda.is_available()
    # start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer, scaler)


def test_reader(config, device, logger):
    loader = build_dataloader(config, 'Train', device, logger)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    set_seed(seed)
    main(config, device, logger, vdl_writer)
    # test_reader(config, device, logger)
