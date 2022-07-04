import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime
import logging
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def test(model,loader):
    classifier = model.eval()

    for j, (data, target) in tqdm(enumerate(loader), total=len(loader)):
        #if not args.use_cpu:
        data, target = data.cuda(), target.cuda()

        pred = classifier(data)
        
        loss = F.cross_entropy(pred,data)
        return loss

def main():
    def log_string(str):
        logger.info(str)
        print(str)

    ''' Create Dir '''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    ''' Log '''
    #args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'model/net'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    #log_string(args)
    
    ''' Data Load '''
    # train_dataset = 
    # test_dataset =
    # trainDataLoader = 
    # testDataLoader = 
    
    ''' Model Load '''
    model = importlib.import_module('model/net')
    classifier = model.LSTM(17,512,5,0.5)
    criterion = model.get_loss()

    classifier = classifier.cuda()
    criterion = criterion.cuda()
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    lowest_avg_loss = 1
    current_loss = 0.0
    ''' Train '''
    logger.info('start training...')
    for epoch in range(start_epoch, 50):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, 50))
    

