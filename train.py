import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import net
import datetime
import logging
import importlib
import shutil
import argparse
from pathlib import Path
from data.dataset import Emodata
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
    #exp_dir = exp_dir.joinpath(timestr)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    ''' Log '''
    #args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'net'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    #log_string(args)
    
    ''' Data Load '''
    data = Emodata()

    train_dataset,test_dataset = torch.utils.data.random_split(data, [294, 126], generator=torch.Generator().manual_seed(42))
    
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8,drop_last=True)
    
    ''' Model Load '''
    model = net
    classifier = model.LSTM(17,512,5,5,0.5)
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
        loss_ = []
        classifier = classifier.train()

        pbar = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9)
        for batch_id,(data,target) in pbar:
            optimizer.zero_grad()
            data,target = data.cuda(),target.cuda()

            pred = classifier(data)

            loss = criterion(pred,target)

            current_loss = loss.item()
            loss_.append(current_loss)
            loss.backward()
            optimizer.step()
            global_step +=1
            pbar.set_description('Loss:%f'% current_loss)
        
        scheduler.step()
        train_mean_loss = np.mean(loss_)
        log_string('Train MSE: %f' % train_mean_loss)

        with torch.no_grad():
            avg_loss = test(classifier.eval(),testDataLoader)

            if (avg_loss <= lowest_avg_loss):
                lowest_avg_loss = avg_loss
                best_epoch = epoch + 1
            log_string('Test Average MSE: %f, Best Average MSE: %f ' % (avg_mse,lowest_avg_mse))

            if (avg_loss <= lowest_avg_loss):
                logger.info('Save model..')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)

                state = {
                    'epoch': best_epoch,
                    'avg_mse': avg_mse,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
    
    logger.info('end')

if __name__ == '__main__':
    main()
            

