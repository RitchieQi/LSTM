import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import net
from model import cnn
import datetime
import logging
import importlib
import shutil
import argparse
from pathlib import Path
from data.dataset import Emodata
from data.test_dataset import Emodata_raw
from tqdm import tqdm

inputsize = 17
outputsize = 5 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def test(model,loader,num_class=5):
    
    corr = 0
    total = 0
    classifier = model.eval()

    for j, (data, target) in tqdm(enumerate(loader), total=len(loader)):
        #if not args.use_cpu:
        data, target = data.cuda(), target.cuda()

        pred = classifier(data)

        top_pred = pred.argmax(1, keepdim=True)
        top_target = target.argmax(1, keepdim=True)
        #print(top_target)
        correct = top_pred.eq(top_target).sum()
        corr = corr + correct.float()
        total = total+target.shape[0]
    acc = corr / total
    print(corr,total)
        #print(acc)
    return acc
    #return instance_acc,class_acc

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
    data = Emodata_raw()
    train_dataset,test_dataset = torch.utils.data.random_split(data, [367, 157], generator=torch.Generator().manual_seed(42))
    
    #train_dataset = Testdata('train')
    #test_dataset = Testdata('test')

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8,drop_last=True)
    
    ''' Model Load '''
    # model = net
    # classifier = model.LSTM(inputsize,512,5,outputsize,0.5)
    model = cnn
    classifier = model.CNN(ninp=1)
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
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    highest_avg_acc = 0.0
    current_loss = 0.0
    ''' Train '''
    logger.info('start training...')
    for epoch in range(start_epoch, 500):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, 500))
        loss_ = []
        classifier = classifier.train()

        pbar = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9)
        for batch_id,(data,target) in pbar:
            optimizer.zero_grad()
            data,target = data.cuda().float(),target.cuda()

            pred = classifier(data)
            
            #print(pred.argmax(1).view(4,1).float())
            #print(target)
            loss = criterion(pred,target)
            #print(loss)
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
            avg_acc = test(classifier.eval(),testDataLoader)

            if (avg_acc >= highest_avg_acc):
                highest_avg_acc = avg_acc
                best_epoch = epoch + 1
            log_string('Test Average acc: %f, Best Average acc: %f ' % (avg_acc,highest_avg_acc))

            if (avg_acc >= highest_avg_acc):
                logger.info('Save model..')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)

                state = {
                    'epoch': best_epoch,
                    'acc': avg_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
    
    logger.info('end')

if __name__ == '__main__':
    main()
            

