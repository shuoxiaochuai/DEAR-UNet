from nets.CAUNet import CAUNet
from main_YWJ import train_one_epoch
import main_YWJ
main_YWJ.tensorboard = False
import torch.nn as nn
from DataSet import load_train_val_data
from tensorboardX import SummaryWriter
import os
import time
import torch
from Losses import *

if __name__ == '__main__':
    ## PARAMS
    model_type = 'CAUNet'
    data_path = '/root/rivermind-data/Segmentation_dataset3' #path where dataset is stored
    save_path = '/root/rivermind-data/results_k/MA/CAUNet/cv_models/fold5_best.pth.tar'
    #save_path = '/root/autodl-tmp/results_training/MA/CAUNet/save/models/Test_session_01.14 15h5929/best_model.LBTW_WeightedDiceBCE.CAUNet.pth.tar'
    #output_dir = '/root/autodl-tmp/results_training/EX/CAUNet/save'
    #tensorboard_folder = os.path.join(output_dir, 'tensorboard_logs')

    session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M%S')
    tasks = ['MA']

    # Tensorboard logs
    #log_dir = os.path.join(tensorboard_folder, session_name)
    #if not os.path.isdir(log_dir):
    #    os.makedirs(log_dir)
    #writer = SummaryWriter(log_dir)

    # Loading a model
    checkpoint = torch.load(save_path, map_location='cuda', weights_only=False)




    if model_type == 'CAUNet':
        model = CAUNet()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        # Choose loss function
        # criterion = LBTW_Loss(nn.MSELoss())
        # criterion = LBTW_Loss(nn.BCELoss())
        #criterion = LBTW_Loss(WeightedDiceBCE())
        base_loss = AdaptiveHybridLoss(
            warmup_epochs=5,  # 前5轮用WeightedDiceBCE
            transition_epochs=10,  # 10轮过渡
            dice_weight=1.0,
            bce_weight=0.5,
            focal_weight=0.4,
            tversky_weight=0.6
        )
        criterion = LBTW_Loss(base_loss)
        # criterion = dice_loss
        # criterion = mean_dice_loss
        # criterion = nn.BCELoss()


    # load weights into model
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')

    # Load Datasets
    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path, batch_size=1, green=True)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0)  # Choose optimize
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=7)

    with torch.no_grad():
        model.eval()
        print('Running Evaluation...')
        lbtw_algorithm = LBTW_algorithm()
        val_loss, val_aupr, *_ = train_one_epoch(val_loader, model, criterion,
                                             optimizer, None, 0, None,
                                             model_type, lbtw_algorithm,batch_size=1)
        print('AUPR evaluated on validation set:', val_aupr)
