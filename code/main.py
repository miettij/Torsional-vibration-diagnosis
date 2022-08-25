from arg_parser import parse_args, save_path_formatter

import torch
import os
import copy

# python3 main.py --arch WDCNN --dataset dummyset
# python3 main.py --arch WDCNN_deconv --dataset dummyset --lr 0.001 --epochs 10 --batch-size 128 --freeze True --deconv-iter 3 --eps 0.001 --stride 3 --bias True

if __name__ == '__main__':

    args = parse_args()

    if args.dataset == 'traditional':
        from datasets.dataset import Gear_Dataset
        print('All good !')

        input_channel_sets=[
        ['acc1'], # Gear housing Accelerometer 1
        ['acc3'], # Gear housing Accelerometer 2
        ['acc4'], # Gear housing Accelerometer 3
        ['acc2'], # Drive shaft Accelerometer 4
        ['acc1', 'acc3', 'acc4'], # Gear housing accelerometers
        ['acc1', 'acc2', 'acc3', 'acc4'], # All accelerometers
        ['torque1'], # Upper torque transducers T1 (drive shaft)
        ['torque2'], # Lower torque transducers T2 (load shaft)
        ['torque1', 'torque2'], # T All
        ['acc1', 'acc3', 'acc4', 'torque1', 'torque2'], # A123 & T all
        ['acc1', 'acc2', 'acc3', 'acc4', 'torque1', 'torque2'], # A all & T all
        ['enc1dd'], # E1 - angular acceleration
        ['enc2dd'], # E2 - angular acceleration
        ['enc3dd'], # E3 - angular acceleration
        ['enc4dd'], # E4 - angular acceleration
        ['enc5dd'], # E5 - angular acceleration
        ['enc1dd','enc2dd','enc3dd','enc4dd','enc5dd'], # E all
        ['enc1dd','enc2dd','enc3dd','enc4dd','enc5dd', 'torque1', 'torque2'], # E all & T all
        ['acc1', 'acc2', 'acc3', 'acc4', 'enc1dd','enc2dd','enc3dd','enc4dd','enc5dd'], # A all & E all
        ['acc1', 'acc2', 'acc3', 'enc1dd','enc2dd','enc3dd','enc4dd','enc5dd'], # A123 & E all
        ['acc1', 'acc2', 'acc3', 'acc4', 'torque1', 'torque2', 'enc1dd','enc2dd','enc3dd','enc4dd','enc5dd'], # A all & T all & E all
        ['acc1', 'acc2', 'acc3', 'torque1', 'torque2', 'enc1dd','enc2dd','enc3dd','enc4dd','enc5dd'] # A123 & T all & E all
        ]

        for input_channels in input_channel_sets:

            keys = input_channels
            input_channel_path = ''

            for input_channel in input_channels:

                if len(input_channel_path)<1:
                    input_channel_path += input_channel
                else:
                    input_channel_path += ','+input_channel

            from utils import get_filepaths

            args.message = input_channel_path
            log_dir=save_path_formatter(args)

            args.log_path=log_dir
            print(log_dir)
            if not os.path.exists(args.log_path):
                os.makedirs(args.log_path)

            if torch.cuda.is_available():
                root_dir = '../original/'
            else:
                root_dir = '../original.tmp/'

            all_filepaths = get_filepaths(root_dir)
            """ 'Traditional' <--> split all files according to:
            250RPM-1500RPM, and 0%-50% to train,
            250RPM-1500RPM, and 50%-75% to val
            250RPM-1500RPM, and 75%-100% to test."""
            train_set = Gear_Dataset(all_filepaths, args, datasplit_start = 0, datasplit_stop = 0.5, train_flag = True, input_channels = input_channels)
            val_set = Gear_Dataset(all_filepaths, args, datasplit_start = 0.5, datasplit_stop = 0.75, train_flag = False, input_channels = input_channels)
            test_set = Gear_Dataset(all_filepaths, args, datasplit_start = 0.75, datasplit_stop = 1, train_flag = False, input_channels = input_channels)

            if args.arch == 'WDCNN':
                from models.wdcnn import WDCNN
                model = WDCNN(input_channels = len(input_channels), n_classes = 10, bias = args.bias)

            elif args.arch == 'SRDCNN':
                from models.srdcnn import SRDCNN
                model = SRDCNN(in_channels = len(input_channels), n_classes = 10, args = args)

            elif args.arch == 'Ince':
                from models.ince import Ince
                model = Ince(in_channels = len(input_channels), n_classes = 10, args = args)

            elif args.arch == 'WDCNN_deconv':
                from models.wdcnn_deconv import WDCNN_deconv
                model = WDCNN_deconv(in_channels = len(input_channels), n_classes = 10, args = args)

            elif args.arch == 'Ince_deconv':
                from models.ince_deconv import Ince_deconv
                model = Ince_deconv(in_channels = len(input_channels), n_classes = 10, args = args)

            elif args.arch == 'SRDCNN_deconv':
                from models.srdcnn_deconv import SRDCNN_deconv
                model = SRDCNN_deconv(in_channels = len(input_channels), n_classes = 10, args = args)

            trainlogfile = os.path.join(args.log_path,'trainstats.txt')
            f = open(trainlogfile, 'w+')
            f.close()
            testlogfile = os.path.join(args.log_path,'teststats.txt')
            f = open(testlogfile, 'w+')
            f.close()

            from optalgos.train import train
            from optalgos.train import test
            untrained_weights = copy.deepcopy(model.state_dict())
            for i in range(5):
                model.load_state_dict(untrained_weights)
                trained_model = train(model, train_set, val_set, args, trainlogfile, args.log_path)
                test(trained_model, test_set, args, testlogfile)
