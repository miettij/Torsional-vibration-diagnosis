import argparse
from collections import OrderedDict
import distutils.util
import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr-decay', default =10, type = int)
    parser.add_argument('-b','--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-p','--patience',default = 10, type=int, help = 'patience for early_stopping')
    parser.add_argument('--epochs', default=70, type=int, help='training epochs')

    parser.add_argument('-a','--arch', default='vgg11', help='architecture')
    parser.add_argument('--tw-stride', default = 32, type=int)
    parser.add_argument('--tw-len', default = 2048, type=int)
    parser.add_argument('--dataset', default='traditional', help='dataset - traditional is currently the only option')


    parser.add_argument('--bias', default=True,type=distutils.util.strtobool, help='use bias term in deconv')

    args = parser.parse_args()

    return args


def save_path_formatter(args):
    args_dict = vars(args)
    data_folder_name = args_dict['dataset']
    folder_string = []

    key_map = OrderedDict()
    key_map['arch'] =''
    key_map['batch_size']='bs'

    key_map['lr']='lr'
    key_map['epochs'] = 'epoch'
    key_map['bias'] = 'bias'


    for key, key2 in key_map.items():
        value = args_dict[key]
        if key2 is not '':
            folder_string.append('{}.{}'.format(key2, value))
        else:
            folder_string.append('{}'.format(value))

    save_path = ','.join(folder_string)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M")
    return os.path.join('./logs2',data_folder_name,args.message,save_path,timestamp).replace("\\","/")
