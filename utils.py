import os
#import importlib
import logging
import sys
import yaml
import argparse
import torch



loggers = {}

def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger



def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


  
    
class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count    
        
        
        
def load_config():
    parser = argparse.ArgumentParser(description='UNet3D for fluid-gas segmentation')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    return config


def get_device():
    if torch.cuda.is_available(): # is there any visible GPU ?
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    return device


def net_to_device(net, logger):  
    device = get_device()      
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net_parallel = torch.nn.DataParallel(net) 
            logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
            return net_parallel.to(device)
        else:
            logger.info('Using just one GPU')
            return net.to(device)
    else:
        logger.warning('CUDA not available, using CPU !')
        return net.to(device)
    

def maybe_save_checkpoint(epoch, config, net, optimizer, loss_func, logger):
    if epoch >= config['first_epoch_to_save_checkpoints'] \
       and config['save_checkpoint_every_epochs'] > 0 \
       and (epoch == config['first_epoch_to_save_checkpoints']
            or epoch % config['save_checkpoint_every_epochs'] == 0) :
        fname = os.path.join('checkpoints', '{}_epoch_{}.pt'\
                             .format(config['experiment_id'], epoch))
        torch.save({'config': config,
                    'epoch': epoch,
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_func': loss_func }, fname)
        logger.info('saved checkpoint to {}'.format(fname))        
        
        
def load_checkpoint(experiment_id, epoch, logger):
    fname = os.path.join('checkpoints', '{}_epoch_{}.pt'\
                         .format(experiment_id, epoch))
    logger.info('loaded checkpoint {}'.format(fname))
    checkpoint = torch.load(fname)
    return checkpoint['config'], checkpoint['model_state_dict'], \
        checkpoint['optimizer_state_dict'], checkpoint['loss_func']
        
