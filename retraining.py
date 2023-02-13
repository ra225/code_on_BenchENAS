
# coding=utf-8
from __future__ import print_function
import sys
import os

import importlib, argparse


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("-gpu_id", "--gpu_id", help="GPU ID", type=str)
    _parser.add_argument("-file_id", "--file_id", help="file id", type=str)

    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu_id
    from datetime import datetime
    import multiprocessing
    from algs.performance_pred.utils import TrainConfig
    from ast import literal_eval
    from comm.registry import Registry
    from train.utils import OptimizerConfig, LRConfig

    from compute.redis import RedisLog
    from compute.pid_manager import PIDManager
    import torch
    import traceback
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.backends.cudnn as cudnn




class TrainModel(object):
    def __init__(self, file_id, logger):

        # module_name = 'scripts.%s'%(file_name)
        module_name = file_id
        o = file_id.find('.')
        if o != -1:
            module_name = file_id[:o]
        if module_name in sys.modules.keys():
            self.log_record('Module:%s has been loaded, delete it' % (module_name))
            del sys.modules[module_name]
            _module = importlib.import_module('.', module_name)
        else:
            _module = importlib.import_module('.', module_name)

        net = _module.EvoCNNModel()

        checkpoint = torch.load("/home/n504/checkpoints/pso_MNIST/" + file_id + ".pt")
        net.load_state_dict(checkpoint['model'])

        cudnn.benchmark = True
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_error = 1.0
        best_loss = checkpoint['loss']
     #   best_loss = 1.097477
        self.net = net

        TrainConfig.ConfigTrainModel(self)
        # initialize optimizer
        o = OptimizerConfig()
        opt_cls = Registry.OptimizerRegistry.query(o.read_ini_file('_name'))
        opt_params = {k: (v) for k, v in o.read_ini_file_all().items() if not k.startswith('_')}
        l = LRConfig()
        lr_cls = Registry.LRRegistry.query(l.read_ini_file('lr_strategy'))
        lr_params = {k: (v) for k, v in l.read_ini_file_all().items() if not k.startswith('_')}
        lr_params['lr'] = float(lr_params['lr'])
        opt_params['lr'] = float(lr_params['lr'])
        self.opt_params = opt_params
        self.opt_cls = opt_cls
        self.opt_params['total_epoch'] = self.nepochs
        self.lr_params = lr_params
        self.lr_cls = lr_cls
        # after the initialization

        self.criterion = criterion
        self.best_error = best_error
        self.best_loss = best_loss

        self.file_id = file_id
        self.logger = logger
        self.counter = 0

    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))

    def get_optimizer(self, epoch):
        # get optimizer
        self.opt_params['current_epoch'] = epoch
        opt_cls_ins = self.opt_cls(**self.opt_params)
        optimizer = opt_cls_ins.get_optimizer(filter(lambda p: p.requires_grad, self.net.parameters()))
        return optimizer

    def get_learning_rate(self, epoch):
        self.lr_params['optimizer'] = self.get_optimizer(epoch)
        self.lr_params['current_epoch'] = epoch
        lr_cls_ins = self.lr_cls(**self.lr_params)
        learning_rate = lr_cls_ins.get_learning_rate()
        return learning_rate

    def train(self, epoch):
        #if self.counter / 8 == 1:
        #    self.counter = 0
        #    self.lr = self.lr * 0.5
        self.net.train()
        optimizer = self.get_optimizer(epoch)
        #import torch.optim as optim
        #optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum().item()
        self.log_record('Train-Epoch:%3d,  Loss: %.5f, Acc:%.5f' % (epoch + 1, running_loss / total, (correct / total)))

    def main(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum().item()
        if 1 - correct / total < self.best_error:
            self.best_error = 1 - correct / total
        if test_loss / total < self.best_loss:
            self.log_record('Validation loss decreased ({%.6f} --> {%.6f}).  Saving model ...'%(self.best_loss, test_loss / total))
            self.best_loss = test_loss / total
            torch.save({'model': self.net.state_dict(), 'loss': self.best_loss}, "/home/n504/checkpoints/pso_MNIST/" + self.file_id + ".pt.fullepoch")
            self.counter = 0
        else:
            self.counter += 1
        self.log_record('Valid-Epoch:%3d, Loss:%.5f, Acc:%.5f, lr:%.5f' % (epoch+1, test_loss / total, correct / total, self.lr))

    def process(self):
        total_epoch = self.nepochs
        full_epoch = self.fullepochs
        scheduler = self.get_learning_rate(total_epoch)
        self.log_record('file_id:%s, parameters:%d' % (self.file_id, sum([p.numel() for p in self.net.parameters()])))
        for p in range(full_epoch):
            scheduler.step()
            self.train(p)
            self.main(p)
        return self.best_error, self.best_loss


class RunModel(object):
    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))

    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        logger = RedisLog(os.path.basename(file_id) + '.txt')
        best_error = 100.0
        best_loss = 100.0
        try:
            m = TrainModel(file_id, logger)
            m.log_record(
                'Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
            best_error, best_loss = m.process()
        except BaseException as e:
            msg = traceback.format_exc()
            print('Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            print('%s' % (msg))
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Exception occurs:%s' % (msg)
            logger.info('[%s]-%s' % (dt, _str))
        finally:
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Finished-Error:%.5f, Finished-Loss:%.5f' % (best_error, best_loss)
            logger.info('[%s]-%s' % (dt, _str))

           # logger.write_file('RESULTS', 'results.txt', '%s=%.5f=%.5f\n' % (file_id, best_error, best_loss))
           # _str = '%s;%.5f;%.5f\n' % (uuid, best_error, best_loss)
           # logger.write_file('CACHE', 'cache.txt', _str)


if __name__ == "__main__":

    RunModel().do_work(_args.gpu_id, _args.file_id)
