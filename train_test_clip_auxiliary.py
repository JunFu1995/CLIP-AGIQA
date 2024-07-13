from cgi import test
import os
from scipy import stats
import random
import numpy as np
from datasets.dataloader import *
import torch.nn as nn
import sys
import yaml
import models 

class IQAManager(object):
    def __init__(self, options, path, percentage, rd):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        self.percentage = percentage
        self.round = rd

        # Network.
        self._net, self.cfg = models.buildModel(options['model'], options['cfgname'])

        # Criterion.
        self._criterion = torch.nn.MSELoss().cuda()

        # Solver.
        self._solver = torch.optim.Adam([{'params':self._net.prompt_learner_aux.parameters()},{'params':self._net.prompt_learner_tar.parameters()}], self._options['base_lr']) 

        dn = self._options['dataset']
        self._train_loader = DataLoader(dn, self._path[dn], self._options['train_index'], \
                                        batch_size=self._options['batch_size'], istrain=True, patch_num=1).get_data()

        self._test_loader = DataLoader(dn, self._path[dn], self._options['test_index'], \
                                       istrain=False).get_data()

        self.unfold = nn.Unfold(kernel_size=(224, 224), stride=64)

        savePath = os.path.join('./save/', 'aux', '%s'%options['dataset'], 'round%d' % options['round'],'%s'%options['model'], 'use_aux_%d_textdepth_%d_visiondepth_%d_lamda_%d'%
        (self.cfg['TRAINER']['Ours']['use_aux'],self.cfg['TRAINER']['Ours']['text_depth'],self.cfg['TRAINER']['Ours']['vision_depth'],self.cfg['TRAINER']['Ours']['lamda']*100)
        )
        if not os.path.isdir(savePath):
            os.makedirs(savePath)

        options['savePath'] = savePath

        self.savePath_aux = os.path.join(options['savePath'], '%s_%s_%d_aux_best.pth'%(options['model'], options['dataset'], options['n_ctx']))
        self.savePath_tar = os.path.join(options['savePath'], '%s_%s_%d_tar_best.pth'%(options['model'], options['dataset'], options['n_ctx']))
        self.testDataPath = os.path.join(options['savePath'], '%s_%s_best'%(options['model'], options['dataset']))

    def train(self):
        """Train the network."""
        print('Training.')
        best_srcc = 0.0
        best_plcc = 0.0
        best_krcc = 0.0
        best_epoch = None

        not_continue_count = 0
        print('Epoch\tTrain loss\tAuxiliary loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC')
        for t in range(self._options['epochs']):
            epoch_loss = []
            pscores = []
            tscores = []
            epoch_loss1 = []
            for X, y, z, _, y1 in self._train_loader:
                # Data.
                X = X.cuda()  
                y = y.cuda().float() 
                z = z.cuda()
                y1 = y1.cuda().float()

                score, aligend_score = self._net(X, z)

                # Clear the existing gradients.
                self._solver.zero_grad()
                loss = self._criterion(score, y.view(len(score), 1).detach())
                if self.cfg['TRAINER']['Ours']['use_aux']:
                    aux_loss = self._criterion(aligend_score, y1.view(len(score), 1).detach())
                else:
                    aux_loss = torch.zeros(1).to(loss.device)

                epoch_loss.append(loss.item())
                epoch_loss1.append(aux_loss.item())
                # Prediction.
                pscores = pscores + score.cpu().tolist()
                tscores = tscores + y.cpu().tolist()

                (loss+self.cfg['TRAINER']['Ours']['lamda']*aux_loss).backward()
                self._solver.step()
            train_srcc, _ = stats.spearmanr(pscores, tscores)
            with torch.no_grad():
                test_srcc, test_plcc, test_data, test_krcc = self.test(self._test_loader)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                best_krcc = test_krcc
                best_epoch = t + 1
                # save model 
                torch.save(self._net.prompt_learner_aux.state_dict(), self.savePath_aux)
                torch.save(self._net.prompt_learner_tar.state_dict(), self.savePath_tar)
                np.save(self.testDataPath, test_data)
                not_continue_count = 0
            else:
                not_continue_count += 1
            
            print('%d\t\t%4.3f\t\t%4.3f\t\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), sum(epoch_loss1) / len(epoch_loss1) , train_srcc, test_srcc, test_plcc, test_krcc))
            #early stop
            if not_continue_count >= 15:
                break 
        print('Best at epoch %d, test srcc %.4f, test plcc %.4f' % (best_epoch, best_srcc, best_plcc))
        return best_srcc, best_plcc, best_krcc

    def test(self, data_loader):
        self._net.train(False)
        num_total = 0
        pscores = []
        tscores = []
        batch_size = 8
        test_data = {}
        for X, y, z,  path , _ in data_loader:
            # Data.
            X = X.cuda()  
            y = y.cuda()  
            z = z.cuda()
            X_sub = self.unfold(X).view(1, X.shape[1], 224, 224, -1)[0]
            X_sub = X_sub.permute(3,0,1,2)

            img = torch.split(X_sub, batch_size, dim=0)
            pred_s = []
            for i in img:
                pred , _= self._net(i, z) 
                pred_s += pred.detach().cpu().tolist()
            score = np.mean(pred_s)
            pscores = pscores + [score]
            tscores = tscores + y.cpu().tolist()
            test_data[path] = [score, y.cpu().tolist()[0]]
            num_total += y.size(0)
        test_srcc, _ = stats.spearmanr(pscores, tscores)
        test_plcc, _ = stats.pearsonr(pscores, tscores)
        test_krcc, _ = stats.kendalltau(pscores, tscores)
        self._net.train(True)  # Set the model to training phase
        return test_srcc, test_plcc, test_data, test_krcc

class flushfile:
    #https: // stackoverflow.com / questions / 230751 / how - can - i - flush - the - output - of - the - print - function
  def __init__(self, f):
    self.f = f
  def write(self, x):
    self.f.write(x)
    self.f.flush()

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='test clip for iqa tasks.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-4,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=64, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset', dest='dataset', type=str, default='AGIQA3k',
                        help='dataset: live|csiq|tid2013|livec|mlive')
    parser.add_argument('--model', dest='model', type=str, default='CNNIQA',
                        help='dataset: live|csiq|tid2013|livec|mlive')   
    parser.add_argument('--n_ctx', dest='n_ctx', type=int, default=8,
                        help='dataset: live|csiq|tid2013|livec|mlive')   
    parser.add_argument('--gpuid', type=str, default='0', help='GPU ID')
    parser.add_argument('--percentage', type=float, default=0.8, help='training portion')
    parser.add_argument('--cfgname', dest='cfgname', type=str, default='cfg_8',
                        help='dataset: live|csiq|tid2013|livec|mlive')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    with open('./config/%s.yaml'%args.cfgname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    ss = '%s_use_aux_%d_textdepth_%d_visiondepth_%d_lamda_%.2f'%\
    (args.model, cfg['TRAINER']['Ours']['use_aux'],cfg['TRAINER']['Ours']['text_depth'],cfg['TRAINER']['Ours']['vision_depth'],cfg['TRAINER']['Ours']['lamda'])
    f = open(os.path.join('./log/', 'aux_%s_%s.log'%(args.dataset,ss)), 'w')
    sys.stdout = flushfile(f)

    ## print config 
    print(ss)
    seed = 10 
    print("Random Seed: ", seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs)
    # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    savePath = os.path.join('./save/', args.model)
    if not os.path.isdir(savePath):
        os.makedirs(savePath)

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset': args.dataset,
        'fc': [],
        'train_index': [],
        'test_index': [],
        'model': args.model,
        'savePath': savePath, 
        'n_ctx': args.n_ctx,
        'cfgname':args.cfgname
 
    }


    path = {
        'AGIQA3k': '/home/fujun/datasets/iqa/AGIQA-3K',
        #'AGIQA1k': '/home/fujun/datasets/iqa/AGIQA-1K',
        'AGIQA2023': '/home/fujun/datasets/iqa/AIGC2023/DATA/'
    }

    if options['dataset'] == 'AGIQA3k':
        index = list(range(0, 300))
    elif options['dataset'] == 'AGIQA1k':
        index = list(range(0, 360))
    elif options['dataset'] == 'AGIQA2023':
        index = list(range(0, 100))

    roudNum = 10
    srcc_all = np.zeros((1, roudNum), dtype=np.float64)
    plcc_all = np.zeros((1, roudNum), dtype=np.float64)
    krcc_all = np.zeros((1, roudNum), dtype=np.float64)
    for i in range(0, roudNum): # 
        print("====================round %d=====================" % i )
        # randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(args.percentage * len(index))]
        test_index = index[round(args.percentage * len(index)):len(index)]

        options['train_index'] = train_index
        options['test_index'] = test_index
        options['round'] = i

        manager = IQAManager(options, path, args.percentage, i)
        best_srcc, best_plcc, best_krcc = manager.train()
        srcc_all[0][i] = best_srcc
        plcc_all[0][i] = best_plcc
        krcc_all[0][i] = best_krcc

    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)
    krcc_mean = np.mean(krcc_all)
    
    print('srcc', srcc_all)
    print('plcc', plcc_all)
    print('krcc', krcc_all)

    print('average mean srcc:%4.4f, plcc:%4.4f, krcc:%4.4f' % (srcc_mean, plcc_mean, krcc_mean))
    print('average std srcc:%4.4f, plcc:%4.4f, krcc:%4.4f' % (srcc_all.std(), plcc_all.std(), krcc_all.std()))

if __name__ == '__main__':
    main()
