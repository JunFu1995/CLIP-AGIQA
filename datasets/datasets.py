import sys
from tkinter import image_names
sys.path.append("..")

import os 
import torch.utils.data as data
import pandas as pd 
import numpy as np 

import xlrd 
import matplotlib.pyplot as plt 

import  clip 
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from scipy import stats
import scipy.io as sio

from PIL import Image
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class AGIQA3k(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        csv_file = os.path.join(root, 'data.csv')
        df = pd.read_csv(csv_file)

        imgnames = df['name'].tolist()
        labels = np.array(df['mos_quality']).astype(np.float32)
        align_labels = np.array(df['mos_align']).astype(np.float32)


        prompts = df['prompt'].tolist() 

        # here we make sure the image with the same object label falls into the same set.  
        object_dict = {} # 300 in total 
        idx = 0 
        for p in prompts:
            if object_dict.get(p) is None:
                object_dict[p] = [idx]
            else:
                object_dict[p] += [idx]     
            idx += 1

        keys = list(object_dict.keys())
        keys.sort() 
        choose_index = []
        for idx in index:
            choose_index += object_dict[keys[idx]]

        # make datasets 
        na = []
        nb = []

        sample = []
        for idx in choose_index:
            p = prompts[idx]
            # print(p) 
            p = p.split(',')
            p = [item.strip() for item in p]

            #newp = '%s.' % p[0] #The prompt of this image is that 
            if len(p) == 1:
                newp = 'The prompt of this image is that %s.' % p[0]
            elif len(p) == 2:
                if 'style' in p[1]:
                    newp = 'The prompt of this image is that %s, and its style is %s.' % (p[0], p[1])
                else:
                    newp = 'The prompt of this image is that %s, and its detail is %s.' % (p[0], p[1])                    
            else:
                newp = 'The prompt of this image is that %s, ' % p[0]
                news = 'its detail is '

                flag = 0
                for i in range(1, len(p)):
                    if 'style'  in p[i]:
                        flag = 1 # always in the end 
                        break 
                    #     news += '%s, ' % p[i]
                    # else:
                    #     news += 'and its style is %s.' % p[i]
                if flag:
                    news += ', '.join(p[1:-1])
                    news += ', and its style is %s.' % p[-1]
                else:
                    news += ', '.join(p[1:-1])
                    news += '.'
                newp += news
            # new_p = newp.rstrip(' ')
            # if newp[-1] == ',':
            #     newp = newp[:-1] + '.'
            #print(newp, newp[-1]) 
            new_p = 'This is a photo aligned with the prompt.'#newp 
            # for pp in p:
            #     pp = pp.split(' ')
            #     pp = [item.replace('-', '') for item in pp if item]
            #     pp = ' '.join(pp)
            #     new_p += [pp]
            # new_p = ' '.join(new_p)
            # print(new_p)
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'img', imgnames[idx]), labels[idx], align_labels[idx], new_p))
                na.append(labels[idx])
                nb.append(align_labels[idx])
        self.samples = sample
        self.transform = transform
        # plt.scatter(na, nb)
        # srcc, _ = stats.spearmanr(na, nb)
        # plt.title("%.2f"%srcc)
        # plt.show()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, align, prompt = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        #print(prompt)
        word_idx = clip.tokenize(prompt)[0]
        #word_idx = word_idx[0].numpy().tolist() # [77]
        # tokenize prompt 
        return sample, target/5.0, word_idx, path , align / 5.0 # for 0 - 1

    def __len__(self):
        length = len(self.samples)
        return length

class AGIQA1k(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        xlsx_file = os.path.join(root, 'AIGC_MOS_Zscore.xlsx')
        df = pd.read_excel(xlsx_file)

        # print(df)
        imgnames = df['Image'].tolist()
        labels = np.array(df['MOS']).astype(np.float32)
        prompts = df['Prompt'].tolist() 

        # clean prompts 
        object_dict = {}
        new_prompts = []
        idx = 0 
        for p in prompts:
            # print(p)
            p = p.replace(',', ' ')
            p = p.rstrip() 
            p = p.split(' ')
            p = [item.replace('-', '') for item in p if item]
            object_pt = ' '.join(p[:-2])
            if object_dict.get(object_pt) is None:
                object_dict[object_pt] = [idx]
            else:
                object_dict[object_pt] += [idx]    # 360 in total             
            p = ' '.join(p)
            new_prompts.append(p)
            idx += 1

        # here we make sure the image with the same object label falls into the same set.    
        keys = list(object_dict.keys())
        keys.sort() 
        choose_index = []
        for idx in index:
            choose_index += object_dict[keys[idx]]
        prompts = new_prompts
        #print(prompts)
        assert len(labels) == len(imgnames) == len(prompts)

        sample = []
        for idx in choose_index:
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'img', imgnames[idx]), labels[idx], prompts[idx]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, prompt = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        word_idx = clip.tokenize(prompt)[0]
        # word_idx = word_idx[0].numpy().tolist() # [77]
        # tokenize prompt 
        return sample, target / 5.0 , word_idx, path #word_idx # for 0 - 1

    def __len__(self):
        length = len(self.samples)
        return length


class AGIQA2023(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        fp = [os.path.join(self.root, i) for i in ['mosz1.mat', 'mosz2.mat', 'mosz3.mat']] # quality, Authenticity, Correspondence
        data = [sio.loadmat(i)['MOSz'][:,0].tolist() for i in fp]

        imgdir = ['Controlnet', 'DALLE', 'Glide', 'Lafite', 'stable-diffusion', 'Unidiffuser']
        imgpath = {}
        step = 400 
        for ind, d in enumerate(imgdir):
            imgpath[d] = {} 
            for i in range(100):
                imgpath[d][i] = [step * ind + i * 4 + j for j in range(4)] 

        sample = []
        for d in imgdir:
            for idx in index:
                for n in imgpath[d][idx]:
                    sample.append((os.path.join(root, 'allimg', '%d.png' % n), data[0][n], data[1][n], data[2][n], d))

        # for d in data:
        #     print(len(d))
        na, nb = data[0], data[2]
        plt.scatter(na, nb)
        srcc, _ = stats.spearmanr(na, nb)
        plt.title("SRCC = %.2f"%srcc)
        plt.xlabel("Perceptual Quality")
        plt.ylabel("Alignment Quality")
        plt.savefig("AIGCIQA.pdf", bbox_inches='tight')
        #plt.show()        
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, authetic, correspond, cat = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target / 100.0 , sample, path , correspond / 100.0

    def __len__(self):
        length = len(self.samples)
        return length

if __name__ == '__main__':
    # root = '/home/fujun/datasets/iqa/AGIQA-3K/'
    # csv_file = os.path.join(root, 'data.csv')
    # df = pd.read_csv(csv_file)

    # imgnames = df['name'].tolist()
    # with open('AGIQA3k.csv', 'w') as f:
    #     for n in imgnames:
    #         p = os.path.join(root, 'img', n)
    #         f.write("%s\n"%p)

    # root = '/home/fujun/datasets/iqa/AIGC2023/DATA/'
    # imgdir = ['Controlnet', 'DALLE', 'Glide', 'Lafite', 'stable-diffusion', 'Unidiffuser']
    # imgpath = {}
    # step = 400 
    # for ind, d in enumerate(imgdir):
    #     imgpath[d] = {} 
    #     for i in range(100):
    #         imgpath[d][i] = [step * ind + i * 4 + j for j in range(4)] 

    # with open('AGIQA2023.csv', 'w') as f:
    #     for d in imgdir:
    #         for idx in range(0, 100):
    #             for n in imgpath[d][idx]:
    #                 p = os.path.join(root, 'allimg', '%d.png' % n)
    #                 f.write("%s\n"%p)
    # root, index, transform, patch_num = '/home/fujun/datasets/iqa/AGIQA-3K', [1], None, 1
    # ds = AGIQA3k(root, index, transform, patch_num)

    # for i in ds:
    #     print(i)
    # root, index, transform, patch_num = '/home/fujun/datasets/iqa/AGIQA-1K', [1], None, 1
    # ds = AGIQA1k(root, index, transform, patch_num)

    # for i in ds:
    #     print(i)

    root, index, transform, patch_num = '/home/fujun/datasets/iqa/AIGC2023/DATA/', [1], None, 1
    ds = AGIQA2023(root, index, transform, patch_num)
