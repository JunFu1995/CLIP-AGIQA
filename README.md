# Vision-Language Consistency Guided Multi-modal Prompt Learning for Blind AI Generated Image Quality Assessment

## Introduction


## Train and Test
First, download datasets from [here].

Second, train and test the model using the following command:
```
python train_test_clip_auxiliary.py --dataset AGIQA3k --model AGIQA
```
The results are recorded in the folder "./log".

## Acknowledgement
This project is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [DBCNN](https://github.com/zwx8981/DBCNN-PyTorch), and [CLIP-IQA](https://github.com/IceClear/CLIP-IQA). Thanks for these awesome works.

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@article{fu2024vision,
  title={Vision-Language Consistency Guided Multi-modal Prompt Learning for Blind AI Generated Image Quality Assessment},
  author={Fu, Jun and Zhou, Wei and Jiang, Qiuping and Liu, Hantao and Zhai, Guangtao},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE}
}
```
## Contact
For any questions, feel free to contact: `fujun@mail.ustc.edu.cn`
