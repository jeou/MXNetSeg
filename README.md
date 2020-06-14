# MXNetSeg

This project provides modular implementation for state-of-the-art semantic segmentation models based on [MXNet](https://github.com/apache/incubator-mxnet) framework and [GluonCV](https://github.com/dmlc/gluon-cv) toolkit.

![](./demo/demo_citys.png)

## Environment

We adopt python 3.6.2 and CUDA 10.1 in this project.

1. prerequisites

   ```shell
   pip install -r requirements.txt
   ```

2. [Detail API](https://github.com/zhanghang1989/detail-api) for Pascal Context dataset

3. We also employ [fitlog](https://github.com/fastnlp/fitlog) to generate training logs in addition to the **logging* package. Run the following command to initialize this project to a fitlog one

   ```shell
   fitlog init
   ```

## Usage

### Training

1. Configure hyper-parameters in ./mxnetseg/models/$MODEL_NAME$/config.py

2. Run the `train.py` script

   ```shell
   python train.py --model fcn --ctx 0 1 2 3 --val
   ```

### Inference

Simply run the `eval.py` with arguments need to be specified

```shell
python eval.py --model fcn --backbone resnet50 --resume ./weights/fcnfcn_resnet18_Cityscapes_05_19_00_31_06_best.params --ctx 0 --data cityscapes --ms
```

## Citations

Please kindly cite our paper if you feel our codes helps in your research.

```BibTex
@article{tang2020attention,
  title={Attention-guided Chained Context Aggregation for Semantic Segmentation},
  author={Tang, Quan and Liu, Fagui and Jiang, Jun and Zhang, Yu},
  journal={arXiv preprint arXiv:2002.12041},
  year={2020}
}
```

