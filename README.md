# MLMLM
MLMLM: Link Prediction with Mean Likelihood Masked Language Model

|  Paper   | Note  |
|  ----  | ----  |
| [MLMLM: Link Prediction with Mean Likelihood Masked Language Model](https://arxiv.org/pdf/2009.07058v1.pdf)  | []() |

# Environment
python3.74 <br/>
pytorch>=1.3.1 <br/>
transformers==2.8.0 <br/>
numpy <br/>
pandas <br/>
tqdm <br/>

# How to run
1. download ```bert-base-uncased``` weights from https://huggingface.co/bert-base-uncased/tree/main to ./pretrained/bert-base-uncased/
2. run following commands
```
cd ./code
CUDA_VISIBLE_DEVICES=0 python main.py
```
