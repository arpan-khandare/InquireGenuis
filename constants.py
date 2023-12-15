import torch

MAX_SEQ_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_wsd_pytorch = "bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6.zip"
extract_directory = ""
extracted_folder = bert_wsd_pytorch.replace(".zip","")

# model_dir = "bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"
model_dir = 'spyzvarun/BERT_WSD'
