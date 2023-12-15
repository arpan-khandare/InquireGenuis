import os
import zipfile
import torch
import streamlit as st
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from constants import *

#  If unzipped folder exists don't unzip again.
# if not os.path.isdir(extracted_folder):
#   with zipfile.ZipFile(bert_wsd_pytorch, 'r') as zip_ref:
#       zip_ref.extractall(extract_directory)
# else:
#   print (extracted_folder," is extracted already")


class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

@st.cache_resource()
def load_wsd_model(model_dir):
    model = BertWSD.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # add new special token
    if '[TGT]' not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
        assert '[TGT]' in tokenizer.additional_special_tokens
        model.resize_token_embeddings(len(tokenizer))
        
    model = model.to(DEVICE)
    model = model.eval()

    return model, tokenizer

# model, tokenizer = load_wsd_model(model_dir)









# model = BertWSD.from_pretrained(model_dir)
# tokenizer = BertTokenizer.from_pretrained(model_dir)

# # add new special token
# if '[TGT]' not in tokenizer.additional_special_tokens:
#     tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
#     assert '[TGT]' in tokenizer.additional_special_tokens
#     model.resize_token_embeddings(len(tokenizer))
    
# model = model.to(DEVICE)
# model = model.eval()

