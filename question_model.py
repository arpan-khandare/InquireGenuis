from transformers import T5ForConditionalGeneration,T5Tokenizer, AutoTokenizer
import streamlit as st

@st.cache_resource()
def load_question_model():
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    question_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    return question_model, question_tokenizer

# model, tokenizer = load_question_model()

# question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
# question_tokenizer = T5Tokenizer.from_pretrained('t5-base')