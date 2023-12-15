import re
from tqdm import tqdm

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

import torch
from torch.nn.functional import softmax

from constants import *
from functions import *
from wsd_model import load_wsd_model
from question_model import load_question_model


model, tokenizer = load_wsd_model(model_dir)
question_model, question_tokenizer = load_question_model()


def get_sense(sentence, sent):

  re_result = re.search(r"\[TGT\](.*)\[TGT\]", sentence)
  if re_result is None:
      print("\nIncorrect input format. Please try again.")

  ambiguous_word = re_result.group(1).strip()
  results = dict()

  wn_pos = wn.NOUN
  for i, synset in enumerate(set(wn.synsets(ambiguous_word, pos=wn_pos))):
      results[synset] =  synset.definition()

  if len(results) ==0:
    return (None,None,ambiguous_word)

  # print (results)
  sense_keys=[]
  definitions=[]
  for sense_key, definition in results.items():
      sense_keys.append(sense_key)
      definitions.append(definition)


  record = GlossSelectionRecord("test", sent, sense_keys, definitions, [-1])

  features = create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_token_segment_id=0,
                                            disable_progress_bar=True)[0]

  with torch.no_grad():
      logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
      for i, bert_input in tqdm(list(enumerate(features)), desc="Progress"):
          logits[i] = model.ranking_linear(
              model.bert(
                  input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
              )[1]
          )
      scores = softmax(logits, dim=0)

      preds = (sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True))


  # print (preds)
  sense = preds[0][0]
  meaning = preds[0][1]
  return sense,meaning,ambiguous_word

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


def get_question(sentence,answer):
  text = "context: {} answer: {} </s>".format(sentence,answer)
  max_len = 256
  encoding = question_tokenizer.encode_plus(text,max_length=max_len, padding='max_length', return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = question_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=200)


  dec = [question_tokenizer.decode(ids) for ids in outs]


  Question = dec[0].replace("question:","")
  Question = Question.replace("<pad>", "")
  Question = Question.replace("</s>", "")
  Question= Question.strip()
  return Question

def get_sentences(passage):
   pass

def getMCQs(sentence):
#   sentences = get_sentences(passage)
#   sentence = sentences[0]
  sentence_for_bert = sentence.replace("**"," [TGT] ")
  sentence_for_bert = " ".join(sentence_for_bert.split())
  # try:
  sense,meaning,answer = get_sense(sentence_for_bert, sentence)
  if sense is not None:
    distractors = get_distractors_wordnet(sense,answer)
  else: 
    distractors = ["Word not found in Wordnet. So unable to extract distractors."]
  sentence_for_T5 = sentence.replace("**"," ")
  sentence_for_T5 = " ".join(sentence_for_T5.split()) 
  ques = get_question(sentence_for_T5,answer)
  return ques,answer,distractors,meaning


# sentence = "Mark's favourite game is **Cricket**."


# question,answer,distractors,meaning = getMCQs(sentence)
# print (question)
# print (answer)
# print (distractors)
# print (meaning)
