from transformers import BertForSequenceClassification, BertTokenizerFast
from pathlib import Path


#Для получения истинного рейтинга
def get_true_rating(rating):
    inverted_dct = {v: k for k, v in dct.items()}
    true_rating = inverted_dct[rating]
    return true_rating

def get_prediction(text):
    text = '[CLS] ' + text + ' [SEP]'
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    outputs_logits = outputs['logits'][0].detach().tolist()
    max_index = outputs_logits.index(max(outputs_logits))
    return max_index

def load_models(path:str):
    model = BertForSequenceClassification.from_pretrained(path+'classifier/', num_labels=8)
    tokenizer = BertTokenizerFast.from_pretrained(path+'tokenizer_fast/', do_lower_case=True)
    return model, tokenizer

def sentiment(rating):
    if rating > 6:
        return 'Positive review'
    else:
        return 'Negative review'

path = '/app/film_ratings/models_rating/'

dct = {1:0, 2:1, 3:2, 4:3, 7:4, 8:5, 9:6, 10:7}
model, tokenizer = load_models(path)
