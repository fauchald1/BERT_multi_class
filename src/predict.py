import config
import engine
import dataset
import pandas as pd
import numpy as np


import functools
import torch.nn as nn
import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
device = 'cpu'

model = BertForSequenceClassification.from_pretrained(config.BERT_PATH,
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

tokenizer = config.TOKENIZER
max_len = config.MAX_LEN

text = "Hei jeg synes dere har dårlig service, jævla drittsekker"
print (text)
text = " ".join(text.split())
print(text)

dataloader = dataset.make_dataloader([text], None)

_, predictions, true_vals = engine.evaluate(dataloader, model, device)

preds_flat = np.argmax(predictions, axis=1).flatten()

print(predictions, preds_flat, true_vals)

outputs = 1/(1 + np.exp(-predictions[0]))

print(outputs)