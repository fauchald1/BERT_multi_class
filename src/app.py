import config
import dataset
import engine

import torch
import flask
import time
from flask import Flask
from flask import request
import functools
import torch.nn as nn
import numpy as np

from transformers import BertForSequenceClassification


app = Flask(__name__)

MODEL = None
DEVICE = 'cpu'
PREDICTION_DICT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    
    review = str(sentence)
    review = [" ".join(review.split())]
    text = [" ".join(text.split()) for text in review]

    print(text, [review])

    dataloader = dataset.make_dataloader(text, None, None)

    _, predictions, true_vals = engine.evaluate(dataloader, MODEL, DEVICE)

    outputs = 1/(1 + np.exp(-predictions[0]))
    
    return outputs


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    preds = sentence_prediction(sentence)
    response = {}
    response["response"] = {
        "1": str(preds[0]),
        "2": str(preds[1]),
        "3": str(preds[2]),
        "4": str(preds[3]),
        "5": str(preds[4]),
        "Time taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BertForSequenceClassification.from_pretrained(config.BERT_PATH,
                                                          num_labels=len(PREDICTION_DICT),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(DEVICE)))
    MODEL.eval()
    app.run(host="0.0.0.0", port="9999")