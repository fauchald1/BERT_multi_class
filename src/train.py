import config
import dataset
import engine
import metrics
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm




def run():
    # Takes a two colummn csv containing a text and some category, in that order.
    df = pd.read_csv(config.TRAINING_FILE)
    df.columns = ['text', 'category']
    df["text"] = df["text"].astype(str)
    df["text"] = [x.replace(':',' ') for x in df["text"]]
    # df=df.sample(frac=0.01, replace=True)
    print(df)
    print(df['category'].value_counts())

    possible_labels = df.category.unique()
    possible_labels = np.sort(possible_labels)
    
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)

    df['label'] = df.category.replace(label_dict)
    print(df.head())

    X_train, X_val, y, y_val = train_test_split(df.index.values, 
                                                      df.label.values, 
                                                      test_size=config.TEST_SIZE, 
                                                      random_state=42, 
                                                      stratify=df.label.values)
    
    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'

    print(df.groupby(['category', 'label', 'data_type']).count())

    print(df[df.data_type=='val'].text.values)

    print("Making train..")
    dataloader_train = dataset.make_dataloader(df[df.data_type=='train'].text.values, 
                                               df[df.data_type=='train'].label.values,
                                               'Random')

    print("Making val..")
    dataloader_validation = dataset.make_dataloader(df[df.data_type=='val'].text.values, 
                                                    df[df.data_type=='val'].label.values,
                                                    'Sequential')

    model = BertForSequenceClassification.from_pretrained(config.BERT_PATH, 
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)


    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)
    

    epochs = config.EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = config.DEVICE
    model.to(device)
    print("Device: ", device)

    best = 0
    for epoch in tqdm(range(1, epochs+1)):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {
                      'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2]
                     }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = engine.evaluate(dataloader_validation, model, device)
        val_f1 = metrics.f1_score_func(predictions, true_vals)
        val_acc = metrics.accuracy_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'Accuracy Score (Normalized): {val_acc}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

        if val_f1 > best:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best = val_f1
            tqdm.write(f'Saving as: {config.MODEL_PATH}')
    

if __name__ == "__main__":
    run()
