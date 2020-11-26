import config
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Function to create dataloaders
def make_dataloader(texts, labels, sampler):
    
    tokenizer = config.TOKENIZER

    # Remove all quotation marks from text, for consistency.
    texts = [str(text.replace('"', '')) for text in texts]
    
    inputs  = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        max_length=config.MAX_LEN,
        return_tensors='pt'
    )

    input_ids = inputs["input_ids"]
    attention_masks = inputs["attention_mask"]
    if labels is None:
        labels = torch.tensor([0]*len(texts))
    else:
        labels = torch.tensor(labels)


    dataset = TensorDataset(input_ids, attention_masks, labels)
    print(len(dataset))

    # We use different samplers for the train and val datasets-
    if sampler=='Random':
        dataloader = DataLoader(dataset, 
                                sampler=RandomSampler(dataset),
                                batch_size=config.TRAIN_BATCH_SIZE)
    elif sampler=='Sequential':
        dataloader = DataLoader(dataset, 
                                sampler=SequentialSampler(dataset),
                                batch_size=config.TRAIN_BATCH_SIZE)

    # Or no sampling if you are just predicting
    else:
        dataloader = DataLoader(dataset, 
                                batch_size=config.TRAIN_BATCH_SIZE)

    return dataloader