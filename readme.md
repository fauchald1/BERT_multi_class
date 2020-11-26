## BERT_multi_class classifier

BERT_multi_class trains a BertForSequenceClassification model from a pretrained version of [BERT](https://arxiv.org/abs/1810.04805) with a .csv file containing texts and classes specified in config.py. All interesting hyperparameters can also be tuned in config.py. The models can then be used for predictions, and this is demonstrated in the jupyter notebook 'predict_labels.ipynb'. Two training datasets and their resulting two models are included in this package, for news category prediction and fake news prediction respectively. A CUDA compatible GPU is highly recommended for both training the models and predicting large amounts of data with the models. 

### Installation 
Make sure you have python installed in your environment (preferably a python version preceding 3.9), then run
`pip install  -r requirements.txt`
to install the needed packages. Sometimes torch doesnt install properly. In this case go to [pytorch](https://pytorch.org/) and follow recommended installation instructions for your environment. `Python3.9` seems to not work correctly for these installations.

### Data
Four data-sets are included in this package. 
* iee_clipped.csv contains processed (see report) full text articles with real/fake labels from [FNID](https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset).
* news_category3000.csv contains processed (see report) headlines and their categories from [kaggle](https://www.kaggle.com/rmisra/news-category-dataset)
* all_the_news folder contains our experiment dataset, and is downloaded directly from [kaggle](https://www.kaggle.com/snapcrack/all-the-news) 
* all_the_news_pred.csv in output/ is the outcome of running through the predict_labels notebook.

### Usage
To train a model on a given dataset, cd into src/ then run `python train.py` with your training data and hyperparameters of choice. This can take some extra time to run for the first time, as the BERT models need to be downloaded from the internet. The best model will be saved (according to f1 score) at config.MODEL_PATH. By default, `config.py` is set to the exact hyperparameters to reproduce the `category3000-len64_batch16.bin` model for news category prediction. Changing the input file, batch and max_length to the corresponding parameters (see report) will produce the `iee_clipped-len512-batch2.bin` for fake/real news prediction.

To do predictions, open the predict_labels notebook and follow along, this notebook makes use of many of our defined functions and parameters in dataset.py, engine.py and config.py, so don't change its location unless you want to link them up again yourself. Pickled label dicts instantiated during training are also included in input/ for the two training datasets, so that we can produce readable labels in our output. This is not automated if you introduce a new dataset, so beware.
