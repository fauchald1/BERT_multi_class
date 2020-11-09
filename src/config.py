
import transformers

DEVICE = "cuda"
TEST_SIZE = 0.15
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "models/model.bin"
TRAINING_FILE = "../input/news_category.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, 
                                                       do_lower_case=True)
