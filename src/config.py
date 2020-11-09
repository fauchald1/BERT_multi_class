
import transformers

DEVICE = "cuda"
TEST_SIZE = 0.15
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 10
BERT_PATH = "bert-base-multilingual-uncased"
MODEL_PATH = "models/translated_only.bin"
TRAINING_FILE = "../input/yelp_nor_001_512.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, 
                                                       do_lower_case=True)
