
import transformers

DEVICE = "cuda"
TEST_SIZE = 0.15
MAX_LEN = 64
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = TRAIN_BATCH_SIZE/2
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "models/category3000-len64_batch16.bin"
TRAINING_FILE = "../input/news_category3000.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, 
                                                       do_lower_case=True)
