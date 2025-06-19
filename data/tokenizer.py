import nltk
from collections import Counter
import torch

class CaptionTokenizer:
    def __init__(self, captions, min_freq=5):
        # Download tokenizer only when needed, to avoid import-time side effects
        nltk.download('punkt', quiet=True)

        self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        counter = Counter()
        for cap in captions:
            counter.update(nltk.word_tokenize(cap.lower()))

        vocab = self.special_tokens + [w for w, c in counter.items() if c >= min_freq]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode_caption(self, caption, max_len=50):
        tokens = nltk.word_tokenize(caption.lower())
        tokens = ['<start>'] + tokens[:max_len-2] + ['<end>']
        ids = [self.word2idx.get(w, self.word2idx['<unk>']) for w in tokens]
        ids += [self.word2idx['<pad>']] * (max_len - len(ids))
        return torch.tensor(ids)

    def vocab_size(self):
        return len(self.word2idx)
