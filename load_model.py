import torch
from train_model import EncoderCNN, DecoderRNN, Vocabulary
import pandas as pd

vocab = Vocabulary(threshold=4)
cleaned_df = pd.read_csv("data/cleaned_captions.csv")
vocab.build_vocabulary(cleaned_df)

encoder = EncoderCNN(embed_size=256)
decoder = DecoderRNN(embed_size=256, hidden_size=256, vocab_size=len(vocab))

encoder.load_state_dict(torch.load("model/encoder.pth", map_location="cpu"))
decoder.load_state_dict(torch.load("model/decoder.pth", map_location="cpu"))

encoder.eval()
decoder.eval()

print("Models loaded successfully!")
