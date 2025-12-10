import os
import pandas as pd
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image 


embed_size = 256
hidden_size = 256 
num_layers = 1
num_epochs = 5
batch_size = 32
learning_rate = 0.001
vocab_threshold = 4 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLEANED_CAPTIONS_FILE = os.path.join("data", "cleaned_captions.csv")
TRAIN_SPLIT_FILE = os.path.join("data", "train_split.csv")
VAL_SPLIT_FILE = os.path.join("data", "val_split.csv")
TEST_SPLIT_FILE = os.path.join("data", "test_split.csv")
MODEL_PATH = "model/" 

os.makedirs(MODEL_PATH, exist_ok=True)


class Vocabulary(object):
    def __init__(self, threshold):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.threshold = threshold

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, dataframe):
        """Build the vocabulary from the cleaned captions dataframe."""
        counter = collections.Counter()
        for i, row in dataframe.iterrows():
            caption = row["caption"]
            if isinstance(caption, str):
                words = caption.split() 
                for word in words:
                    counter[word] += 1

        
        words = [word for word, cnt in counter.items() if cnt >= self.threshold]

        
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

        
        for word in words:
            self.add_word(word)

class FlickrDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, vocab, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        caption = row["caption"]

        if self.transform:
            image = self.transform(image)

        tokens = caption.split() 
        caption_tokens = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        
        target = torch.LongTensor(caption_tokens)
        return image, target

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove final fc
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).squeeze()
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

if __name__ == "__main__":
    # Create vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary(threshold=vocab_threshold)
    cleaned_df = pd.read_csv(CLEANED_CAPTIONS_FILE)
    vocab.build_vocabulary(cleaned_df)
    print(f"Vocabulary size: {len(vocab)}")

    train_df = pd.read_csv(TRAIN_SPLIT_FILE)
    val_df = pd.read_csv(VAL_SPLIT_FILE)
    test_df = pd.read_csv(TEST_SPLIT_FILE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)) 
    ])

    def collate_fn(data):
        """Creates mini-batch tensors from the list of tuples (image, caption)."""
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        images = torch.stack(images, 0)

        lengths = torch.tensor([len(cap) for cap in captions])
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab('<pad>'))
        return images, captions, lengths

    train_dataset = FlickrDataset(train_df, "data/flickr30k-images", vocab, transform)
    val_dataset = FlickrDataset(val_df, "data/flickr30k-images", vocab, transform)
    test_dataset = FlickrDataset(test_df, "data/flickr30k-images", vocab, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("DataLoaders created successfully!")

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab('<pad>'))
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions)
            
            targets = captions[:, 1:] 

            outputs = outputs[:, :targets.shape[1], :]

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), torch.exp(loss).item()))
        
        print("Running validation...")
        decoder.eval()
        encoder.eval()
        with torch.no_grad():
            val_loss = 0
            val_total_steps = len(val_loader)
            for i, (images, captions, lengths) in enumerate(val_loader):
                images = images.to(device)
                captions = captions.to(device)

                features = encoder(images)
                outputs = decoder(features, captions)
                targets = captions[:, 1:]
                
                outputs = outputs[:, :targets.shape[1], :]

                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Perplexity: {:5.4f}'
                  .format(epoch, num_epochs, val_loss/val_total_steps, torch.exp(torch.tensor(val_loss/val_total_steps)).item()))

        decoder.train()
        encoder.train()

    torch.save(decoder.state_dict(), os.path.join(MODEL_PATH, 'decoder.pth'))
    torch.save(encoder.state_dict(), os.path.join(MODEL_PATH, 'encoder.pth'))
    print(f"Models saved to {MODEL_PATH}")

    print("Training complete!")