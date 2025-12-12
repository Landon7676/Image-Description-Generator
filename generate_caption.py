import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd

from train_model import EncoderCNN, DecoderRNN, Vocabulary

EMBED_SIZE = 256
HIDDEN_SIZE = 256
VOCAB_THRESHOLD = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_CAPTIONS_FILE = os.path.join(BASE_DIR, "data", "cleaned_captions.csv")

ENCODER_PATH = os.path.join(BASE_DIR, "model", "encoder.pth")
DECODER_PATH = os.path.join(BASE_DIR, "model", "decoder.pth")


def load_vocab():
    cleaned_df = pd.read_csv(CLEANED_CAPTIONS_FILE)
    vocab = Vocabulary(threshold=VOCAB_THRESHOLD)
    vocab.build_vocabulary(cleaned_df)
    return vocab


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])


def generate_caption(image_path, encoder, decoder, vocab, transform, max_len=20):
    encoder.eval()
    decoder.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)  

    with torch.no_grad():
        features = encoder(image)  

        # Start with <start> token
        start_token = torch.tensor([vocab('<start>')]).to(DEVICE)  
        sampled_ids = []

        emb_start = decoder.embed(start_token).unsqueeze(1)  
        inputs = torch.cat((features.unsqueeze(1), emb_start), dim=1)  

        outputs, (h, c) = decoder.lstm(inputs)
        output = outputs[:, -1, :]               
        logits = decoder.linear(output)         
        predicted = logits.argmax(1)             
        sampled_ids.append(predicted.item())

        for _ in range(max_len - 1):
            if predicted.item() == vocab('<end>'):
                break

            emb = decoder.embed(predicted).unsqueeze(1)   
            outputs, (h, c) = decoder.lstm(emb, (h, c))   
            output = outputs[:, -1, :]
            logits = decoder.linear(output)
            predicted = logits.argmax(1)
            sampled_ids.append(predicted.item())

    words = []
    for word_id in sampled_ids:
        word = vocab.idx2word.get(word_id, '<unk>')
        if word == '<end>':
            break
        words.append(word)

    sentence = " ".join(words)
    return sentence


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image (e.g., data/flickr30k-images/1000092795.jpg)",
    )
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        print(f"Error: image file not found: {image_path}")
        return

    print("Loading vocabulary...")
    vocab = load_vocab()
    print(f"Vocab size: {len(vocab)}")

    print("Loading models...")
    encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=True))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE, weights_only=True))

    transform = get_transform()

    print("Generating caption...")
    caption = generate_caption(image_path, encoder, decoder, vocab, transform)
    print("\nImage:", image_path)
    print("Caption:", caption)


if __name__ == "__main__":
    main()
