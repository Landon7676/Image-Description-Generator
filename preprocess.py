import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


# PATHS — EDIT THESE

IMAGE_DIR = "data/flickr30k-images"
CAPTION_FILE = "data/flickr30k-captions/Results.csv"
CLEANED_CAPTIONS_FILE = "data/cleaned_captions.csv"
TRAIN_SPLIT_FILE = "data/train_split.csv"
VAL_SPLIT_FILE = "data/val_split.csv"
TEST_SPLIT_FILE = "data/test_split.csv"

os.makedirs('data', exist_ok=True)

df = pd.read_csv(CAPTION_FILE, delimiter='|')
df.columns = ["image_name", "comment_number", "caption"]
df = df.drop("comment_number", axis=1)

df["image_name"] = df["image_name"].str.strip()
df["caption"] = df["caption"].str.lower().str.strip()

print("Total caption rows:", len(df))
print(df.head())

df.to_csv(CLEANED_CAPTIONS_FILE, index=False)
print(f"Cleaned captions saved to {CLEANED_CAPTIONS_FILE}")


unique_image_names = df["image_name"].unique()

train_img_names, temp_img_names = train_test_split(
    unique_image_names, test_size=0.2, random_state=42
)
val_img_names, test_img_names = train_test_split(
    temp_img_names, test_size=0.5, random_state=42
) 

train_df = df[df["image_name"].isin(train_img_names)]
val_df = df[df["image_name"].isin(val_img_names)]
test_df = df[df["image_name"].isin(test_img_names)]

train_df.to_csv(TRAIN_SPLIT_FILE, index=False)
val_df.to_csv(VAL_SPLIT_FILE, index=False)
test_df.to_csv(TEST_SPLIT_FILE, index=False)

print(f"Train split saved to {TRAIN_SPLIT_FILE} (images: {len(train_img_names)}, captions: {len(train_df)})")
print(f"Validation split saved to {VAL_SPLIT_FILE} (images: {len(val_img_names)}, captions: {len(val_df)})")
print(f"Test split saved to {TEST_SPLIT_FILE} (images: {len(test_img_names)}, captions: {len(test_df)})")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class FlickrDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
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

        return image, caption

# CREATE DATASET + LOADER
# The dataset and loader creation here is for demonstration/testing purposes
# In a real scenario, you'd create separate datasets/loaders for train, val, test splits
# using the respective dataframes (train_df, val_df, test_df).
dataset = FlickrDataset(df, IMAGE_DIR, transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Dataset created successfully! (Note: This uses the full cleaned dataset, not splits)")