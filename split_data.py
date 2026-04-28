import os
import shutil
import random

SOURCE_DIR = "data"
OUTPUT_DIR = "data_split"

SPLIT_RATIO = (0.7, 0.15, 0.15)  # train, val, test

classes = ["no_tumor", "tumor"]

for cls in classes:
    files = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(files)

    train_end = int(len(files) * SPLIT_RATIO[0])
    val_end = train_end + int(len(files) * SPLIT_RATIO[1])

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_files in splits.items():
        split_path = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(split_path, exist_ok=True)

        for file in split_files:
            src = os.path.join(SOURCE_DIR, cls, file)
            dst = os.path.join(split_path, file)
            shutil.copy(src, dst)

print("Dataset split completed!")