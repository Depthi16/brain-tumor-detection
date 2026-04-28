import os
categories = ["tumor", "no_tumor"]
path = "dataset"
for cat in categories:
    folder = os.path.join(path, cat)
    if os.path.exists(folder):
        files = os.listdir(folder)
        print(f"{cat}: {len(files)} files")
    else:
        print(f"{cat}: folder not found")
