import os

# Paths to dataset
train_path = "data/train"
test_path = "data/test"

# Count number of classes in training set (subfolders)
train_classes = len([folder for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))])

# Count number of images in test set (since test folder has flat images)
test_images = len([file for file in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, file))])

# Print dataset details
print("===== DATASET SUMMARY =====")
print(f"Training classes found : {train_classes}")
print(f"Total training images  : {sum([len(files) for _, _, files in os.walk(train_path)])}")
print(f"Testing images found   : {test_images}\n")

# Display sample classes from train folder
print("Sample classes from training folder:")
print(sorted(os.listdir(train_path))[:10])

# Display sample test images
print("\nSample test images:")
print(sorted(os.listdir(test_path))[:10])
