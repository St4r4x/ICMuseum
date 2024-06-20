import os
import csv

def generate_csv(img_dir, csv_path):
    data = [
        (img_name, os.path.splitext(img_name)[0].split('_')[0])
        for img_name in os.listdir(img_dir)
        if img_name.endswith('.jpg')
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        writer.writerows(data)

generate_csv('./data/train', 'train_labels.csv')
generate_csv('./data/test', 'test_labels.csv')