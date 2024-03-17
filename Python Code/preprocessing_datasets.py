import os
import cv2
import shutil
import random
import pandas as pd
import tkinter as tk

from keras.utils import image_dataset_from_directory
from tkinter.filedialog import askopenfilename, askdirectory

root = tk.Tk()
root.withdraw()

# batch_size = 32

open_attribute_list_file = askopenfilename(title="Select Attribute List file")
open_images_directory = askdirectory(title="Select Images directory")

def preprocess(batch_size):
    attribute_list_df = pd.read_csv(open_attribute_list_file, sep='\s+', skiprows=1, usecols=['Male'])
    images_folder = os.listdir(open_images_directory)

    attribute_list_df.index.name = 'filename'
    attribute_list_df.replace(to_replace=-1, value=0, inplace=True)
    attribute_list_df.rename(columns={"Male": "Gender"}, inplace=True)
    attribute_list_df.reset_index(drop=False, inplace=True)

    attribute_list_df['isExist'] = attribute_list_df['filename'].isin(images_folder)

    attribute_list_df.drop(attribute_list_df[attribute_list_df['isExist'] == False].index, inplace=True)
    attribute_list_df.reset_index(drop=True, inplace=True)

    print('Splitting Datasets')
    splitted_folder = 'Splitted'
    train_folder = os.path.join(splitted_folder, 'Train')
    validation_folder = os.path.join(splitted_folder, 'Validation')
    test_folder = os.path.join(splitted_folder, 'Test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for index, row in attribute_list_df.iterrows():
        filename = row['filename']
        gender_label = row['Gender']

        if index % 10 < 8:
            splitted_folder = train_folder
        elif index % 10 == 8:
            splitted_folder = validation_folder
        else:
            splitted_folder = test_folder

        if gender_label == 0:
            gender_folder = os.path.join(splitted_folder, 'Female')
        else:
            gender_folder = os.path.join(splitted_folder, 'Male')

        os.makedirs(gender_folder, exist_ok=True)

        source_path = os.path.join(open_images_directory, filename)
        destination_path = os.path.join(gender_folder, filename)

        if not os.path.exists(destination_path):
            print(f'Copying file {source_path} to {gender_folder}')
            shutil.copy(source_path, destination_path)
        else:
            print(f'Skipping file {filename} in {gender_folder} as it already exists.')

    print('Splitting Datasets completed')

    image_shape = cv2.imread(source_path)
    image_size = (image_shape.shape[0], image_shape.shape[1])
    print(image_size)

    train_datasets = image_dataset_from_directory(
        directory=train_folder,
        seed=random.randint(1, 1000),
        class_names=os.listdir(train_folder),
        image_size=image_size,
        batch_size=batch_size
    )

    validation_datasets = image_dataset_from_directory(
        directory=validation_folder,
        seed=random.randint(1, 1000),
        class_names=os.listdir(validation_folder),
        image_size=image_size,
        batch_size=batch_size
    )

    test_datasets = image_dataset_from_directory(
        directory=test_folder,
        seed=random.randint(1, 1000),
        class_names=os.listdir(test_folder),
        image_size=image_size,
        batch_size=batch_size
    )

    return train_datasets, validation_datasets, test_datasets, image_shape
