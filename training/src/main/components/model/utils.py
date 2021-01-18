import os
import random
import datetime

# import matplotlib.pyplot as plt
# import pandas as pd

from google.cloud import storage


def blob_exists(
    filename, projectname="res-nbcupea-dev-ds-sandbox-001", bucket_name="red-dev"
):
    client = storage.Client(projectname)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    return blob.exists()


def show_random_text_and_tags(data_loader, seed=0):
    """Prints a random text and the associated tags from `data_loader.data`,
    a pandas DataFrame containing example texts and associated
    one-hot-encoded tags

    Arguments:
        data_loader - DataLoader - an instance of DataLoader with a `data`
            attribute
        seed - int - random seed
    """
    random.seed(seed)
    i = random.choice(range(data_loader.data.shape[0]))
    print("Text:", "\n")
    print(data_loader.data.loc[i, data_loader.text_col])
    print("\n", "Associated tag(s):")
    print((data_loader.data.loc[i, :].index[data_loader.data.loc[i, :] == 1].tolist()))


def smooth_curve(points, factor=0.75):
    """Smoothes data in `points`, the smoothing being controlled by `factor`
    (between 0 and 1)

    Arguments:
        points - list - floats
        factor - float - between 0 and 1, closer to 1, the greater the smoothing

    Returns:
        smoothed_points - list - smoothed floats
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def get_model_path(model_folder, most_recent=True):
    """Generates a model path based on the model root folder (`model_folder`)

    Arguments:
        model_folder - string - the root folder where the models are saved
        most_recent - boolean - if true, retrieves the most recently updated
            model's path, else returns a path indexed by today's date

    Returns:
        model_path - string - path to the folder to save/load a model
    """
    today = datetime.datetime.today().strftime("%d-%m-%Y")
    if most_recent:
        all_subdirs = [
            os.path.join(model_folder, d)
            for d in os.listdir(model_folder)
            if os.path.isdir(os.path.join(model_folder, d)) and d[0] != "."
        ]
        model_path = max(all_subdirs, key=os.path.getmtime)
    else:
        model_path = os.path.join(model_folder, today)
    return model_path
