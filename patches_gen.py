import os
import math
import csv
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import poisson


# ----------------------------- FUNZIONI -----------------------------

def get_image_path_full(path):
    images = [os.path.join(path, f) for f in os.listdir(path)
              if f.endswith('.npy') and not f.startswith('.') and '.sys.' not in f]
    return images[:10000]

def rotate_image(image, SIZE):
    angle = random.randint(0, 360)
    center = (SIZE // 2, SIZE // 2)
    M_rotation = cv2.getRotationMatrix2D(center, angle, 1)
    transformed_image = cv2.warpAffine(
        image,
        M_rotation,
        (SIZE, SIZE),
        #flags=cv2.INTER_LANCZOS4,           # interpolazione di alta qualità
        borderMode=cv2.BORDER_CONSTANT,     # bordi neri
        borderValue=0
    )
    return transformed_image

def get_bounding_box(image):
    non_zero_indices = np.argwhere(image > 0)
    if non_zero_indices.size == 0:
        return 0, 0, 0, 0
    ymin, xmin = non_zero_indices.min(axis=0)
    ymax, xmax = non_zero_indices.max(axis=0)
    width = xmax - xmin
    height = ymax - ymin
    return xmin, ymin, width, height

def load_and_process_images(paths, size=130):
    images = []
    bboxes = []
    for path in paths:
        image = np.load(path)
        image = rotate_image(image, size)
        images.append(image)
        bbox = get_bounding_box(image)
        bboxes.append(bbox)
    return images, bboxes

def place_image(image, bbox, matrix, mask):
    IMAGE_SIZE = 130
    MATRIX_SIZE = 512
    for _ in range(1000):
        x_start = random.randint(0, MATRIX_SIZE - IMAGE_SIZE)
        y_start = random.randint(0, MATRIX_SIZE - IMAGE_SIZE)
        selected_area = mask[y_start:y_start + IMAGE_SIZE, x_start:x_start + IMAGE_SIZE]
        if np.all(selected_area):
            matrix[y_start:y_start + IMAGE_SIZE, x_start:x_start + IMAGE_SIZE] = image
            mask[y_start:y_start + IMAGE_SIZE, x_start:x_start + IMAGE_SIZE] = False
            return x_start + bbox[0], y_start + bbox[1], bbox[2], bbox[3]
    return None





# ------------------------- PERCORSI & SETUP -------------------------

path_HToBB = '/eos/user/f/fcampono/Patches/imgs_blobs/npy_HToBB/npy_HToBB_025'
path_QCD   = '/eos/user/f/fcampono/Patches/imgs_blobs/npy_QCD/npy_QCD_025'
path_TTBar = '/eos/user/f/fcampono/Patches/imgs_blobs/npy_TTBar/npy_TTBar_025'
output_image_dir = "/eos/user/f/fcampono/Patches/imgs_patches/P_NoBkg/P_NoBkg_025"
#output_image_dir = '/home/fede/Desktop/Patches_fixed_bug/test_assemblate'

output_csv_path = os.path.join(output_image_dir, "bbox_P_NoBkg_025.csv")
os.makedirs(output_image_dir, exist_ok=True)

SIZE = 130
MATRIX_SIZE = 512
NUM_IMGS = 3000

# -------------------------- CARICAMENTO DATI -------------------------

imgs_HToBB = get_image_path_full(path_HToBB)
imgs_QCD   = get_image_path_full(path_QCD)
imgs_TTBar = get_image_path_full(path_TTBar)

list_HToBB, bboxes_HToBB = load_and_process_images(imgs_HToBB)
list_QCD, bboxes_QCD     = load_and_process_images(imgs_QCD)
list_TTBar, bboxes_TTBar = load_and_process_images(imgs_TTBar)

# ----------------------- GENERAZIONE CANVAS --------------------------

rows = []

# All'inizio, crea liste di indici disponibili per ogni classe:
available_hbb = list(range(len(list_HToBB)))
available_qcd = list(range(len(list_QCD)))
available_ttbar = list(range(len(list_TTBar)))


df = pd.DataFrame(columns=["img_name", "x_min", "y_min", "w", "h", "label_blob"])

for i in range(NUM_IMGS):

    canvas = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    mask = np.ones((MATRIX_SIZE, MATRIX_SIZE), dtype=bool)
    img_name = f"image_P_NoBkg_025_{i:04d}.npy"

    num_hbb = poisson(1).rvs()
    num_qcd = poisson(1).rvs()
    num_ttbar = poisson(1).rvs()

    #num_hbb = min(num_hbb, len(available_hbb))
    #num_qcd = min(num_qcd, len(available_qcd))
    #num_ttbar = min(num_ttbar, len(available_ttbar))

    blobs_placed = False

    # HBB
    selected_hbb = random.sample(available_hbb, num_hbb)
    for idx in selected_hbb:
        image = list_HToBB[idx]
        bbox = bboxes_HToBB[idx]
        placed_bbox = place_image(image, bbox, canvas, mask)
        if placed_bbox is not None:
            blobs_placed = True
            new_row = pd.DataFrame([[img_name, *placed_bbox, "hbb"]],
                                   columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
    available_hbb = [idx for idx in available_hbb if idx not in selected_hbb]

    # QCD
    selected_qcd = random.sample(available_qcd, num_qcd)
    for idx in selected_qcd:
        image = list_QCD[idx]
        bbox = bboxes_QCD[idx]
        placed_bbox = place_image(image, bbox, canvas, mask)
        if placed_bbox is not None:
            blobs_placed = True
            new_row = pd.DataFrame([[img_name, *placed_bbox, "qcd"]],
                                   columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
    available_qcd = [idx for idx in available_qcd if idx not in selected_qcd]

    # TTBar
    selected_ttbar = random.sample(available_ttbar, num_ttbar)
    for idx in selected_ttbar:
        image = list_TTBar[idx]
        bbox = bboxes_TTBar[idx]
        placed_bbox = place_image(image, bbox, canvas, mask)
        if placed_bbox is not None:
            blobs_placed = True
            new_row = pd.DataFrame([[img_name, *placed_bbox, "ttbar"]],
                                   columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
    available_ttbar = [idx for idx in available_ttbar if idx not in selected_ttbar]

    if not blobs_placed:
        new_row = pd.DataFrame([[img_name, '', '', '', '', '']], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
    
    canvas_norm = cv2.normalize(src=canvas, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    np.save(os.path.join(output_image_dir, img_name), canvas_norm)


def assign_label_image(group):
    labels = set(group['label_blob'])
    if 'hbb' in labels:
        return 1
    elif 'qcd' in labels:
        return 0
    else:
        return 2

label_image_df = df.groupby('img_name').apply(assign_label_image).reset_index()
label_image_df.columns = ['img_name', 'label_image']

df = df.merge(label_image_df, on='img_name', how='left')
df['label_image'] = df['label_image'].fillna(2)

df.to_csv(output_csv_path, index=False)

