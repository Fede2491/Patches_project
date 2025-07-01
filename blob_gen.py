import cv2
import time
import os
import pandas as pd
import numpy as np
import awkward as ak
import uproot
import vector
import fastjet
import logging
from matplotlib import pyplot as plt
from energyflow.utils import (center_ptyphims, reflect_ptyphims, rotate_ptyphims, pixelate)
import json

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Registrazione dei vettori
vector.register_awkward()

# Funzione per leggere il file ROOT
def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):

    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    table = uproot.open(filepath)['tree'].arrays()

    p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi

    x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=1)
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)

    return x_particles, x_jets, y

# Funzione per cambiare il vettore
def changeVector(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(pt**2 + pz**2)
    return e, px, py, pz

def wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi

# Funzione per ottenere i percorsi dei file
#def get_filepaths(base_dir, num_parts=10, num_files_per_part=10):
#    filepaths_list = []
#    for i in range(num_parts):  # 0 to 9
#        part_dir = os.path.join(base_dir, f"JetClass_Pythia_train_100M_part{i}")
#        for j in range(num_files_per_part):  # 000 to 009, 010 to 019, etc.
#            filename = f"HToBB_{i*10 + j:03d}.root"  # Filename of the root file
#            filepath = os.path.join(part_dir, filename)
#            filepaths_list.append(filepath)  # Save the file path
#return filepaths_list

def get_filepaths(base_dir, num_parts=10, num_files_per_part=10, exclude_files=None):
    if exclude_files is None:
        exclude_files = []  # Lista vuota se non viene passata una lista di esclusione
    
    filepaths_list = []
    for i in range(num_parts):  # 0 to 9
        part_dir = os.path.join(base_dir, f"JetClass_Pythia_train_100M_part{i}")
        for j in range(num_files_per_part):  # 000 to 009, 010 to 019, etc.
            filename = f"HToBB_{i*10 + j:03d}.root"  # Filename of the root file
            
            # Escludi il file se il numero è nella lista di esclusione
            if f"{i*10 + j:03d}" not in exclude_files:
                filepath = os.path.join(part_dir, filename)
                filepaths_list.append(filepath)  # Aggiungi il percorso del file
    
    return filepaths_list
    
# PARAMETRI
N_JETS = 100000
MAX_IMAGES = 12000
NUM_BLOBS = 25
NUM_PIXEL = 130
NUM_ZOOM  = 3

image_count = 0
images_id = 0

###### Directory base dei file ROOT
base_dir = "/eos/user/f/fcampono/PRIN_JetImgs/Dataset/Dataset_root/"

# file esclusi dalla generazione delle prime 10 k immagini a 100 jets
exclude_files = []

# Ottieni la lista dei file escludendo quelli indicati
list_dir = get_filepaths(base_dir, exclude_files=exclude_files)

#list_dir[0]

# Itera sui file ROOT
for index_root in range(len(list_dir)): 
    if image_count == MAX_IMAGES:
        print("Numero massimo di immagini generato. Interruzione del processo.")
        break

    filepath = list_dir[index_root]
    print(f"Inizio elaborazione file {index_root + 1}/{len(list_dir)}: {filepath}")
    
    try:
        x_particles, x_jet, y = read_file(filepath)
        
        jetdef1 = fastjet.JetDefinition(fastjet.antikt_algorithm, 1)
        
        rows = []  # Lista per raccogliere le righe del DataFrame

        for count, (particles, jet) in enumerate(zip(x_particles[:N_JETS], x_jet[:N_JETS])):
            partons = []
            for i in range(particles.shape[1]):
                pt, eta, phi, en = particles[0][i], particles[1][i], particles[2][i], particles[3][i]
                if pt > 0:
                    e, px, py, pz = changeVector(pt, eta, phi)
                    partons.append(fastjet.PseudoJet(float(e), float(px), float(py), float(pz)))
                
            cs1 = fastjet.ClusterSequence(partons, jetdef1)
            jets = sorted(cs1.inclusive_jets(), key=lambda j: j.pt(), reverse=True)
        
            for idx, subjet in enumerate(jets):
                row = {
                    'jet': count + 1,
                    'subjet_index': idx + 1,
                    'pt_subjet': subjet.pt(),
                    'eta_subjet': subjet.eta(),
                    'phi_subjet': subjet.phi(),
                }
        
                for k, c in enumerate(subjet.constituents()):
                    row[f'pt_cost_{k}'] = c.pt()
                    row[f'eta_cost_{k}'] = c.eta()
                    row[f'phi_cost_{k}'] = c.phi()
        
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df_filtrato = df.fillna(0.0)
        
        # Seleziona tutte le colonne con phi_cost_ e aggiungi anche phi_subjet
        phi_cols = [col for col in df_filtrato.columns if col.startswith('phi_cost_')]

        # Applica la funzione di wrapping a tutte le colonne selezionate
        df_filtrato[phi_cols] = df_filtrato[phi_cols].apply(wrap_phi)

        image_sum = np.zeros((NUM_PIXEL, NUM_PIXEL))
        blob_number = 0
        
        n_jet_total = int(df_filtrato['jet'].max())
        n_valid_jet = (n_jet_total // NUM_BLOBS) * NUM_BLOBS

        #for i in range(1, df_filtrato['jet'].max() + 1):
        for i in range(1, n_valid_jet + 1):

            df_new = df_filtrato[df_filtrato['jet']== i].iloc[:,5:]
            columns = df_new.shape[1]  
            columns_res = int(columns / 3)
        
            events_array = df_new.values.reshape(-1, columns_res, 3)
            events_centered = [center_ptyphims(event, center='ptscheme') for event in events_array]
            events_reflected_and_rotated = [reflect_ptyphims(rotate_ptyphims(event, center='ptscheme')) for event in events_centered]   
            
            images = [pixelate(event, 
                               npix= NUM_PIXEL, 
                               img_width= NUM_ZOOM, 
                               nb_chan=1, 
                               norm=False, 
                               charged_counts_only=False) for event in events_reflected_and_rotated]
        
            images = np.array(images).reshape(len(images), NUM_PIXEL, NUM_PIXEL)
            image_sum += np.sum(images, axis=0)
        
            blob_number += 1
        
            if blob_number == NUM_BLOBS:
                #image_sums.append(image_sum)
                image_log = np.log1p(image_sum)
                np.save(f"/eos/user/f/fcampono/Patches/blobs/npy_HToBB_025/HToBB_025_fileroot_{index_root:03d}_id_{images_id:05d}", image_log)

                image_count += 1
                images_id += 1

                image_sum = np.zeros((NUM_PIXEL,NUM_PIXEL))
                blob_number = 0

                if image_count == MAX_IMAGES:
                    print("Numero massimo di immagini generato. Interruzione immediata.")
                    break


    except Exception as e:
        print(f"Errore nell'elaborazione del file {filepath}: {e}")
        continue
        
