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

# Registrazione dei vettori
vector.register_awkward()

def read_file(
        filepath,
        max_num_particles=200,
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

def changeVector(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(pt**2 + pz**2)
    return e, px, py, pz


# === INIZIALIZZAZIONE ===
filepath = "/home/fede/Desktop/PRIN_project/Jet_Images/Barycenter/TTBar_000.root"
x_particles, x_jets, y = read_file(filepath)

nEvent = 10000
rows = []

for count, (particles, jet) in enumerate(zip(x_particles[:nEvent], x_jets[:nEvent])):
    
    np.random.seed(count)  # seed riproducibile per questo evento

    partons = []
    for i in range(particles.shape[1]):
        pt, eta, phi, en = particles[0][i], particles[1][i], particles[2][i], particles[3][i]
        if pt >= 0:
            eta_random = np.random.uniform(-3, 3)
            phi_random = np.random.uniform(-10, 10)
            eta_n = eta_random
            phi_n = phi_random
            partons.append((pt, eta_n, phi_n, en))
    rows.append(partons)

print("Numero di eventi processati:", len(rows))
 
 
 # Partendo dal risultato di 'rows'
max_parts = max(len(event) for event in rows)  # Massimo numero di partons in un evento

# Inizializzazione di un dizionario per costruire il DataFrame
data = {}
for i in range(max_parts):
    data[f'pt_cost_{i}'] = []
    data[f'eta_cost_{i}'] = []
    data[f'phi_cost_{i}'] = []

# Riempimento dei dati
for event in rows:
    for i in range(max_parts):
        if i < len(event):
            pt, eta, phi, en = event[i]
            data[f'pt_cost_{i}'].append(pt)
            data[f'eta_cost_{i}'].append(eta)
            data[f'phi_cost_{i}'].append(phi)
        else:
            data[f'pt_cost_{i}'].append(0.0)
            data[f'eta_cost_{i}'].append(0.0)
            data[f'phi_cost_{i}'].append(0.0)

# Creazione del DataFrame
df_filtrato = pd.DataFrame(data)
#df_filtrato.head()

def wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi

# Seleziona tutte le colonne con phi_cost_ e aggiungi anche phi_subjet
phi_cols = [col for col in df_filtrato.columns if col.startswith('phi_cost_')]

# Applica la funzione di wrapping a tutte le colonne selezionate
df_filtrato[phi_cols] = df_filtrato[phi_cols].apply(wrap_phi)



columns = df_filtrato.shape[1]  # colonne
columns_res = int(columns / 3)  # Numero di componenti [pt, eta, phi]

df_new = df_filtrato.iloc[:10000,:]
columns = df_new.shape[1]  
columns_res = int(columns / 3)

events_array = df_new.values.reshape(-1, columns_res, 3)


import numpy as np
import matplotlib.pyplot as plt

SIZE = 1.5
PIXEL = 512

# Estrai le variabili
pt_vals = events_array[:, :, 0].flatten()
eta_vals = events_array[:, :, 1].flatten()
phi_vals = events_array[:, :, 2].flatten()

# Crea l’istogramma 2D pesato
hist, xedges, yedges = np.histogram2d(
    eta_vals,
    phi_vals,
    bins=PIXEL,
    range=[[-SIZE, SIZE], [-SIZE, SIZE]],
    weights=pt_vals
)

# Applica log1p all’istogramma
hist_log = np.log1p(hist)


# Visualizza con imshow
#plt.figure(figsize=(8, 8))
#plt.axis('off')
#plt.imshow(
#    hist_log.T,  # trasponi per allineare assi
#    origin='lower',
#    extent=[-1.5, 1.5, -1.5, 1.5],
#    cmap='gray',
#)


#plt.show()

