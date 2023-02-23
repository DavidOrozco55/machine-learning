import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

labels = {}
options = []

def get_normal_dist (breedDf):
  mu = breedDf['edad'].mean()
  variance = breedDf['edad'].var()
  sigma = math.sqrt(variance)
  x = np.arange(0, 20, 0.005)
  return mu, sigma,  x

  
def submit():
  selected_indices = combobox.curselection()

  all_indices = set(range(combobox.size()))
  selected_items = [combobox.get(idx) for idx in selected_indices]
  unselected_indices = all_indices.difference(selected_indices)
  unselected_options = [combobox.get(idx) for idx in unselected_indices]
  for selectedBreed in selected_items:
      ln = labels[selectedBreed]
      ln.set_visible(True)
  for unslb in unselected_options:
    ln = labels[unslb]
    ln.set_visible(False)

  fig.canvas.draw_idle()


def getBreedGroupsFromFile (): 
  dataDogs = pd.read_excel('data_13feb_perros.xlsx').dropna()
  df = pd.DataFrame(dataDogs, columns=['edad', 'raza'])
  df['raza'] = df['raza'].str.strip()
  return df.groupby('raza')

root = tk.Tk()
root.geometry("700x700")

fig = Figure(figsize=(6, 5), dpi=100)
plot = fig.add_subplot(111)
for name, group in getBreedGroupsFromFile():
  if (len(group.index) >= 2):
    mu, sigma, x = get_normal_dist(group)
    label, = plot.plot(x, stats.norm.pdf(x, mu, sigma), label=name)
    labels[name] = label
    options.append(name)
plot.set_xlabel("Edad")
plot.set_ylabel("Frecuencia")
plot.legend()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

submit_button = tk.Button(root, text="Seleccionar", command=submit, height=2, width=10)
submit_button.pack()

combobox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=len(options))
combobox.pack()

for option in options:
  combobox.insert("end", option)

combobox.selection_set(0, 'end')
root.mainloop()
