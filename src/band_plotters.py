import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.plotter import BSDOSPlotter

DATA_DIRECTORY = Path("../../data")

def plot(material_id, data_directory=DATA_DIRECTORY, e_bounds=[-4, 4]):
    
    data_directory = Path(data_directory)
    
    # get bands data
    filename_bands = data_directory/f"bands/{material_id}.json"
    if not filename_bands.exists():
        raise FileNotFoundError("No such file %s" % filename_bands)
        
    bands_dict=json.load(open(filename_bands))
    bands=BandStructureSymmLine.from_dict(bands_dict)

    # create plotter object
    bsp=BSDOSPlotter(vb_energy_range=-e_bounds[0], cb_energy_range=e_bounds[1], fixed_cb_energy=True, font="DejaVu Sans")

    filename_dos = data_directory/f"dos/{material_id}.json"
    if filename_dos.exists():
        dos_dict=json.load(open(filename_dos))
        dos=CompleteDos.from_dict(dos_dict)
        ax = bsp.get_plot(bands, dos=dos)
    else:
        ax = bsp.get_plot(bands)  
    plt.show()

def bare_plot(material_id, data_directory=DATA_DIRECTORY, plot_dos=False, e_bounds=[-4, 4], bs_legend=None, rgb_legend=False):
    data_directory = Path(data_directory)
    # get bands data
    filename_bands = data_directory/f"bands/{material_id}.json"
    if not filename_bands.exists():
        raise FileNotFoundError("No such file %s" % filename_bands)
        
    bands_dict=json.load(open(filename_bands))
    bands=BandStructureSymmLine.from_dict(bands_dict)

    # create plotter object
    
    bsp=BSDOSPlotter(vb_energy_range=-e_bounds[0], cb_energy_range=e_bounds[1], fixed_cb_energy=True, font="DejaVu Sans", axis_fontsize=0, tick_fontsize=0, bs_legend=bs_legend, rgb_legend=rgb_legend, fig_size=(8, 8), dos_legend=None)

    filename_dos = data_directory/f"dos/{material_id}.json"
    if filename_dos.exists() and plot_dos:
        dos_dict=json.load(open(filename_dos))
        dos=CompleteDos.from_dict(dos_dict)
        ax = bsp.get_plot(bands, dos=dos)

        for axi in ax:
            axi.spines['left'].set_visible(False)
            axi.spines['bottom'].set_visible(False)
            axi.spines['right'].set_visible(False)
            axi.spines['top'].set_visible(False)
            axi.tick_params(left=False, bottom=False)
            axi.yaxis.grid(False)
            
        plt.subplots_adjust(wspace=0)
        
    else:
        ax = bsp.get_plot(bands)  

        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.yaxis.grid(False)
    
    plt.subplots_adjust(left=-0.001, right=1, top=1+0.001, bottom=0)
