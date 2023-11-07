import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import numpy as np
from skimage.transform import resize

from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.plotter import BSDOSPlotter

from fastai import *
from fastai.vision.all import *
from .Tiff32Image import *

# DATA_DIRECTORY = Path("../../data")
DATA_DIRECTORY = Path("../../../storage/2dmatpedia")
# "henry's local data path"
# DATA_DIRECTORY = Path("../../MPhys_Project/data extraction+fingerprinting/FULL_MATPEDIA_DATA")

MAX_ENERGY_MINUS_EFERMI_NEAR_EFERMI =  28.8
MIN_ENERGY_MINUS_EFERMI_NEAR_EFERMI =  -19.3

def plot(material_id, data_directory=DATA_DIRECTORY, e_bounds=[-4, 4], bs_projection="elements", dos=True):
    
    data_directory = Path(data_directory)
    
    # get bands data
    filename_bands = data_directory/f"bands/{material_id}.json"
    if not filename_bands.exists():
        raise FileNotFoundError("No such file %s" % filename_bands)
        
    bands_dict=json.load(open(filename_bands))
    bands=BandStructureSymmLine.from_dict(bands_dict)

    # create plotter object
    bsp=BSDOSPlotter(vb_energy_range=-e_bounds[0], cb_energy_range=e_bounds[1], fixed_cb_energy=True, font="DejaVu Sans", bs_projection=bs_projection)

    filename_dos = data_directory/f"dos/{material_id}.json"
    if filename_dos.exists() and dos:
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

def plot_from_bands_picture(material_id, band_energies_minus_efermi, data_directory=DATA_DIRECTORY, e_bounds=[-4, 4], verbose=True):
    data_directory = Path(data_directory)
    
    # get bands data
    filename_bands = data_directory/f"bands/{material_id}.json"
    if not filename_bands.exists():
        raise FileNotFoundError("No such file %s" % filename_bands)
        
    band_energies_minus_efermi = np.squeeze(band_energies_minus_efermi) # remove length 1 dimensions
            
        
    bands_dict=json.load(open(filename_bands))
    band_energies_width = np.array(bands_dict["bands"]["1"]).shape[1]
    
    if band_energies_width != band_energies_minus_efermi.shape[1]:
        if verbose:
            print(f"Dimensions of energy array don't match those of {material_id}: resizing.")
        band_energies_minus_efermi = resize(band_energies_minus_efermi, (band_energies_minus_efermi.shape[0], band_energies_width), preserve_range=True)
        
    bands_dict["projection"] = None
    # bands 
    bands_dict["bands"] = {1: band_energies_minus_efermi+bands_dict["efermi"]}
    bands=BandStructureSymmLine.from_dict(bands_dict)

    # create plotter object
    bsp=BSDOSPlotter(vb_energy_range=-e_bounds[0], cb_energy_range=e_bounds[1], fixed_cb_energy=True, font="DejaVu Sans", bs_projection=None)

    ax = bsp.get_plot(bands)  
    
    return ax
    
def plot_from_bands_tensor(material_id, band_energies_tensor_normalized, data_directory=DATA_DIRECTORY, e_bounds=[-4, 4], verbose=True):
    band_energies_minus_efermi = band_energies_tensor_normalized.detach().cpu().numpy()
    band_energies_minus_efermi = band_energies_minus_efermi * (MAX_ENERGY_MINUS_EFERMI_NEAR_EFERMI - MIN_ENERGY_MINUS_EFERMI_NEAR_EFERMI) + MIN_ENERGY_MINUS_EFERMI_NEAR_EFERMI
    
    return plot_from_bands_picture(material_id, band_energies_minus_efermi, data_directory=data_directory, e_bounds=e_bounds, verbose=verbose)
    
def view_prediction(material_id, model, data_directory=DATA_DIRECTORY, e_bounds=[-4, 4], verbose=True):
    fig, ax = plt.subplots(1, 2)
    
    image_filename = data_directory/f"images/energies8/{material_id}.tiff"
    input_tensor = torch.from_numpy(load_tiff_uint32_image(image_filename).astype(np.float64))
    input_tensor = IntToFloatTensor(div=2**16-1)(input_tensor)
    input_tensor = input_tensor[None, None, :, :]
    input_tensor = input_tensor.float().cuda()
    output_tensor = model.forward(input_tensor)
    
    input_tensor = input_tensor.squeeze().cpu()
    output_tensor = output_tensor.detach().squeeze().cpu()
    
    ax[0].set_title("Input")
    ax[0].imshow(input_tensor.numpy())
    
    ax[1].set_title("Reconstruction")
    ax[1].imshow(output_tensor.numpy())
    
    ax_input = plot_from_bands_tensor(material_id, input_tensor)
    ax_input.set_title("Input")
    
    ax_output = plot_from_bands_tensor(material_id, input_tensor)
    ax_output.set_title("Reconstruction")
    
    return ax
