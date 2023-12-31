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

from torchvision import transforms
from fastai import *
from fastai.vision.all import *
from .Tiff32Image import *

# DATA_DIRECTORY = Path("../../data")
DATA_DIRECTORY = Path("/storage/2dmatpedia")
ANUPAM_PATH = Path("/notebooks/band-fingerprint/fingerprints/anupam_original.csv")

# "henry's local data path"
# DATA_DIRECTORY = Path("../../MPhys_Project/data extraction+fingerprinting/FULL_MATPEDIA_DATA")

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
    
def plot_from_bands_tensor(material_id, band_energies_tensor_normalized, min_energy_minus_efermi, max_energy_minus_efermi, data_directory=DATA_DIRECTORY, e_bounds=[-4, 4], verbose=True):
    band_energies_minus_efermi = band_energies_tensor_normalized.detach().cpu().numpy()
    band_energies_minus_efermi = band_energies_minus_efermi * (max_energy_minus_efermi - min_energy_minus_efermi) + min_energy_minus_efermi
    
    return plot_from_bands_picture(material_id, band_energies_minus_efermi, data_directory=data_directory, e_bounds=e_bounds, verbose=verbose)

def pad_or_crop_to_height(image, desired_height):
    # Get the current size of the image
    current_height = image.shape[0]
    
    if current_height < desired_height:

        # Calculate the pad width for each axis
        pad_width = [((desired_height-current_height) // 2, (desired_height-current_height + 1) // 2),
                     (0, 0)]
                     

        # Pad the image with zeros using np.pad
        image = np.pad(image, pad_width, mode='constant', constant_values=0)

    # Crop the padded image to the desired size
    image = image[:desired_height]

    return image

    
def view_prediction(material_id, model, min_energy_minus_efermi, max_energy_minus_efermi, data_directory=DATA_DIRECTORY, image_directory="energies_12_nearest_bands",
                    device="gpu", e_bounds=[-4, 4], verbose=True, width=None, height=None, height_mode="pad", act_func=None):
    fig, ax = plt.subplots(2, 1)
    
    image_filename = data_directory/f"images/{image_directory}/{material_id}.tiff"
    input_numpy = load_tiff_uint16_image(image_filename).astype(np.float64)
    
    if width:
        input_numpy = resize(input_numpy, (input_numpy.shape[0], width))
    
    if height:
        if height_mode.lower() == "pad":
            input_numpy = pad_or_crop_to_height(input_numpy, height)
        elif height_mode.lower() == "squish":
            input_numpy = resize(input_numpy, (height, input_numpy.shape[1]))#
        else:
            print("Invalid height_mode: can only be pad or squish.")
    
    input_tensor = torch.from_numpy(input_numpy)
    input_tensor = input_tensor / (2**16-1)
    
    input_tensor = input_tensor[None, None, :, :]
    if device == "gpu":      
        input_tensor = input_tensor.float().cuda()
        model.cuda()
    else:
        input_tensor = input_tensor.float().cpu()
        model.cpu()   
        
    output_tensor = model.forward(input_tensor)
    
    if act_func:
        output_tensor = act_func(output_tensor[0])
    
    input_tensor = input_tensor.squeeze().cpu()
    output_tensor = output_tensor.detach().squeeze().cpu()
    
    ax[0].set_title("Input")
    ax[0].imshow(input_tensor.numpy())
    
    ax[1].set_title("Reconstruction")
    ax[1].imshow(output_tensor.numpy())
    
    ax_input = plot_from_bands_tensor(material_id, input_tensor, min_energy_minus_efermi, max_energy_minus_efermi, e_bounds=e_bounds, verbose=False)
    ax_input.set_title("Input")
    
    ax_output = plot_from_bands_tensor(material_id, output_tensor, min_energy_minus_efermi, max_energy_minus_efermi, e_bounds=e_bounds, verbose=False)
    ax_output.set_title("Reconstruction")
    
    return ax

def view_prediction_images(material_id, model, data_directory=DATA_DIRECTORY, image_directory="no_dos_bw_dpi_10/band_images",
                    device="gpu", e_bounds=[-4, 4], verbose=True, width=None, height=None, height_mode="pad", act_func=None):
    
    input_image_path = data_directory/f"images/{image_directory}/{material_id}.png"
    input_image = Image.open(input_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Adjust height and width as needed
        transforms.ToTensor(),
    ])
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    
    # Generate predictions
    with torch.no_grad():
        reconstructed_image_tuple = model(input_tensor)

    # Access the relevant tensor from the tuple
    reconstructed_image = F.sigmoid(reconstructed_image_tuple[0])

    # Convert tensors to NumPy arrays for visualization
    input_image_np = np.transpose(input_tensor.squeeze().numpy(), (1, 2, 0))
    reconstructed_image_np = np.transpose(reconstructed_image.squeeze().numpy(), (1, 2, 0))

    # # resize? not sure if correct
    # reconstructed_image_np = reconstructed_image_np/255.0
    
    #print(input_image_np)
    #print(reconstructed_image_np)
    # Display the input and reconstructed images
    plt.subplot(1, 2, 1)
    plt.imshow(input_image_np)
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image_np)
    plt.title('Reconstructed Image')

    plt.show()
    
    # visdom view
#     import visdom

#     vis = visdom.Visdom()

#     # Send input and reconstructed images to Visdom
#     vis.image(input_image_np.transpose((2, 0, 1)), win='Input Image', opts=dict(title='Input Image'))
    #vis.image(reconstructed_image_np.transpose((2, 0, 1)), win='Reconstructed Image', opts=dict(title='Reconstructed Image'))
    
    return 0

def load_band_image_array(material_id, npz_path, npz_filename, npz_key="images"):
    
    anupam_df = pd.read_csv(ANUPAM_PATH, index_col="ID")
    i = anupam_df.index.get_loc(material_id)
    images = np.load("{0}/{1}.npz".format(npz_path, npz_filename))[npz_key]
    
    input_array = images[i]
    #input_tensor = torch.from_numpy(input_array).cpu()

    return input_array

def binarize(array_data, threshold=0.8):
    array_data[array_data>=threshold] = 1.0
    array_data[array_data<=threshold] = 0.0
    return array_data

def view_prediction_npz(material_id, model, npz_path, npz_filename, npz_key="images", bool_binarise=False, threshold=0.8):
    model.cpu()

    input_array = load_band_image_array(material_id, npz_path, npz_filename, npz_key="images")
    input_tensor = torch.from_numpy(input_array).cpu()
    input_tensor = input_tensor.unsqueeze(0).float()
    
    with torch.no_grad():
        prediction = F.sigmoid(model(input_tensor)[0])
        
    prediction = prediction.detach().squeeze().numpy()
    
    if(bool_binarise):
        prediction = binarize(prediction, threshold=threshold)
    
    fig, ax  = plt.subplots(2, 1)
    ax[0].set_title("Input")
    ax[0].imshow(input_array)
    
    ax[1].set_title("Reconstruction")
    ax[1].imshow(prediction)
    
    
    
    
    
# def view_prediction(material_id, learner, min_energy_minus_efermi, max_energy_minus_efermi, data_directory=DATA_DIRECTORY, image_directory="energies_12_nearest_bands", device="gpu", e_bounds=[-4, 4], verbose=True, width=None):
#     fig, ax = plt.subplots(2, 1)
    
#     image_filename = data_directory/f"images/{image_directory}/{material_id}.tiff"
#     image = TiffImage.create(image_filename, with_input=True)
 
#     (_, _, prediction, inp) = learner.predict(image, with_input=True)
#     # if width:
#     #     input_numpy = resize(input_numpy, (input_numpy.shape[0], width))
    
#     print(torch.equal(inp, prediction))
    
#     prediction = prediction[0]
    
#     ax[0].set_title("Input")
#     ax[0].imshow(np.array(image))
    
#     ax[1].set_title("Reconstruction")
#     ax[1].imshow(prediction.numpy())
    
#     ax_input = plot_from_bands_tensor(material_id, tensor(image, min_energy_minus_efermi, max_energy_minus_efermi, e_bounds=e_bounds, verbose=False)
#     ax_input.set_title("Input")
    
#     ax_output = plot_from_bands_tensor(material_id, prediction, min_energy_minus_efermi, max_energy_minus_efermi, e_bounds=e_bounds, verbose=False)
#     ax_output.set_title("Reconstruction")
    
#     return ax
