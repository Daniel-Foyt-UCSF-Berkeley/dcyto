# dcyto


dcyto is a tool to segment and analyse images of cells to produce flow cytometry like data. 

dcyto was writen by Daniel Foyt. To learn more about the modivation for this tool, look out for our preprint [here](https://huanglab.ucsf.edu/publications).


# Installation

You can install dcyto with native python if you have python3.7+

**Dependencies**, installed automaticaly when installed with conda/pip

 - cellpose (see note below) 
 - scikit-image
 - matplotlib
 - pathlib
 - pandas
 - fcswrite

### Installation Instructions

Simple install with pip from TestPyPI:

    python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dcyto

To install the gpu enabled version of Cellpose which grately increases the speed of segmentation, see the [Cellpose GitHub](https://github.com/MouseLand/cellpose).

# Example
to run dcyto on a directory of tif images:

```python
import dcyto

images_directory = 'path/to/tif/images/directory'
segmentation_out_directory = 'path/to/save/directory'
	
# run segmentation with Cellpose and save to segmentation_out_directory
image_paths, CP_out_paths = dcyto.segment_with_cellpose(images = images_directory,
                                                        save_folder = segmentation_out_directory,
                                                        chan = 0) # channel to segment
```
![alt text](https://github.com/Daniel-Foyt-UCSF-Berkeley/dcyto/blob/main/image1.PNG?raw=true)

```python 

# add segmentation to tif files
outline_save = 'path/to/save/appended/tifs/directory'
dcyto.add_channel_to_images(image_paths,
			    CP_out_paths,
                            outline_save,
                            chan_colores = ['grays','blue','green','red','yellow'],
                            segmentation = True,
                            outline = True,
                            line_thick = 4)


# use segmentation to extract the intensities in each channel in the images and save to a csv and fcs file
results_DataFrame = dcyto.extract_intensities(image_paths = image_paths,
                                              segment_paths = CP_out_paths,
					      save_path_name = results.csv,
					      fcs = True, #FCS3.0 
				              chan_names = ['WF','BFP','GFP','mApple'])

```

![alt text](https://github.com/Daniel-Foyt-UCSF-Berkeley/dcyto/blob/main/image2.PNG?raw=true)
