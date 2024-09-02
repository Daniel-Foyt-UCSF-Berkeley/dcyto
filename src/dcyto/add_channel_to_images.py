import numpy as np    
import os 
import skimage
from .utils import check_shape
from .utils import save_imageJ_format
    
def add_channel_to_images(original,
                          to_add,
                          save_destination,
                          chan_colores = None,
                          segmentation = True,
                          outline = True,
                          line_thick = 4):
    
    ''' 
    Add channel to image and save images to directory 
    
    Args:
        original (list): paths to original images to be appened to 
        
        to_add (list): paths to single channel images to add 
        
        save_destination (list or str): list of absolute paths/names to 
            save iamges to. If str, absolute paht to a directory to
            save images to. Basename of original will be used as file name 
            
        chan_colores (list, default None): list of color values (grays,red,yellow,
            green,blue,cyan,magenta) for imageJ LUTs. If none, all channels will be 
            gray. 
   
        segmentation (bool, default true): weather or not to_add
            images are segments 
        
        outline (bool, default True): if segmentation == True, wether
            or not to add outline of segmentaiton image to output 
            image. Segmentaion images must be in the form where each
            unique pixle value corasponds to a segment. 
        
        line_thick (int, default 4): if outline == True and if segmentation ==
            True, thickness of outline of segments
        
    Returns:
        list: absolute paths of saved images 
    
    
    '''
    # get/create save destination paths 
    save_destination_paths = []
    if type(save_destination) == str and os.path.isdir(save_destination)==True:
         for path in original:
            base = os.path.basename(path)
            save_destination_paths.append(os.path.join(save_destination,base))
    else:
        save_destination_paths = save_destination
         
            
    for image_path, add_path, save_path in zip(original,to_add,save_destination_paths):
        # read in images
        image = skimage.io.imread(image_path)
        image = check_shape(image)
        toadd =skimage.io.imread(add_path)
        
        to_array = []
        if len(image.shape)>2:
            for chan in range(image.shape[0]):
                to_array.append(image[chan,:,:])
        else:
            to_array.append(image)
        
        # add mask outline
        if segmentation==True:
            if outline == True:
                mask_OL = skimage.segmentation.find_boundaries(toadd)
                mask_OL = skimage.segmentation.expand_labels(mask_OL,line_thick)
                to_array.append(mask_OL)
            else:
                to_array.append(toadd)
        
        else:
            to_array.append(toadd)
            
        if chan_colores == None:
            chan_colores = len(to_array)*['grays']
            
        if len(chan_colores)<len(to_array):
            chan_colores += (len(to_array) - len(chan_colores))*['grays']
            
        if  len(chan_colores)>len(to_array):
            chan_colores = chan_colores[:len(to_array)]
            
        stack = np.array(to_array)
        save_imageJ_format(stack,chan_colores,save_path)
        
    return(save_destination_paths)
