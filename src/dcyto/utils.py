import cellpose.models #2.0
import torch
from matplotlib import pyplot as plt
import skimage
import numpy as np
import math
import os
import pathlib
import pandas as pd
import struct
import fcswrite


def plot_image(image,title = '',size = 20,cmap='gray',cmapp_bar = True):
    '''
    plot image with matplotlib
    
    Args:
        image (numpy array): image to be plotted 
        
        title (str, default ''): title to be plotted above image 
        
        size (int): size of image to be plotted 
        
        cmap (str, default gray): color map of intesities 
        
        cmapp_bar (cool, default True): weather to plot color map 
    
    '''
    fig, ax = plt.subplots(figsize=(size,size))
    plt.imshow(image, cmap=cmap)
    if cmapp_bar == True:
        cbar = plt.colorbar()
    ax.set_title(title)
    plt.show()

def pad_image_with_zeros(image, pad_width):
    '''
    Pad an image with zeros.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        
        pad_width (tuple): The number of pixels to pad (top, bottom, left, right).

    Returns:
        numpy.ndarray: The padded image.
        
    '''
    # Determine the padding for each axis: ((top, bottom), (left, right), (before, after))
    if type(pad_width)==int:
        pad_width = [pad_width]*4
    padding = ((pad_width[0], pad_width[1]), (pad_width[2], pad_width[3]))

    # Pad the image with zeros
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    return padded_image

def Stich(num_xy, image_paths, pad_fraction = 0.01, chan=0, size = 2048):
    '''
    stich images togeather with a pad of zeros betwwen images 
    
    Args:
        num_xy (int): number of images in the x and y dimention 
            of final stiched image total images == num_xy^2
        
        image_paths (list): paths to images to be stiched togeather 
        
        pad_fraction (float, default 0.01): fraction of image that is
            padded with zeros in both dimentions e.g. (200,200) image
            pad_fraction = 0.01 -> (202,202) image
            
        chan (int): channel index to use in stiched image 
        
        size (int, default 2048): final size in pixles of stiched image 
    
    Returns:
        Tuples of numpy array, image_cords {image name : (x1,y1), (x2,y2)}:
        stiched image, image path and cordanace to each tile excluding pad

    
    '''
    
    # make 0 for final image
    tile_full = np.zeros(shape = (size, size))
    
    total_images = num_xy*num_xy
  
    path_to_use = image_paths
    
    pad = int(math.floor(size/num_xy)*pad_fraction)
    
    # make pad even number 
    if pad%2 != 0:
        pad+=1
        
    if pad < 1:
        pad = 2
    half_pad = int(pad/2)
    
     # get image+pad size 
    image_size = math.floor(size/num_xy)-pad    
    
    image_cords = {} # ((x,y)(x,y)) 
    # get positions 
    image_count = 0
    for x in range(num_xy):
        for y in range(num_xy):
            
            if image_count>len(path_to_use)-1:
                continue
            image = skimage.io.imread(path_to_use[image_count])
            
            image = check_shape(image)
            
            # get the chan to segment 
            if len(image.shape) > 2:
                image = image[chan,:,:]
            
            # downsample image 
            image_down_sample = skimage.transform.resize(image, (image_size,image_size), anti_aliasing=True) # uses interploation, might be slow 
            
            # add pad of 0s
            image_pad = pad_image_with_zeros(image_down_sample, half_pad)
            
            single_imageX = image_pad.shape[1]
            single_imageY = image_pad.shape[0]
            
            # get location of image to add
            shiftX = single_imageX*x
            shiftY = single_imageY*y
            
            # add image 
            tile_full[shiftX:shiftX+single_imageX,shiftY:shiftY+single_imageY] += image_pad
            image_cords[path_to_use[image_count]] = ((shiftX+half_pad,shiftY+half_pad),
                                                     (shiftX+single_imageX-half_pad,shiftY+single_imageY-half_pad))
            image_count += 1
            
    return tile_full,image_cords

def up_scale(image, new_dim):
    '''
    resize image useing interpolation of order 0
    wraper for skimage.transform.resize
    
    Args:
        image (numpy array): image to be resized 
        
        new_dim (int): x and y dimention to resize to
    
    Returns:
        numpy array: image resized with interpolation order 0, 
        anti_aliasing=True and range of values preserved 
    
    '''
    return(skimage.transform.resize(image, (new_dim,new_dim),order=0, anti_aliasing=True,preserve_range=True))

def crop_from_cord(image, image_cords):
    '''
    crop images given cordants 
    
    Args:
        image (numpy array): array to be croped 
        
        image_cords (dict): {image name : (x1,y1), (x2,y2)} points
            define a box to be croped 
        
    Returns:
        dict: {image name: numpy array of croped image}
    
    '''
    results = {}
    for cords in image_cords:
        point1, point2 = image_cords[cords] 
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        croped = image[x1:x2,y1:y2]
    
        results[cords] = croped
    return(results)

def up_scal_segs(small,size,eroded):
    '''
    enlarge image with binary erosion (2d only)
    
    Args:
        small (numpy array): array to enlage and erode 
        
        size (int): number of pixles in sigle dimention of final image 
        
        eroded (int): number of pixles to erode the binay mask of small 
            after increasing size 
            
    Returns:
        numpy array: 2d array with size (size,size) and eroded 
    
    '''
    big = up_scale(small , size)
    
    # define square to erode mask 
    selem = skimage.morphology.square(eroded)
    
    # Apply erosion to shrink the segments
    eroded_image = skimage.morphology.binary_erosion((big>0).astype(int), selem)
    big = eroded_image*big
    return(big)

def get_unique_path(paths,sep = '-'):
    '''
    get the unique path identifier separated with sep
    
    Args:
        paths (list): paths to find unique subpaths of each 
        
        sep (str, default '-'): separator used in output 
            inplace of system separator 
            
    Returns:
        list: stings of unique parts of paths separated with sep
    
    '''
    common = os.path.dirname(find_longest_common_substring([os.path.normpath(os.path.dirname(p)) for p in paths]))
    unique = [lstrip_substring(os.path.dirname(p), common) for p in paths]
    unique_ = [u.replace(os.sep,sep) if os.sep in u else u for u in unique]
    unique_ = [s[1:]  for s in unique_ if s[0]==sep]
    
    return(unique_)

def lstrip_substring(s, substring):
    '''
    Removes a specified substring from the beginning of a string if it exists.

    This function checks if the string `s` starts with the given `substring`.
    If it does, the substring is removed from the start of the string. If the 
    string does not start with the substring, the original string is returned unchanged.

    Args:
    
        s (str): The original string from which to remove the substring.
        
        substring (str): The substring to remove from the beginning of the string.

    Returns:
        str: The modified string with the substring removed from the beginning, or
        the original string if the substring is not found at the start.
    
    '''
    
    if s.startswith(substring):
        return s[len(substring):]
    return s

def setup_cellpose(model_type = 'cyto'):
    
    '''
    setsup cellpose model for evaluation, trys to run on GPUs via cuda 
    
    Args:
        model_type (str, default 'cyto'): cellpose2 model to call
    
    Returns:
        model: cellpose model to evaluate
    
    '''
    # check for cuda GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8000000000 else "cpu")
    device_type = device.type
    gpu = True if "cuda" in str(device_type) else False
    
    # https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py
    model = cellpose.models.Cellpose(model_type=model_type,
                                     device=device,
                                     gpu=gpu)
    
    return(model)

def bbox2_cord(img):
    '''
    get bounding box around binary image
    
    Args:
        img (numpy.array): binary image
        
    Returns:
        Tuples: ymin,ymax,xmin,xmax of bounding box
    
    '''
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return(ymin,ymax,xmin,xmax)

def re_shape(image):
    '''
    reshape image to the form shape = (channel/slice, y, x) 
    
    Args:
        image (numpy array): image to reshape with shape 
            (y, x, channel/slice)
    
    Returns:
        numpy array: with the shape (channel/slice, y, x)
    
    '''
    reshaped = []
    for chan in range(min(image.shape)):
        reshaped.append(image[:,:,chan])
    return(np.array(reshaped))

def check_shape(test_image):
    
    '''
    check shape of images and reshap only 3d images
    
    Args:
        test_image (numpy array): array to reshape/check 
    
    Returns:
        numpy array: reshaped array sothat chan/slice is the first index of shape 
        change to (channel/slice, y, x) from (y, x, channel/slice)
    
    '''
    # chekc if image needs reshaping
    if len(test_image.shape) == 3:
        if test_image.shape[2]<test_image.shape[0] and test_image.shape[2]<test_image.shape[1]:
            seg_image = re_shape(test_image)
        else:
            seg_image = test_image
     
    # can not reshape
    elif len(test_image.shape) == 2: 
            seg_image = test_image   
    else:
        raise CustomError("image dose not have correct shape, image shape = "+str(test_image.shape))
        
    return(seg_image)

def apply_mask_extract(image, mask):

    '''
    maks cells and calulate the sum of all pixles in mask for each channel 
    
    Args:
        image (str or numpy.array): path to .tif image or image to extraxt intensities from 
        
        mask (str or numpy.array): path to .tif image or mask with each segment having a unique value
    
    Returns:
        Tuples of list: (cell_ID, cell_sums, cell_areas, cell_edge)
        cell_ID: unique values from mask 
        cell_sums: sum of all pixles in each segment
        cell_areas: number of pixles in each segment 
        cell_edge: 1 if segment touches and edge, 0 otherwise
    
    '''
    cell_ID = []
    cell_sums = []
    cell_areas = []
    cell_edge = []
    
    if type(image)==str and os.path.isfile(image) and image.endswith('.tif'):
        image = skimage.io.imread(image)
        
    if type(mask)==str and os.path.isfile(mask) and mask.endswith('.tif'):
        mask = skimage.io.imread(mask)
    
    # fix images with shape eg (2048, 2048, 4) to (4, 2048, 2048) 
    image = check_shape(image)

    # get cell IDs of cells on edge of image 
    edge_cells = check_for_edge_cell(mask)

    # ittiraate through unique segments 
    for cell_num in np.unique(mask)[1:]: # fist segment is background
        edge = 0
        if cell_num in edge_cells:
             edge = 1
        cell_mask = (mask==cell_num).astype(int)

        # get cord for box crop
        ymin,ymax,xmin,xmax = bbox2_cord(cell_mask)
        
        #crop single cell, much faster this way 
        box_crop_image = image[0:,ymin:ymax+1, xmin:xmax+1]
        box_crop_mask = cell_mask[ymin:ymax+1, xmin:xmax+1] # this is right
    
        area = box_crop_mask.sum() # faster then sum(sum(array))
        
        masked_Cell = box_crop_image*box_crop_mask
        
        sums = []
        
        # singel channel case
        if len(image.shape) == 2:
            sums.append(masked_Cell.sum())
    
        # multi channel case
        elif len(image.shape) > 2:
            for chanel_index in range(image.shape[0]):
                sum_intensities = masked_Cell[chanel_index,:,:].sum()
                sums.append(sum_intensities)
        
        cell_sums.append(sums)
        cell_areas.append(area)
        cell_ID.append(cell_num)
        cell_edge.append(edge)   
        
    return(cell_ID, cell_sums, cell_areas, cell_edge)

def check_for_edge_cell(cell_segment):
    
    '''
    finds segments on edge of image
    
    Args:
        cell_segment (numpy array): segmentation to find edge segments
    
    Returns:
        list: number of segments that have atleast one pixle on edge of
        image 
    
    '''
    shape = np.shape(cell_segment)
    # get all pixle values on edge of image 
    left = list(cell_segment[0:shape[1],0:1].flatten())
    right = list(cell_segment[0:shape[1],shape[1]-1:shape[1]].flatten())
    top = list(cell_segment[0:1,0:shape[1]].flatten())
    bottom = list(cell_segment[shape[1]-1:shape[1],0:shape[1]].flatten())
    # get unique values 
    edge_cells = set(left+right+top+bottom)
    return(edge_cells)

def copy_directory_structure(src_dir, dest_dir, file_extension):
    
    '''
    copy a directory structure in another location 
    
    Args:
        src_dir (str): path to directory to be copied
        
        dest_dir (str): path to location of coppied directory(s)
        
        file_extension (str): file extenison to include, if file extions 
            in subdirecoties only these will be included
            
    Returns:
        str(s): aboslute paths to all subdirecoties in src and dest dir
    
    '''
    
    src_path = pathlib.Path(src_dir)
    dest_path = pathlib.Path(dest_dir)

    src_directories = []  # List to hold absolute paths of source directories
    dest_directories = []  # List to hold absolute paths of destination directories

    # Traverse the source directory
    for root, dirs, files in os.walk(src_path):
        # Check if the directory contains files with the specified extension
        if any(f.endswith(file_extension) for f in files):
            # Calculate the relative path and target directory in the destination
            relative_path = pathlib.Path(root).relative_to(src_path)
            target_dir = dest_path / relative_path

            # Create the directory in the destination
            target_dir.mkdir(parents=True, exist_ok=True)

            # Add the directory paths to the lists
            src_directories.append(str(pathlib.Path(root).resolve()))
            dest_directories.append(str(target_dir.resolve()))

    return src_directories, dest_directories

def divide_list(lst, n):
    '''
    devides list into 'equal' parts 
    
    Args:
        lst (list): list to be devided 
        
        n (int): number of devisions of list 
        
    Returns:
        list: contains sublist of length = len(lst)/n with minimum length ==1 
    
    '''
    return [lst[i:i + n] for i in range(0, len(lst), n)]            

def create_directory(directory_path):
    
    '''
    creates directory in file system, checks if already a directory 
    
    Args:
        directory_path (str): path to direcotry to be made 
        
    Returns: 
        str: directory_path
    
    '''
    
    try:
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")
    return(directory_path) 

def imagej_metadata_tags(metadata, byteorder):
    
    '''
    Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.
    #https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack

    '''
    
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder+('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder+'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder+('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts)//4, bytecounts, True))

def make_LUTs():
    '''
    build LUTs for imageJ fromated colors
    
    Args:
    
    Returns:
        Tuples of numpy arrays: (grays,red,yellow,green,blue,cyan,magenta)
        LUT arrays in 256 bit color 
    
    
    '''
    grays = np.tile(np.arange(256, dtype='uint8'), (3, 1))

    red = np.zeros((3, 256), dtype='uint8')
    red[0] = np.arange(256, dtype='uint8')

    yellow = np.zeros((3, 256), dtype='uint8')
    yellow[0] = np.arange(256, dtype='uint8')
    yellow[1] = np.arange(256, dtype='uint8')
    yellow_blue = [] 
    [yellow_blue.append([x]*5) for x in range(52)]
    yellow_blue = np.array(np.concatenate(yellow_blue)[:256])
    yellow[2] = yellow_blue
    
    green = np.zeros((3, 256), dtype='uint8')
    green[1] = np.arange(256, dtype='uint8')
    
    blue = np.zeros((3, 256), dtype='uint8')
    blue[2] = np.arange(256, dtype='uint8')
    
    cyan = np.zeros((3, 256), dtype='uint8') 
    cyan[1] = np.arange(256, dtype='uint8')
    cyan[2] = np.arange(256, dtype='uint8')

    magenta = np.zeros((3, 256), dtype='uint8') 
    magenta[0] = np.arange(256, dtype='uint8')
    magenta[2] = np.arange(256, dtype='uint8')
    
    return(grays,red,yellow,green,blue,cyan,magenta)

def save_imageJ_format(image,chan_colores,save_path):
    
    '''
    save images in imageJ tif format with specfic colors 
    
    Args:
        image (numpy array): array to be saved in imageJ fromat 
        
        chan_colores (list, default None): list of color values (grays,red,yellow,
            green,blue,cyan,magenta) for imageJ LUTs.
            
        save_path (str): path to save .tif imageJ image to 
    
    Returns:
        str: save_path
    
    '''
    
    if len(chan_colores) != min(image.shape):
        raise TypeError('chanel colors len != number of chanesl in image')
    
    # get 'LTUs'
    grays,red,yellow,green,blue,cyan,magenta = make_LUTs()
    
    colors = {'grays':grays,
              'red':red,
              'yellow':yellow,
              'green':green,
              'blue':blue,
              'cyan':cyan,
              'magenta':magenta}
    
    for col in chan_colores:
        if colors.get(col,'None') == 'None':
            raise TypeError('chanel color = '+col+' not in colors list')
    
    ijtags = imagej_metadata_tags({'LUTs': [colors[col] for col in chan_colores]}, '>')

    skimage.io.imsave(save_path,
                               image.astype('uint16'),
                               imagej=True,
                               byteorder='>',
                               metadata={'mode': 'composite'},
                               extratags=ijtags,
                              check_contrast = False,
                              plugin='tifffile')
    
    return(save_path)
    
def find_longest_common_substring(strings):
    
    '''
    gets longest common substring in string 
    
    Args:
        strings (list): strings to find most common sustring
        
    Returns:
        str: longest common substring  
    '''
    
    if not strings:
        return ""

    # Start with the shortest string in the list
    shortest_str = min(strings, key=len)

    # Initialize the longest common substring
    longest_common_substr = ""

    # Check all possible substrings of the shortest string
    for i in range(len(shortest_str)):
        for j in range(i + 1, len(shortest_str) + 1):
            # Extract the substring
            substring = shortest_str[i:j]

            # Check if this substring is common to all strings
            if all(substring in string for string in strings):
                # Update the longest common substring found
                if len(substring) > len(longest_common_substr):
                    longest_common_substr = substring

    return longest_common_substr
    
def DataFrame_to_fcs(DF, save_path_name, verbose = True):
    
    '''
    save pandas datafram to fcs file, numaric columns only
    
    Args:
        DF (pandas DataFrame): data frame with numaric columns, 'sample'
            needs to be included if separate fcs files desired 
            
        save_path_name (str): path to place to save fcs file(s), directoy or .fcs file 
        
        verbose (bool, default True): print where fcs file(s) is saved
    
    '''
    
    numeric_columns = DF.select_dtypes(include=['number']).columns
    
    # no sample column in datafram 
    if ('sample' in DF.columns) == False:
        if save_path_name.endswith('.fcs') != True:
            save_path_name+='.fcs'
            
        DF_fcs = DF[numeric_columns]
        fcswrite.fcswrite.write_fcs(fcs_path,
                                    DF_fcs.columns.tolist(),
                                    DF_fcs.values,
                                    text_kw_pr={},
                                    endianness="big",
                                    compat_chn_names=False,
                                    compat_copy=True,
                                    compat_negative=True,
                                    compat_percent=True,
                                    compat_max_int16=10000)
        
        if verbose == True:
            print('fcs file at '+fcs_path)
        
    if len(np.unique(DF['sample']))==1:
        
        # use sample name if directory given
        if os.path.isdir(save_path_name) == True:
            base_name = np.unique(DF['sample'])[0]
        else:
            base_name = os.path.basename(save_path_name)
            
        # add '.fcs' to file names
        if base_name.endswith('.fcs') != True:
            base_name+='.fcs'
        
        if os.path.isdir(save_path_name) == True:
            fcs_path = os.path.join(save_path_name,base_name)
        else:
            fcs_path = os.path.join(os.path.dirname(save_path_name),base_name)
            
        DF_fcs =DF[numeric_columns]
        fcswrite.fcswrite.write_fcs(fcs_path,
                                    DF_fcs.columns.tolist(),
                                    DF_fcs.values,
                                    text_kw_pr={},
                                    endianness="big",
                                    compat_chn_names=False,
                                    compat_copy=True,
                                    compat_negative=True,
                                    compat_percent=True,
                                    compat_max_int16=10000)
        if verbose == True:
            print('fcs file at '+fcs_path)
        
    else:
        if save_path_name.endswith('.fcs'):
            print('multiple samples will be included in same .fcs file, for single file per sample set save_path_name to a directory')
        
            fcs_path = save_path_name
            DF_fcs = DF[numeric_columns]
            fcswrite.fcswrite.write_fcs(fcs_path,
                                        DF_fcs.columns.tolist(),
                                        DF_fcs.values,
                                        text_kw_pr={},
                                        endianness="big",
                                        compat_chn_names=False,
                                        compat_copy=True,
                                        compat_negative=True,
                                        compat_percent=True,
                                        compat_max_int16=10000)
            if verbose == True:
                print('fcs file at '+fcs_path)
        
        else:
        # multiple samples multippl e.fcs files
            for sample in np.unique(DF['sample']):
                fcs_name = sample+'.fcs'
                fcs_path = os.path.join(save_path_name,fcs_name)
                DF_fcs = DF[DF['sample'] == sample]
                DF_fcs = DF[numeric_columns]
                fcswrite.fcswrite.write_fcs(fcs_path,
                                            DF_fcs.columns.tolist(),
                                            DF_fcs.values,
                                            text_kw_pr={},
                                            endianness="big",
                                            compat_chn_names=False,
                                            compat_copy=True,
                                            compat_negative=True,
                                            compat_percent=True,
                                            compat_max_int16=10000)
                if verbose == True:
                    print('fcs file at '+fcs_path)
