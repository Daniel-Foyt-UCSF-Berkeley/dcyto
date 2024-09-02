import numpy as np    
import os 
import skimage
from .utils import setup_cellpose
from .utils import up_scal_segs
from .utils import Stich
from .utils import copy_directory_structure
from .utils import divide_list
from .utils import crop_from_cord

def segment_with_cellpose(images,
                          save_folder,
                          chan = 0,
                          
                          # cellpose settings
                          model_type = 'cyto',
                          invert=False,
                          normalize=True,
                          diameter=150,
                          net_avg=False,
                          resample=True,
                          interp=True,
                          flow_threshold=2,
                          cellprob_threshold=0.0,
                          min_size=15,
                          rescale= None,
                          
                          # stiching settings 
                          stich_for_CP = False,
                          stiched_image_tiles = 5,
                          eroded = 3):
    '''
    
    run cellpose on images for a specfic channel/slice

    Args:
        images (list or str): numpy arrays or absolute paths to .tif images
            or directory with .tif files. All images must have the '.tif' end
            or they will not be segmented.
        
        save_folder (str): path to directory where cellpose mask are saved
            if images is a directory with subfolders, file structure will be copied. 
        
        chan (int, default 0): channel or slice to run segmentation on. 
        
        model_type (str, defalt 'cyto'): builtin cellpose model to use, 
            can be path to coustom model.
        
        diameter (float, defalt 150): Estimate of cell diameter (pixles) for cellpose.
            If None, cellpose will estimate. Defaults to 150.
        
        invert: (bool, default False): invert image pixel intensity before running
            cellpose network (if True, image is also normalized)

        normalize (bool, default True): normalize data so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel

        net_avg (bool, default False): runs the 4 built-in cellpose networks and 
            averages them if True, runs one network if False

        resample(bool, default True): run  cellpose dynamics at original
            image size (will be slower but create more accurate boundaries)

        interp (bool, default True): interpolate during 2D cellpose dynamics

        flow_threshold (float, default 0.4): cellpose flow error threshold
            (all cells with errors below threshold are kept) 

        cellprob_threshold (float, default 0.0): all pixels with value above 
            cellpose threshold kept for masks, decrease to find more and larger masks

        min_size (int, default 15): minimum number of pixels per mask, can turn off with -1

        rescale (float, default None): if diameter is set to None,
            and rescale is not None, then rescale is used instead of diameter for resizing image
        
        stich_for_CP (bool, defaule False): stich images togeather with gap
            between and downsample to (2048,2048) to speed up segmentaion but lose resolution
        
        stiched_image_tiles (int, default 5): number of images in each dimension
            when stiched, total images = (stiched_image_tiles)^2
        
        eroded (int, default 3): if stich_for_CP==True, number of pixles to 
            erode segments so background not included in segment


    Returns:
        Tuples of list or list: if input is list of np.arrays, returns list of paths
            to segmented images named seg_image_#.tif indexed to images. otherwise, 
            returns (list of paths to input images, list of paths to segmented images) 
            segmentation images will have same basenaem as images if paths or directory 
            given

    ''' 
    if type(images) == str and type(save_folder) == str:
        if os.path.normpath(images) == os.path.normpath(save_folder):
            raise ValueError('images and save_folder can not be the same')
    if type(images) == str and os.path.isdir(images) == False:
        raise ValueError('images is of type str but it is not a directory')
    if type(save_folder) != str:
        raise TypeError('save_folder must be of type str')
    if os.path.isdir(save_folder) == False:
        raise ValueError('save_folder is not a directory') 
    if type(chan) != int:
        raise TypeError('chan must be of type int')
    if type(model_type) != str:
        raise TypeError('model_type must be of type str')
    if type(diameter) != int and diameter != None:
        raise TypeError('diameter must be of type int or None')
    if type(invert) != bool:
        raise TypeError('invert must be of type bool')
    if type(normalize) != bool:
        raise TypeError('normalize must be of type bool')
    if type(net_avg) != bool:
        raise TypeError('net_avg must be of type bool')
    if type(resample) != bool:
        raise TypeError('resample must be of type bool')
    if type(interp) != bool:
        raise TypeError('interp must be of type bool')
    if type(flow_threshold) != float and type(flow_threshold) != int:
        raise TypeError('flow_threshold must be of type flaot or int')
    if type(cellprob_threshold) != float:
        raise TypeError('cellprob_threshold must be of type flaot')
    if type(min_size) != int:
        raise TypeError('min_size must be of type flaot') 
    if type(rescale) != float and rescale != None:
        raise TypeError('rescale must be of type flaot or None')
    if type(stich_for_CP) != bool:
        raise TypeError('stich_for_CP must be of type bool')
    if stich_for_CP == True and type(stiched_image_tiles) != int:
        raise TypeError('stiched_image_tiles must be of type int if stich_for_CP==True')
    if stich_for_CP == True and type(eroded) != int:
        raise TypeError('eroded must be of type int if stich_for_CP==True')

                
    # check if images is a directory 
    if type(images) == str and os.path.isdir(images):
        tif_dir, out_dir = copy_directory_structure(images, save_folder, '.tif')
        
        # get full images/save paths 
        all_IN =[]
        all_OUT =[]
        for IN, OUT in zip(tif_dir, out_dir):
            file_base = [file for file in os.listdir(IN) if file.endswith('.tif')]
            all_IN += [os.path.join(IN,i) for i in file_base]
            all_OUT += [os.path.join(OUT,i) for i in file_base]
        if len(all_IN)==0:
            raise Exception('no .tif files in directory or subdirectory')
             
    # check if list of paths 
    elif type(images) == list and type(images[0]) == str:
        all_IN = [i for i in images if os.path.isfile(i)==True and i.endswith('.tif')]
        all_OUT = [os.path.join(save_folder,i) for i in all_IN]
        
        if len(all_IN)==0:
            raise Exception('no .tif files in images')
    
    else: # list of np.arrys
        all_IN = images
        all_OUT = [os.path.join(save_folder,'seg_image_'+str(i)+'.tif') for i in range(len(all_IN))]
      
    # setup cellpose
    CP_model = setup_cellpose(model_type)
    
    # segment images
    if stich_for_CP == False:
        for image, save_path in zip(all_IN,all_OUT):
        
            # check to see what form images is in
            if type(image) == str:
                image = skimage.io.imread(image)

            # fix images with shape (2048, 2048, 4) to (4, 2048, 2048) 
            seg_image = check_shape(image)
            
            # get the chan to segment 
            if len(seg_image.shape) > 2:
                seg_image = seg_image[chan,:,:]
            
            # run cellpose 
            segment, flows, styles, diams = CP_model.eval([seg_image],  
                                                          invert=invert,
                                                          normalize=normalize,
                                                          diameter=diameter,
                                                          net_avg=net_avg,
                                                          resample=resample,
                                                          interp=interp,
                                                          flow_threshold=flow_threshold,
                                                          cellprob_threshold=cellprob_threshold,
                                                          min_size=min_size,
                                                          rescale= rescale)
            #save segments 
            skimage.io.imsave(save_path,segment[0],check_contrast=False)
        
        # directory
        if type(images) == str:
            return(all_IN, all_OUT)
        # lsit of paths
        elif type(images)==list and type(images[0])==str:
            return(all_IN, all_OUT)
        # numpy arrys 
        else:
            return(all_OUT)
            
    if stich_for_CP == True:
        
        # check correct form of images 
        if type(all_IN[0]) != str:
            raise TypeError('if stich_for_CP == True, image type must be str (path to images or directory)')
        
        # link paths togeather
        path_dict = {}
        for IN, OUT in zip(all_IN, all_OUT):
            path_dict[IN] = OUT

        # stich, segment, upscal, save 
        for image_list in divide_list(all_IN, stiched_image_tiles**2):
            
            # tile images in smaller window
            tiled, image_cords = Stich(stiched_image_tiles, image_list, pad_fraction = 0.01 ,chan=chan, size = 2048)
            
            # correct for down sampling
            if type(diameter)==int:
                diameter = int(diameter/stiched_image_tiles)
            if type(min_size)==int:
                min_size = int(min_size/stiched_image_tiles)
            
            # run cellpose
            tiled_seg, flows, styles, diams = CP_model.eval([tiled],
                                                          invert=invert,
                                                          normalize=normalize,
                                                          diameter=diameter,
                                                          net_avg=net_avg,
                                                          resample=resample,
                                                          interp=interp,
                                                          flow_threshold=flow_threshold,
                                                          cellprob_threshold=cellprob_threshold,
                                                          min_size=min_size,
                                                          rescale= rescale)
    
            croped = crop_from_cord(tiled_seg[0], image_cords)
            
            # crop, upscale, save 
            for image_path in croped:
                small = croped[image_path]
                big = up_scal_segs(small,2048,eroded)
                skimage.io.imsave(path_dict[image_path],big,check_contrast=False) 
                
        return(all_IN, all_OUT)

    
