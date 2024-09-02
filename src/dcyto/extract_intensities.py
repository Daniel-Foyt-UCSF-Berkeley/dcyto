# extract_intensities                    
from .utils import apply_mask_extract
from .utils import get_unique_path
from .utils import DataFrame_to_fcs
import os
import multiprocessing.dummy as mp_dummy
import multiprocessing
import pandas as pd
import numpy as np

def extract_intensities(image_paths,
                        segment_paths,
                        chan_names,
                        save_path_name = None,
                        fcs = True,
                        threads = -1,
                        verbose = True):
    
        '''
        extract intensities from each channel for each segment 
        
        Args:
            image_paths (list): list of absolute paths to .tif images
                All images must have the '.tif' end or they will not be included.
                
            segment_paths (list): list of absolute paths to .tif images of segments
                All segmentaitons must have the '.tif' end or they will not be included.
                
            save_path_name (str or None, default None): paht and name of csv to save results. '.csv' will be
                included if not in path. 
                
            chan_names (list or None, default None): list of lables for columns in CSV corresponding
                to channel names. If None channels names channel0, channel1, channel(n)...
            
            fcs (bool, default True): save_path_name must not be None. Weather to create
                fcs files and save. If all images are in the same directory, fcs file will
                be named same as basename of save_path_name. else sample name will be used
                for each fcs whcih is derived from the image_paths 
                
            threads (int, None, default -1): number of cores/threads to use. If None, 
                run iterativly. if -1 run on all cores from multiprocessing.cpu_count()
                
            verbose (bool, default True): print thread count and where analysis is saved 
                
        Returns: Panda DataFrame: columns names:'cell_id' unique to image not dataset,
                 'segment_area', 'image_path', 'segment_path', 'edge_cell' 1(True) or 0(False),
                 channel_names(1)_sum, channel_names(2)_sum, channel_names(n)_sum...,
                 channel_names(1)_mean, channel_names(2)_mean, channel_names(n)_mean...,
                 'sample' derived from 'image_path'
                 
        '''
        if len(image_paths)!=len(segment_paths):
            raise Exception('len(image_paths)!=len(segment_path)')
        if type(image_paths)!= list:
            raise TypeError('image_paths must be of type list (paths to images)')
        if type(segment_paths)!= list:
            raise TypeError('segment_paths must be of type list (paths to segmentation)')
        if type(chan_names)!=list and chan_names!= None:
            raise TypeError('chan_names must be of type list or None')
        if type(save_path_name)!=str and save_path_name!=None:
            raise TypeError('save_path_name must be of type str or None')
        if save_path_name!=None and os.path.isdir(os.path.dirname(save_path_name)) == False:
            raise ValueError(os.path.dirname(save_path_name) + ' is not a directory')
        if type(threads)!= int and threads!=None:
            raise TypeError('threads must be of type int or None')
        if threads == 0 or threads > multiprocessing.cpu_count(): 
            raise ValueError('threads must be an int or None. if int, 0<threads<='+str(multiprocessing.cpu_count())+' or -1')
        if threads < 0 and threads!= -1:
            raise ValueError('threads must be an int or None. if int, 0<threads<='+str(multiprocessing.cpu_count())+' or -1')
        if type(fcs)!= bool:
            raise TypeError('fcs must be of type bool')
        if fcs ==True and save_path_name == None:
            raise ValueError('if fcs is True, save_path_name must be a path to a directory and not None')
            
            
        segment_path_good = []
        image_paths_good = []
        
        # check for mask
        for image_path, segment_path in zip(image_paths, segment_paths):
            if os.path.exists(image_path) == True and os.path.exists(segment_path):
                if image_path.endswith('.tif') and image_path.endswith('.tif'):
                    segment_path_good.append(segment_path)
                    image_paths_good.append(image_path)
                    
        if len(segment_path_good)==0:
            print('no paths with mask to extract')
            return(None)
        
        cell_ids = []
        cell_sums = []
        cell_area = []
        image_PATHS = [] 
        segment_PATHS = []
        cell_edge = []
        
        # run on multiple cores 
        if threads != None:
            
            # set up threads/cores 
            if threads == -1: # run on all cores
                threads = multiprocessing.cpu_count()
                
            threads = min(multiprocessing.cpu_count(),threads)
                
            if verbose == True:
                print('running analysis on '+str(threads) +' cores')
                
            pool = mp_dummy.Pool(threads)
                
            # run calulations on mulitiple cores 
            results = pool.starmap(apply_mask_extract,
                                   zip(image_paths_good,segment_path_good),
                                   None)
            pool.close() 
            pool.join()           
                
            # unpack results 
            for im_result,image_path, segment_path in zip(results,image_paths_good,segment_path_good):
                ID, sums, areas, edge = im_result
                image_PATHS+=[image_path]*len(ID)
                segment_PATHS+=[segment_path]*len(ID)
                cell_ids+=ID
                cell_sums+=sums
                cell_area+=areas
                cell_edge+=edge
            
        if threads == None:
            if verbose == True:
                print('running analysis on iteratvly')
                
            for image_path, segment_path in zip(image_paths_good, segment_path_good):
                ID, sums, areas, edge = apply_mask_extract(image_path, segment_path)
                image_PATHS+=[image_path]*len(ID)
                segment_PATHS+=[segment_path]*len(ID)
                cell_ids+=ID
                cell_sums+=sums
                cell_area+=areas
                cell_edge+=edge
            
        # make data structure for dataframe
        data = {'cell_id':cell_ids,
                'segment_area':cell_area,
                'image_path':image_PATHS,
                'segment_path':segment_PATHS,
                'edge_cell':cell_edge}
        
        # find max sums to know number of columns
        chan_nums = max([len(x) for x in cell_sums])
        
        
        # unpack sums data 
        sums_all = []
        for chan in range(chan_nums):
            sums_all.append([])
            
        if chan_names==None:
            chan_names = []
            
        chan_names_full = []
        for chan in range(chan_nums):
            # catch nonnamed channels 
            if chan+1 > len(chan_names):
                chan_names_full.append('channel'+str(chan))
            else:
                chan_names_full.append(chan_names[chan])
            
        for cell in cell_sums:
            for i,chan in enumerate(chan_names_full):
                # catch channels not there
                if (len(cell)-1)<i:
                    SUM = np.nan
                else:
                    SUM = cell[i]
                sums_all[i].append(SUM)
        
        chan_names_sums = []
        # add sums to DF
        for i,chan in enumerate(chan_names_full):
            data[chan+'_sum'] = sums_all[i]
            chan_names_sums.append(chan+'_sum')
        
        # add means to DF
        for chan in chan_names_sums:
            mean = np.array(data[chan])/np.array(data['segment_area'])
            data[chan[:chan.find('_sum')]+'_mean'] = mean.tolist()
        
        # add column for sample_ID
        data['sample'] = get_unique_path(data['image_path'])
        
        # create datafram 
        DF = pd.DataFrame(data = data, columns =list(data.keys()))
        
        # save datafram as CSV
        if save_path_name != None:
            # add .csv if not in file name
            if save_path_name.endswith('.csv') != True:
                save_path_name+='.csv'
            DF.to_csv(save_path_name, index = False)
            
            if verbose == True:
                print('csv at '+save_path_name)
        
        if fcs == True:
            if len(np.unique(DF['sample']))==1:
                save_path_name = save_path_name[:-4]+'.fcs' # .fcs to path
            else:
                save_path_name = os.path.dirname(save_path_name) # directory path 
            DataFrame_to_fcs(DF, save_path_name, verbose = verbose)
         
        return(DF)
