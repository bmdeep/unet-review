import os
import h5py
import numpy as np
from PIL import Image
from utils import write_hdf5
import glob
import time


def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test, Nimgs, channels, height, width, verbose=True):
    
    imgs = np.empty((Nimgs,height,width,channels))
    gts = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            if verbose:
                print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            if verbose:
                print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            gts[i] = np.asarray(g_truth)
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                if verbose:
                    print("specify if train or test!!")
                exit()
            if verbose:
                print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    if verbose:
        print("imgs max: "+str(np.max(imgs)))
        print("imgs min: "+str(np.min(imgs)))
    assert(np.max(gts)==255 and np.max(border_masks)==255)
    assert(np.min(gts)==0   and np.min(border_masks)==0)
    if verbose:
        print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    gts = np.reshape(gts,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(gts.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    
    return imgs, gts, border_masks


def prepare_dataset(
    # working dirs
    save_data_dir, original_base_dir,
    # configs
    Nimgs, channels, height, width,
    # file names
    imgs_train_file_name=None, msks_train_file_name=None, bmsks_train_file_name=None, 
    imgs_test_file_name =None, msks_test_file_name =None, bmsks_test_file_name =None,
    verbose = True
):
    start_time = time.time()
    
    if not imgs_train_file_name : imgs_train_file_name  = "DRIVE_dataset_imgs_train" 
    if not msks_train_file_name : msks_train_file_name  = "DRIVE_dataset_msks_train" 
    if not bmsks_train_file_name: bmsks_train_file_name = "DRIVE_dataset_bmsks_train" 
    if not imgs_test_file_name  : imgs_test_file_name   = "DRIVE_dataset_imgs_test" 
    if not msks_test_file_name  : msks_test_file_name   = "DRIVE_dataset_msks_test" 
    if not bmsks_test_file_name : bmsks_test_file_name  = "DRIVE_dataset_bmsks_test"
    
    #------------Path of the images -----------------------------
    #train
    original_imgs_train_dir = f"{original_base_dir}/training/images/"
    groundTruth_imgs_train_dir = f"{original_base_dir}/training/1st_manual/"
    borderMasks_imgs_train_dir = f"{original_base_dir}/training/mask/"
    #test
    original_imgs_test_dir = f"{original_base_dir}/test/images/"
    groundTruth_imgs_test_dir = f"{original_base_dir}/test/1st_manual/"
    borderMasks_imgs_test_dir = f"{original_base_dir}/test/mask/"
    #------------------------------------------------------------
    
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    #getting the training datasets
    imgs_train, msks_train, border_masks_train = get_datasets(
        original_imgs_train_dir,
        groundTruth_imgs_train_dir,
        borderMasks_imgs_train_dir,
        "train",
        Nimgs,channels,height,width,
        verbose
    )
    if verbose:
        print("saving train datasets...")
    write_hdf5(imgs_train, f"{save_data_dir}/{imgs_train_file_name}.hdf5")
    write_hdf5(msks_train, f"{save_data_dir}/{msks_train_file_name}.hdf5")
    write_hdf5(border_masks_train,f"{save_data_dir}/{bmsks_train_file_name}.hdf5")

    #getting the testing datasets
    imgs_test, msks_test, border_masks_test = get_datasets(
        original_imgs_test_dir,
        groundTruth_imgs_test_dir,
        borderMasks_imgs_test_dir,
        "test",
        Nimgs,channels,height,width,
        verbose
    )
    if verbose:
        print("saving test datasets...")
    write_hdf5(imgs_test, f"{save_data_dir}/{imgs_test_file_name}.hdf5")
    write_hdf5(msks_test, f"{save_data_dir}/{msks_test_file_name}.hdf5")
    write_hdf5(border_masks_test,f"{save_data_dir}/{bmsks_test_file_name}.hdf5")
    
    end_time = time.time()
    if verbose:
        print(f"preparing dataset is finished and took {end_time-start_time:0.3}s.")
        
def prepare_dataset_mode(
    # working dirs
    save_data_dir, original_base_dir,
    # configs
    height, width, channels=3,
    # file names
    imgs_file_name=None, msks_file_name=None, bmsks_file_name=None, 
    mode = 'train',
    verbose = True
):
    start_time = time.time()

    if not mode in ['train', 'test']:
        raise ValueError("the value of <mode> variable is wrong! you should choose 'train' or 'test'.")

    mode_name = {'train': 'training', 'test': 'test'}
    
    if not imgs_file_name : imgs_file_name  = f"DRIVE_dataset_imgs_{mode}" 
    if not msks_file_name : msks_file_name  = f"DRIVE_dataset_msks_{mode}" 
    if not bmsks_file_name: bmsks_file_name = f"DRIVE_dataset_bmsks_{mode}" 
    
    #------------Path of the images -----------------------------
    original_imgs_dir = f"{original_base_dir}/{mode_name[mode]}/images/"
    groundTruth_imgs_dir = f"{original_base_dir}/{mode_name[mode]}/1st_manual/"
    borderMasks_imgs_dir = f"{original_base_dir}/{mode_name[mode]}/mask/"
    #------------------------------------------------------------

    Nimgs = len(glob.glob(f"{original_imgs_dir}*.tif"))
    
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # getting the dataset
    imgs, msks, border_masks = get_datasets(
        original_imgs_dir,
        groundTruth_imgs_dir,
        borderMasks_imgs_dir,
        mode,
        Nimgs,channels,height,width,
        verbose
    )
    if verbose:
        print(f"saving {mode} datasets...")
    write_hdf5(imgs, f"{save_data_dir}/{imgs_file_name}.hdf5")
    write_hdf5(msks, f"{save_data_dir}/{msks_file_name}.hdf5")
    write_hdf5(border_masks,f"{save_data_dir}/{bmsks_file_name}.hdf5")

    end_time = time.time()
    if verbose:
        print(f"preparing dataset is finished and took {end_time-start_time:0.3}s.")