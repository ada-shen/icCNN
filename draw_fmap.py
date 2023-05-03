import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import shutil
from PIL import Image
import h5py
import argparse

def channel_max_min_whole(f_map):
    T, C, H, W = f_map.shape
    max_v = np.max(f_map,axis=(0,2,3),keepdims=True)
    min_v = np.min(f_map,axis=(0,2,3),keepdims=True)
    print(max_v.shape,min_v.shape)
    return (f_map - min_v)/(max_v - min_v + 1e-6)

def self_max_min(f_map):
    if np.max(f_map) - np.mean(f_map) != 0:
        return (f_map-np.min(f_map))/(np.max(f_map)-np.mean(f_map))*255.0
    else:
        return (f_map-np.min(f_map))/(np.max(f_map)-np.mean(f_map)+1e-5)*255.0

def get_file_path(path):
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            paths.append(os.path.join(root,file))
    return paths

# every channel
def draw_fmap_from_npz(data, save_dir,SHOW_NUM,save_channel):
    N, C, H, W = data.shape
    print('data shape:', data.shape)
    for i in range(N):
        if i in SHOW_NUM:
            for j in save_channel:#range(10):
                print("-----")
                print(i)
                print(j)
                fig = data[i,j]
                fig = cv.resize(fig,(112,112))
                # to visualize more clear, do max min norm
                # fig = self_max_min(fig)
                print(i,j)
                cv.imwrite(save_dir + 'sample'+str(i) + '_channel'+str(j) + '.bmp', fig*255.0)

# mean
def draw_fmap_from_npz_mean(data, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)

    T, C, H, W = data.shape

    for i in range(T):

        mean_f_map = data[i].sum(axis=0)/C
        mean_f_map = cv.resize(mean_f_map,(112,112))
        # to visualize more clear, do max min norm
        mean_f_map = self_max_min(mean_f_map)
        cv.imwrite(save_dir + 'voc_'+str(i) + '_mean' + '.bmp', mean_f_map)

def addTransparency(img, factor = 0.3):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0,0,0,0))
    img = Image.blend(img_blender, img, factor)
    return img

def put_mask(img_path,mask_path,output_fold,Th,factor):
    img = Image.open(img_path)
    img = addTransparency(img, factor)
    mask_img = cv.resize(cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR),(224,224))
    print('----')
    print(img_path)
    print(mask_path)
    ori_img = cv.resize(cv.imread(img_path),(224,224))


    zeros_mask = cv.resize(cv.imread(mask_path),(224,224))
    mask_for_red = np.zeros((224,224))
    # mask_for_red = pct_max_min(zeros_mask,Th)
    for i in range(zeros_mask.shape[0]):
        for j in range(zeros_mask.shape[1]):
            if np.sum((zeros_mask[i][j]/255.0)>Th): # vgg/cub 0.5 # VOC animal 0.5
                mask_for_red[i][j] = 1
                mask_img[i][j] = ori_img[i][j]
            else:
                mask_for_red[i][j] = 0
    red = np.zeros((224,224))
    for i in range(mask_for_red.shape[0]):
        for j in range(mask_for_red.shape[1]):
            if j > 2 and mask_for_red[i][j-1] == 0 and mask_for_red[i][j] == 1:
                red[i][j] = 1
                red[i][j-1] = 1
                red[i][j-2] = 1
                red[i][j-3] = 1
                if j < (mask_for_red.shape[1]-2):
                    red[i][j+1] = 1
                    red[i][j+2] = 1
                    #red[i][j+3] = 1
            if j < (mask_for_red.shape[1]-3) and mask_for_red[i][j] == 1 and mask_for_red[i][j+1] == 0:
                red[i][j] = 1
                if j > 1:
                    red[i][j-1] = 1
                    red[i][j-2] = 1
                    #red[i][j-3] = 1
                red[i][j+1] = 1
                red[i][j+2] = 1
                red[i][j+3] = 1
            if i > 2 and mask_for_red[i-1][j] == 0 and mask_for_red[i][j] == 1:
                red[i-1][j] = 1
                red[i-2][j] = 1
                red[i-3][j] = 1
                red[i][j] = 1
                if i < (mask_for_red.shape[0]-2):
                    red[i+1][j] = 1
                    red[i+2][j] = 1
                    #red[i+3][j] = 1
            if i < (mask_for_red.shape[0]-3) and mask_for_red[i][j] == 1 and mask_for_red[i+1][j] == 0:
                if i > 1:
                    red[i-1][j] = 1
                    red[i-2][j] = 1
                    #red[i-3][j] = 1
                red[i][j] = 1
                red[i+1][j] = 1
                red[i+2][j] = 1
                red[i+3][j] = 1


    for i in range(mask_for_red.shape[0]):
        for j in range(mask_for_red.shape[1]):
            if red[i][j] == 1:
                mask_img[i][j] = [0,0,255]
    return mask_img

# image add mask
def image_add_mask(show_num,image_dir,mask_dir,save_dir,save_channel,factor,animal,show_num_per_center):
    for i in show_num:
        if animal == 'bird':
            image_paths = image_dir + 'vocbird_' + str(i) + '.jpg'
        else:
            image_paths = image_dir + str(i) + '.jpg'
        for j,channel in enumerate(save_channel):
            mask_path = mask_dir + 'sample'+str(i) + '_channel'+ str(channel) + '.bmp'
            mask_img = put_mask(img_path = image_paths,mask_path=mask_path,output_fold=save_dir,Th=Th,factor=factor)
            mask_img = cv.resize(mask_img,(112,112))
            cv.imwrite(os.path.join(save_dir+'factor'+str(factor)+'_Th'+str(Th)+'_sample'+str(i)+'_center'+str(j//show_num_per_center)+'_channel'+str(channel)+'.bmp'), mask_img)


# randomly shuffle feature maps of N samples
def permute_fmaps_N(data,file_name):
    N,_,_,_ = data.shape
    permute_idx = np.random.permutation(np.arange(N))
    data = data[permute_idx,...]
    print (data.shape,data.dtype)
    np.savez(file_name + '_pert', f_map = data)

def get_cluster(matrix):
    cluser = []
    visited = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        tmp = []
        if(visited[i]==0):
            for j in range(matrix.shape[1]):
                if(matrix[i][j]==1 ):
                    tmp.append(j)
                    visited[j]=1;
            cluser.append(tmp)
    for i,channels in enumerate(cluser):
        print('Group',i,'contains',len(channels),'channels.')
    return cluser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()      # add positional arguments

    parser.add_argument('-Th', type=int, default=0.2)
    parser.add_argument('-factor', type=int, default=0.5)
    parser.add_argument('-show_num', type=int, default=10)
    parser.add_argument('-model', type=str)
    parser.add_argument('-animal', type=str)
    parser.add_argument('-fmap_path', type=str)
    parser.add_argument('-loss_path', type=str)
    parser.add_argument('-folder_name', default=None, type=str)

    args = parser.parse_args()


    # fixed
    Th = args.Th # >Th --> in the red circle
    factor = args.factor # the smaller the factor, the darker the area outside the red circle
    animals = ['bird','cat','dog','cow','horse','sheep','cub', 'celeba']
    show_num_per_center = args.show_num
    file_path = args.fmap_path
    # the No. of sample to visualize; the No. starts from 0
    # The id of images that you want to visualize
    voc = [
            [1],#voc_bird
            [1],#voc_cat
            [1],#voc_dog
            [1],#voc_cow
            [1],#voc_horse
            [1],#voc_sheep
            [1],#cub
            [1] #celeba 
          ]

    # if args.loss_path == None:
    #     cluster_label = [[],[],[],[],[]]
    #     cluster_label[0] = np.array(range(100))
    #     cluster_label[1] = np.array(range(100,200))
    #     cluster_label[2] = np.array(range(200,300))
    #     cluster_label[3] = np.array(range(300,400))
    #     cluster_label[4] = np.array(range(400,512))
    # else:
    loss = np.load(args.loss_path)
    gt = loss['gt'][-1] # show channel id of different groups
    cluster_label = get_cluster(gt)
    print ('groups and channels', cluster_label)



    animal_id =  animals.index(args.animal)# 0~5; which category you want to draw feature maps
    save_channel = []
    for i in range(len(cluster_label)):
        for j in range(show_num_per_center):
            save_channel.append(cluster_label[i][j])

    if args.folder_name == None:
        model_name = args.model+'_'+args.animal
    else:
	    model_name = args.folder_name

    SHOW_NUM = voc[animal_id]
    animal = animals[animal_id]
    # load data
    data = np.load(file_path)['f_map']
    print('data shape:', data.shape,data.dtype) # verify the data.shape, e.g. bird category has 421 samples
    # channel normalization
    data = channel_max_min_whole(data) #

    save_dir='./fmap/'+model_name+'/'+animal+'/'
    #
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)
    # draw feature map and save feature maps
    draw_fmap_from_npz(data,save_dir=save_dir,SHOW_NUM=SHOW_NUM,save_channel=save_channel)   #############iccnn
    # draw_fmap_from_npz_mean(data, save_dir=save_dir)

    if args.animal == 'cub':
	    img_dir = './images/hook_cub_test/'
    elif args.animal == 'celeba':
	    img_dir = './images/hook_celeba_test/'
    else:
	    img_dir =  './images/voc'+animal+'_test/'
    mask_dir = './fmap/'+model_name+'/'+animal+'/'   # i.e. the dir of feature maps (same with the 'save_dir' above)
    masked_save_dir = './fmap/'+model_name+'/'+animal+'_masked/' # save dir of images with the red circle we want!
    if os.path.exists(masked_save_dir):
        shutil.rmtree(masked_save_dir)
        os.makedirs(masked_save_dir)
    else:
        os.makedirs(masked_save_dir)
    image_add_mask(show_num=SHOW_NUM,image_dir=img_dir,mask_dir=mask_dir,save_dir=masked_save_dir,save_channel=save_channel,factor=factor,animal=animal,show_num_per_center=show_num_per_center)
