import os
import h5py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import imageio
import cv2
import numpy as np
from model.main_model_old import supv_main_model as main_model
from dataset.AVE_dataset import AVEDataset


def video_frame_sample(frame_interval, video_length, sample_num):

    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num


def normlize(x, min = 0, max = 255):

    num, row, col = x.shape
    for i in range(num):
        xi = x[i, :, :]
        xi = max *(xi - np.min(xi))/(np.max(xi) - np.min(xi))
        x[i, :, :] = xi
    return x


def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.3):
    print(np.max(im_cloud))

    im_cloud[:, :, 1] = 0
    im_cloud[:, :, 2] = 0
    return (a1*im_map + a2*im_cloud).astype(np.uint8)


if __name__ == '__main__':
    config_path = 'configs/main.json'
    with open(config_path) as fp:
        config = json.load(fp)

    # features, labels, and testing set list
    dir_video  = "./data/visual_feature.h5"
    dir_audio  = "./data/audio_feature.h5"
    dir_labels = "./data/labels.h5"
    dir_order_test = "./data/test_order.h5"

    # access to original videos for extracting video frames
    raw_video_dir = "../../Dataset/AVE_Dataset/AVE" # videos in AVE dataset
    lis = os.listdir(raw_video_dir)
    f = open("../../Dataset/AVE_Dataset/Annotations.txt", 'r')
    dataset = f.readlines() 
    len_data = len(dataset)
    print("The dataset contains %d samples" % len_data)
    f.close()
    with h5py.File(dir_order_test, 'r') as hf:
        test_order = hf['order'][:]

    # pre-trained models
    # ---> CMGAM
    # mainModel = main_model(config['model']).cuda()
    # ckpt = torch.load('./Exps/Supv/final/model_epoch_43_top1_80.199_task_Supervised_best_model.pth.tar')
    # mainModel.load_state_dict(ckpt, strict=False)
    # mainModel.eval()
    # mainModel.double()
    # att_layer = mainModel.visual_spatial_channel_att._modules.get('affine_v_s_att')
    # att_layer = mainModel.visual_spatial_channel_att._modules.get('affine_channel_att')
    # <---
        
    # ---> CMBS
    mainModel = main_model(config['model']).cuda()
    ckpt = torch.load('./Exps/Supv/CMBS_Supervised_model.pth.tar')
    mainModel.load_state_dict(ckpt, strict=False)
    mainModel.eval()
    mainModel.double()
    att_layer = mainModel.spatial_channel_att._modules.get('affine_v_c_att')
    print("att_layer: ", att_layer)
    # <---
        
    # load testing set
    dataloader = DataLoader(
        AVEDataset('./data/', split='test'),
        batch_size=402,
        shuffle=False,
        num_workers=0,
        pin_memory=False 
    )

    # generate attention maps
    att_map = torch.zeros((4020, 7 * 7, 1))

    visual_inputs, audio_inputs , labels = iter(dataloader).__next__()
    visual_inputs = visual_inputs.cuda()
    audio_inputs =  audio_inputs.cuda()
    labels = labels.numpy()
    batch_size, t_size, w, h, dim_v = visual_inputs.shape
    print('visual_inputs: ', visual_inputs.shape) # ([b, 10, 7, 7, 512])
    print('audio_inputs: ', audio_inputs.shape) # ([b, 10, 128])
    print('labels: ', labels.shape)     

    map = att_layer.register_forward_hook(lambda m, i, o: att_map.copy_(o.data))
    h_x = mainModel(visual_inputs, audio_inputs)
    map.remove()
    print("att_map: ", att_map.shape) # ([b*10, 49, 1])

    z_t = att_map.squeeze(2)
    alpha_t = F.softmax( z_t, dim = -1 ).view( z_t.size( 0 ), -1, z_t.size( 1 ) )
    att_weight = alpha_t.view(402, t_size, w, h).cpu().data.numpy() # attention maps of all testing samples
    print("att_weight: ", att_weight.shape) # (402, 10, 7, 7)

    c = 0
    t = 10
    sample_num = 16
    extract_frames = np.zeros((sample_num * t, 224, 224, 3))
    save_dir = "visual_att/attention_maps_Channel/"
    original_dir = "visual_att/original/"

    for num in range(len(test_order)):
        print("test_order: ", num)
        data = dataset[test_order[num]]
        x = data.split("&")

        # extract video frames
        video_index = os.path.join(raw_video_dir, x[1] + '.mp4')
        vid = imageio.get_reader(video_index, 'ffmpeg')
        vid_len = len(vid)
        frame_interval = int(vid_len / t)

        frame_num = video_frame_sample(frame_interval, t, sample_num)
        imgs = []

        for i, im in enumerate(vid):
            x_im = cv2.resize(im, (224, 224))
            imgs.append(x_im)

        vid.close()
        cc = 0
        for n in frame_num:
            extract_frames[cc, :, :, :] = imgs[n]
            cc += 1

        # process generated attention maps
        att = att_weight[num, :, :, :]
        att = normlize(att, 0, 255)
        att_scaled = np.zeros((10, 224, 224))
        for k in range(att.shape[0]):
            att_scaled[k, :, :] = cv2.resize(att[k, :, :], (224, 224)) # scaling attention maps 
    
        att_t = np.repeat(att_scaled, 16, axis = 0) # 1-sec segment only has 1 attention map. Here, repeat 16 times to generate 16 maps for a 1-sec video
        heat_maps = np.repeat(att_t.reshape(160, 224, 224, 1), 3, axis = -1)
        c += 1

        att_dir = save_dir + x[1]
        ori_dir =  original_dir + x[1]
        if not os.path.exists(att_dir):
          os.makedirs(att_dir)
        if not os.path.exists(ori_dir):
          os.makedirs(ori_dir)

        for idx in range(160):
            heat_map = heat_maps[idx, :, :, 0]
            im = extract_frames[idx, :, :, :]
            im = im[:, :, (2, 1, 0)]
            heatmap = cv2.applyColorMap(np.uint8(heat_map), cv2.COLORMAP_JET)

            att_frame = heatmap * 0.2 + np.uint8(im) * 0.6
            n = "%04d" % idx
            vid_index = os.path.join(att_dir, 'pic' + n + '.jpg')
            cv2.imwrite(vid_index, att_frame)
            # ori_frame = np.uint8(im)
            # ori_index = os.path.join(ori_dir, 'ori' + n + '.jpg')
            # cv2.imwrite(ori_index, ori_frame)

        
        







