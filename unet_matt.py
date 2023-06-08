import numpy as np
import cv2
import os
import torch
import time
import sys
from segmentation.segment import Segmentator
from segmentation.model import PetModel
# from segmat import segmating
from PIL import Image
from matting.generator import TrimapGenerator
from matting.fba_matting.models import FBA
from matting.fba_mat import FBAMatting
from matting.utils.mask_utils import apply_mask


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')



device=get_default_device()

print('device_name------------',str(device))

try:


    model_unet = PetModel(
        "UnetPlusPlus", "efficientnet-b5", in_channels=3, out_classes=1,decoder_attention_type='scse')
    # model_unet.to(str(device))
    # Loading the saved model weights
    model_unet.load_state_dict(torch.load('/home/rishabh/frinks/unet_with_matting/molbio_channel_poc_final/tablet_model_with_attention150.pth',map_location=device))
    # model_unet.to(device)
    model_unet.eval()


    segmentator = Segmentator()


    matting_model=FBA(encoder="resnet50_GN_WS")
    matting_model.load_state_dict(torch.load('/home/rishabh/frinks/unet_with_matting/molbio_channel_poc_final/fba_matting.pth', map_location=device))
    matting_model.to(device)
    matting_model.eval()


    fba = FBAMatting(device=device,
                    input_tensor_size=2048,
                    batch_size=1,
                    model=matting_model)
    

    trimap_gen=TrimapGenerator(prob_threshold=230,kernel_size=1,erosion_iters=3)

    print("INFO: Models loaded successfully")


except Exception as e:
    print(e)
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    print("INFO Error while model loading")
    exit()





# matting=True

def matted_img(frame,matting=True):

# for name in sorted(os.listdir(img_dir)):

    # frame=cv2.imread('/home/rishabh/Music/New folder/1671095563907.bmp')
    # frame=cv2.imread(img_dir+'/'+name)

    # frame = frame[590:1283, 865:1755]
    # cv2.imwrite('frame.bmp',frame)

    contours,final = segmentator.parts_segment(
        model=model_unet, img=frame, threshold=float(0.3))
    # Cropping the segmented image to fit the matrix
    final=final[:,:,0]
    model_mask=cv2.merge([final,final,final])

    res_img=frame.copy()

    # print(final.shape)
    # print(model_mask.shape)

    if matting :

        trimap=trimap_gen(final)

        # cv2.imwrite(f'mask_{name}',trimap)

        alpha_img=fba(res_img,trimap)

        result_mask = np.array(alpha_img)

        final_image=apply_mask(res_img, mask=result_mask, device=str(device))

        res_img=final_image[:,:,:3].copy()

        # print('rgb_shape',final_image.shape)
        res_img[final_image[:,:,-1] == 0] = [0,0,0]

        # cv2.imwrite(f'{name}',res_img)

    else:
        res_img[model_mask==0]=0
        # cv2.imwrite(f'{name}',res_img)
        
        # print('rgb_shape',res_img.shape)


    return res_img


img_dir='/home/rishabh/Downloads/pills/test'
save_dir='/home/rishabh/Downloads/pills/results'

for img in os.listdir(img_dir):

    image=cv2.imread(img_dir+'/'+img)

    res_img=matted_img(image,matting=True)

    cv2.imwrite(save_dir+'/'+img,res_img)


    # image_mat=[Image.fromarray(np.uint8(res_img))]

    # mask_mat=[Image.fromarray(np.uint8(model_mask))]



    # contours, _ = cv2.findContours(model_mask[:,:,0], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # if len(contours)>0:
    #     cv2.drawContours(res_img, contours, -1, (0,255,0), 1)
    # else:
    #     pass









