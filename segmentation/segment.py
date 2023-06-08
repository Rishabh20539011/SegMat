import numpy as np
import cv2
import os
import torch



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device=get_default_device()

class Segmentator:
    
    def __init__(self):
        pass
        

    def segment(self, model, img, threshold_value=0.5, device=str(device)):
        a_img = img.copy()
        shape = img.shape[:-1]
        img = cv2.resize(img, (512,512))
        img = img[...,::-1].copy()

        img = img[np.newaxis, ...]
        img = torch.from_numpy(img)

        if str(device) == "cuda":
            img = img.permute([0, 3, 1, 2]).to("cuda")
        else:
            img = img.permute([0, 3, 1, 2])
        # img = img.to("cuda")
        with torch.no_grad():
            model.eval()
            logits = model(img)
        if str(device) == "cuda":
            pr_masks = logits.sigmoid().cpu()
        else:
            pr_masks = logits.sigmoid()

        mask = pr_masks[0].permute([1,2,0])
        mask = cv2.resize(np.array(mask), (shape[1], shape[0]))
        (T, mask) = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
        mask = mask[..., np.newaxis]
        mask = mask.astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours, mask

    
    # Function that returns output for welding defect 
    def weld_segment(self, model, img, threshold_value, device=device):
        contours, mask, a_img = self.segment( model, img, threshold_value, device=device)
        cv2.drawContours(a_img, contours, -1, (0, 255, 0), 3)
        return a_img, mask

    def weld_batch_process(self, model, imgs, threshold_value, device=device):
        # Saving a copy to draw the final result on it
        processed_imgs = []
        actual_imgs = []
        shapes = []
        for img in imgs:
            actual_imgs.append(img.copy())
            img = self.boost_contrast(img)
            shapes.append(img.shape[:-1])
            # Reshaping and preprocessing for passing to the model
            img = cv2.resize(img, (512,512))
            processed_imgs.append(img)
        # img = img[np.newaxis, ...]
        processed_imgs = np.array(processed_imgs)
        img = torch.from_numpy(processed_imgs)
        # print("Time for preprocessing the image: ", time.time()-before)
        img = img.permute([0, 3, 1, 2]).to(device)
        # Getting the prediction
        # before = time.time()
        with torch.no_grad():
            logits = model(img)
        # print("Time for segmentation model:", time.time() - before)
        # Post processing to get the segmentation mask
        # before = time.time()
        pr_masks = logits.sigmoid().cpu()
        masks = []
        for mask, actual_img, shape in zip(pr_masks, actual_imgs, shapes):
            mask = mask.permute([1,2,0])
            mask = cv2.resize(np.array(mask), (shape[1], shape[0]))
            (T, mask) = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
            mask = mask[..., np.newaxis]
            mask = mask.astype("uint8")
            masks.append(mask)
            # Drawing the contours on actual image
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(actual_img, contours, -1, (0, 0, 255), 3)
        # print("Time for postprocessing the image: ", time.time()-before)
        return actual_imgs, masks


    def boost_contrast(self, img):
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        # mergnig
        limg = cv2.merge((cl,a,b))
        # Converting to bgr
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img

    # Function that returns output for part segmentations
    def parts_segment(self, model, img, threshold, device=device):

        contours, mask= self.segment( model, img, threshold, device=device)

        # mask= self.segment( model, img, threshold, device=device)

        contours = sorted(contours, key=len, reverse=True)
        # new_mask = np.zeros(mask.shape)
        # final = cv2.fillPoly(new_mask, pts=[contours[0]], color=(255, 255, 255))
        # a_img[final[:,:,0]==0] = 0
        # return contours, a_img, final
        return contours,mask



