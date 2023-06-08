import pathlib
from typing import Union, List, Tuple

import PIL
import cv2
import numpy as np
import torch
from PIL import Image

# from carvekit.ml.arch.fba_matting.models import FBA
from matting.utils.transforms import (
    trimap_transform,
    groupnorm_normalise_image,
)
# from carvekit.ml.files.models_loc import fba_pretrained
from matting.utils.image_utils import convert_image, load_image
# from matting.utils.models_utils import get_precision_autocast, cast_network
from matting.utils.pool_utils import batch_generator, thread_pool_processing

import time
import os

__all__ = ["FBAMatting"]



class FBAMatting():
    """054
    FBA Matting Neural Network to improve edges on image.
    """

    def __init__(
        self,
        device="cpu",
        input_tensor_size: Union[List[int], int] = 512,
        batch_size: int = 1,
        # encoder="resnet50_GN_WS",
        model=None,
        fp16: bool = True,
    ):
        """
        Initialize the FBAMatting model

        Args:
            device: processing device
            input_tensor_size: input image size
            batch_size: the number of images that the neural network processes in one run
            encoder: neural network encoder head
            load_pretrained: loading pretrained model
            fp16: use half precision

        """
        super(FBAMatting, self).__init__()
        self.fp16 = fp16
        self.device = device
        self.batch_size = batch_size
        self.input_tensor_size=input_tensor_size
        if isinstance(input_tensor_size, list):
            self.input_image_size = input_tensor_size[:2]
        else:
            self.input_image_size = (input_tensor_size, input_tensor_size)

        self.model=model

        # if load_pretrained:
        #     self.load_state_dict(torch.load(os.getenv('matting_model_path'), map_location=self.device))
        # self.eval()

    def data_preprocessing(
        self, data) ->torch.FloatTensor:
        """
        Transform input image to suitable data format for neural network

        Args:
            data: input image

        Returns:
            input for neural network

        """
        resized = data.copy()
        # if self.batch_size == 1:
        #     resized.thumbnail(self.input_image_size, resample=3)
        # else:
        #     resized = resized.resize(self.input_image_size, resample=3)

        # noinspection PyTypeChecker

        if self.input_tensor_size>512:
            resized= cv2.resize(resized,self.input_image_size,cv2.INTER_LINEAR)
            print('1',resized.shape)

        else:
            resized= cv2.resize(resized,self.input_image_size,cv2.INTER_AREA)
            print('2',resized.shape)


        image = np.array(resized, dtype=np.float64)
        image = image / 255.0  # Normalize image to [0, 1] values range

        if len(resized.shape) == 3:
            image = image[:, :, ::-1]
        elif len(resized.shape) == 2:
            image2 = np.copy(image)
            h, w = image2.shape
            image = np.zeros((h, w, 2))  # Transform trimap to binary data format
            image[image2 == 1, 1] = 1
            image[image2 == 0, 0] = 1
        else:
            raise ValueError("Incorrect color mode for image")
        h, w = image.shape[:2]  # Scale input mlt to 8
        h1 = int(np.ceil(1.0 * h / 8) * 8)
        w1 = int(np.ceil(1.0 * w / 8) * 8)
        x_scale = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        image_tensor = torch.from_numpy(x_scale).permute(2, 0, 1)[None, :, :, :].float()
        print('3',resized.shape)

        if len(resized.shape) == 3:
            return (image_tensor, groupnorm_normalise_image(image_tensor.clone(), format="nchw"))
        else:

            return (image_tensor,torch.from_numpy(trimap_transform(x_scale)).permute(2, 0, 1)[None, :, :, :].float())


    @staticmethod
    def data_postprocessing(
        data: torch.tensor, trimap:Union[PIL.Image.Image,np.ndarray]
    ):
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data: output data from neural network
            trimap: Map with the area we need to refine

        Returns:
            Segmentation mask as PIL Image instance

        """
        # if trimap.mode != "L":
            # raise ValueError("Incorrect color mode for trimap")

        data= torch.squeeze(data, 0)



        pred = data.numpy().transpose((1, 2, 0))

        # print('--------pred ka shape ----------',trimap.size)
        # print('--------pred ka shape ----------',trimap.shape)


        pred = cv2.resize(pred, (trimap.shape[1],trimap.shape[0]), cv2.INTER_LANCZOS4)[:, :, 0]
        # noinspection PyTypeChecker
        # Clean mask by removing all false predictions outside trimap and already known area

        trimap_arr = np.array(trimap.copy())
        pred[trimap_arr[:, :] == 0] = 0
        # pred[trimap_arr[:, :] == 255] = 1
        pred[pred < 0.3] = 0
        pred=pred*255
        # return Image.fromarray(pred * 255).convert("L")
        return pred

    # def __call__(
    #     self,
    #     images: List[Union[str, pathlib.Path, PIL.Image.Image]],
    #     trimaps: List[Union[str, pathlib.Path, PIL.Image.Image]],
    # ) -> List[PIL.Image.Image]:
    
    def __call__(self,image,trimap):
        
        """
        Passes input images though neural network and returns segmentation masks as PIL.Image.Image instances

        Args:
            images: input images
            trimaps: Maps with the areas we need to refine

        Returns:
            segmentation masks as for input images, as PIL.Image.Image instances

        """

                # if len(image) != len(trimap):
                #     raise ValueError(
                #         "Len of specified arrays of images and trimaps should be equal!"
                #     )

                # collect_masks = []
                # autocast, dtype = get_precision_autocast(device=self.device, fp16=self.fp16)
                # print("--------------------------------------")
                # print(autocast, dtype)
                # with autocast:
                # cast_network(self.model, dtype)
                # for idx_batch in batch_generator(range(len(image)), self.batch_size):
                # before1 = time.time()
                # before = time.time()
                # inpt_images = thread_pool_processing(
                #     lambda x: convert_imagee(load_image(image)), idx_batch
                # )

                # inpt_trimaps = thread_pool_processing(
                #     lambda x: convert_image(load_image(trimap), mode="L"), idx_batch
                # )
                # print("time for converting image: ", time.time() - before)
                # before = time.time()

        # print(self.data_preprocessing(image))

        inpt_img_transformed = (self.data_preprocessing(image))[1]
        # print(inpt_img_transformed.shape)
        inpt_trimaps_transformed = (self.data_preprocessing(trimap))[1]
        # print(inpt_img_transformed.shape)

        inpt_img = (self.data_preprocessing(image))[0]
        inpt_trimaps = (self.data_preprocessing(trimap))[0]


        print(type(inpt_trimaps))
        # print("time for preprocessing image: ", time.time() - before)
        # before = time.time()
        # inpt_img_batches_transformed = torch.vstack([i[1] for i in inpt_img_batches])
        # inpt_img_batches = torch.vstack([i[0] for i in inpt_img_batches])

        # inpt_trimaps_transformed = torch.vstack([i[1] for i in inpt_trimaps_batches])
        # inpt_trimaps_batches = torch.vstack([i[0] for i in inpt_trimaps_batches])
        # print("time for stacking image: ", time.time() - before)
        # print("Time for preprocessing imgs, trimaps in matting: ", time.time() - before1)
        # before = time.time()
        with torch.no_grad():
            inpt_img = inpt_img.to(self.device)
            inpt_trimaps= inpt_trimaps.to(self.device)
            inpt_img_transformed = inpt_img_transformed.to(self.device)
            inpt_trimaps_transformed = inpt_trimaps_transformed.to(self.device)

            # output = super(FBAMatting, self).__call__(
            #     inpt_img_batches,
            #     inpt_trimaps_batches,
            #     inpt_img_batches_transformed,
            #     inpt_trimaps_transformed,
            # )

            output =self.model(inpt_img,inpt_trimaps,inpt_img_transformed,inpt_trimaps_transformed)
            output_cpu = output.cpu()

            del (inpt_img,inpt_trimaps,output,inpt_img_transformed,inpt_trimaps_transformed)
        # print("Time for processing in matting: ", time.time() - before)
        # before = time.time()
        masks = self.data_postprocessing(output_cpu, trimap)
        # print("Time for postprocessing in matting: ", time.time() - before)
        # print("--------------------------------------")
        # collect_masks += masks
        return masks
