# SegMat
Using Image Matting to improve the Segmentation results, which avoids the confusion between foreground and background in boundary pixels of each contour.

Sometimes we can not generate accurate results through our segmentation model because the probabbility of edge pixels is very low to classify under the object, This can be due to many reasons like Less number of data for model to learn well, inaccurate annotations by the user ,Unsharp images,Lack of contranst btween fg and bg etc.., So we should include one extra model which is specialized in performing this task for us which is Image Matting.     

This Method proves to be very useful when talking abouth the consideration of edge pixels and can be used in variety of task where we need a clear distinction in between the fg and bg . Some of the AI in built cameras use this methodology to sapture sharp images.   

The Working of this process is shown below in the image---

![Screenshot from 2023-06-10 18-43-29](https://github.com/Rishabh20539011/SegMat/assets/101064926/d31a6f85-d123-4cec-9f25-da00281e029f)

In this repo we are using architecture of [Unet++]{https://smp.readthedocs.io/en/latest/models.html#segmentation_models_pytorch.UnetPlusPlus} for Segmentation and [FBA-MATT]{https://github.com/MarcoForte/FBA_Matting} for matting which gives us Real time performance.

1. Current SOTA for Segmentation through CNNs--- https://github.com/qubvel/segmentation_models.pytorch
2. Current SOTA for Image Matting --- https://paperswithcode.com/sota/image-matting-on-composition-1k-1
