# SegMat
Utilizing Image Matting for enhancing Segmentation results is a technique that addresses the confusion between foreground and background in boundary pixels of each contour.

In certain cases, our segmentation model may not produce accurate outcomes due to the low probability of classifying edge pixels as part of the object. This can be attributed to various factors such as insufficient training data, imprecise user annotations, unsharp images, and insufficient contrast between foreground and background. To overcome this, incorporating an additional specialized model like Image Matting becomes essential.

Image Matting proves to be highly advantageous when dealing with edge pixels, and it can be employed in various tasks that require a clear differentiation between foreground and background. Some AI-integrated cameras employ this methodology to capture sharp images.

The Working of this process is shown below in the image---

![Screenshot from 2023-06-10 18-43-29](https://github.com/Rishabh20539011/SegMat/assets/101064926/d31a6f85-d123-4cec-9f25-da00281e029f)

In this repo we are used the following  architecture for real time performance---
1. [Unet++](https://smp.readthedocs.io/en/latest/models.html#segmentation_models_pytorch.UnetPlusPlus) for Segmentation
2. [FBA-MATT](https://github.com/MarcoForte/FBA_Matting) for matting  

 Useful Resources:
1. Segmentation--- https://github.com/qubvel/segmentation_models.pytorch
2. Image Matting --- https://paperswithcode.com/sota/image-matting-on-composition-1k-1
