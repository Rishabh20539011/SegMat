from PIL import Image
from matting.cv_gen import CV2TrimapGenerator
from matting.add_ops import prob_filter, prob_as_unknown_area, post_erosion
import cv2

class TrimapGenerator(CV2TrimapGenerator):
    def __init__(
        self, prob_threshold: int =231, kernel_size: int = 5, erosion_iters: int =2
    ):
        """
        Initialize a TrimapGenerator instance

        Args:
            prob_threshold: Probability threshold at which the
            prob_filter and prob_as_unknown_area operations will be applied
            kernel_size: The size of the offset from the object mask
            in pixels when an unknown area is detected in the trimap
            erosion_iters: The number of iterations of erosion that
            the object's mask will be subjected to before forming an unknown area
        """
        super().__init__(kernel_size, erosion_iters=0)
        self.prob_threshold = prob_threshold
        self.__erosion_iters = erosion_iters

    def __call__(self,mask):
        """
        Generates trimap based on predicted object mask to refine object mask borders.
        Based on cv2 erosion algorithm and additional prob. filters.
        Args:
            original_image: Original image
            mask: Predicted object mask

        Returns:
            Generated trimap for image.
        """
        filter_mask = prob_filter(mask, prob_threshold=self.prob_threshold)
        trimap = super(TrimapGenerator, self).__call__(filter_mask)
        new_trimap = prob_as_unknown_area(trimap,mask, prob_threshold=self.prob_threshold)
        new_trimap = post_erosion(new_trimap, self.__erosion_iters)

        # cv2.imwrite('new_trimap.bmp',new_trimap)
        
        return new_trimap
