import matplotlib.pyplot as plt
import numpy as np
import cv2
from Operations import LogicalOperations, ArtihmeticOperations


class ImageUtils: 
    
    @staticmethod
    def image_information(image):
        """
        This function shows the image and its histogram, computes the image contrast, entropy and the average intensity.
        """
        ImageUtils.image_show(image, title='Original Image')

        ImageUtils.image_histogram(image, show=True)

        contrast = ImageUtils.image_contrast(image)
        entropy = ImageUtils.image_entropy(image)
        average_intensity = ImageUtils.image_average_intensity(image)
        
        print(f'Contrast: {contrast:.2f}')
        print(f'Entropy: {entropy:.2f}')
        print(f'Average Intensity: {average_intensity:.2f}')

    @staticmethod
    def image_show(image, title=None):
        """
        This function shows an image.
        """
        plt.figure(figsize=(5, 5))
        if title: plt.title(str(title))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    @staticmethod
    def image_histogram(image, show=True):
        """
        This function computes the histogram of an image.
        """
        hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

        if show:
            plt.figure(figsize=(5, 5))
            plt.hist(hist, bins=bins)
            plt.title('Histogram')
            plt.xlabel('Intensity level')
            plt.ylabel('Frequency')
            plt.show()

        return hist
    
    @staticmethod
    def image_contrast(image):
        """
        This function computes the contrast of an image.
        """
        contrast = np.max(image) - np.min(image)
        print(f'Contrast: {contrast}')
        return contrast
    
    @staticmethod
    def image_entropy(image):
        """
        This function computes the entropy of an image.
        """
        hist, _ = np.histogram(image.ravel(), bins=256, range=[0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + (hist == 0)))
        return entropy
    
    @staticmethod
    def image_average_intensity(image):
        """
        This function computes the average intensity of an image.
        """
        average_intensity = np.mean(image)
        return average_intensity
    

    @staticmethod
    def apply_logical_operation(image1, image2, operation=None):
        """
        This function performs logical operations between two images.
        """
        if operation == LogicalOperations.AND:
            result = cv2.bitwise_and(image1, image2)
        elif operation == LogicalOperations.OR:
            result = cv2.bitwise_or(image1, image2)
        elif operation == LogicalOperations.XOR:
            result = cv2.bitwise_xor(image1, image2)
        elif operation == LogicalOperations.NOT:
            result = cv2.bitwise_not(image1)
        else:
            raise ValueError('Invalid operation. Choose a valid Logical Operation.')

        return result
    
    @staticmethod
    def apply_arithmetic_operation(image1, image2, operation=None, alpha=None):
        """
        This function performs arithmetic operations between two images.
        """
        if operation == ArtihmeticOperations.ADD and alpha is not None:
            result = cv2.addWeighted(src1=image1, alpha=1 - alpha, src2=image2, beta=alpha, gamma=0)
        elif operation == ArtihmeticOperations.SUBTRACT:
            result = cv2.subtract(image1, image2)
        elif operation == ArtihmeticOperations.MULTIPLY:
            result = cv2.multiply(image1, image2)
        elif operation == ArtihmeticOperations.DIVIDE:
            result = cv2.divide(image1, image2)
        else:
            raise ValueError('Invalid operation. Choose a valid Artihmetic Operation.')

        return result

    @staticmethod
    def thresholding(image, threshold=None):
        """
        This function performs thesholding on an image.
        """
        if threshold is None:
            threshold, result = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        else:
            _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        return result
    

    

    