import matplotlib.pyplot as plt
import numpy as np
import cv2
from Operations import LogicalOperations, ArtihmeticOperations, SharpeningOperations
from tabulate import tabulate
from skimage.exposure import match_histograms 
from skimage import filters
from os import listdir


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
    def images_information(images, titles):
        """
        This function shows the image and its histogram, computes the image contrast, entropy and the average intensity.
        """
        size = len(images)
        fig, axes = plt.subplots(size, 2, figsize=(10, 4*size))
        table_header = ['Image','Contrast', 'Entropy', 'Average Intensity']
        table_data = []
        for i in range(size):
            axes[i, 0].set_title(titles[i])
            axes[i, 0].axis('off')
            axes[i, 0].imshow(images[i], cmap='gray')

            hist = images[i].ravel()
            #axes[i, 1].set_title('Histogram')
            axes[i, 1].set_xlabel('Intensity level')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].hist(hist, bins=256, range=(0, 256))

            table_data.append([
                titles[i], 
                f'{ImageUtils.image_contrast(images[i]):.2f}', 
                f'{ImageUtils.image_entropy(images[i]):.2f}', 
                f'{ImageUtils.image_average_intensity(images[i]):.2f}'
            ])
            
        table_string = tabulate(table_data, headers=table_header, tablefmt="grid")
        print(table_string)
        plt.show()

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
        hist = image.ravel()
        #hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

        if show:
            plt.figure(figsize=(5, 5))
            plt.hist(hist, bins=256, range=(0, 256))
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
        #print(f'Contrast: {contrast}')
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
    
    @staticmethod
    def contrast_stretching(image, min=None, max=None):
        """
        This function performs contrast stretching on an image.
        """
        if min is None:
            min = np.min(image)
        if max is None:
            max = np.max(image)
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip((i - min) * 255 / (max - min), 0, 255)
        return cv2.LUT(image, lookUpTable), lookUpTable
    

    @staticmethod
    def histogram_equalization(image):
        return cv2.equalizeHist(image)
    
    @staticmethod
    def histogram_specification(image, image_ref):
        return match_histograms(image, image_ref)
    
    @staticmethod
    def edge_detection(image, operation, kernel=None):
        if operation == SharpeningOperations.CANNY:
            result =  cv2.Canny(image, 75, 150)
        elif operation == SharpeningOperations.CUSTOM:
            result =  cv2.filter2D(image, -1, kernel)
        elif operation == SharpeningOperations.SOBEL:
            result =  filters.sobel(image)
        elif operation == SharpeningOperations.ROBERTS:
            result =  filters.roberts(image)
        elif operation == SharpeningOperations.PREWITT:
            result =  filters.prewitt(image)
        elif operation == SharpeningOperations.LOG:
            result = cv2.GaussianBlur(image, (3, 3), 0)
            result =  cv2.Laplacian(result, cv2.CV_16S, ksize=3)
            result = cv2.convertScaleAbs(result)
        
        if result.dtype != np.uint8:
            result = (result - result.min()) / (result.max() - result.min()) * 255
            result = result.astype(np.uint8)
        
        return result
    
    @staticmethod
    def load_data(path):
        persons = listdir(path)
        labels = []
        data = []
        for person in persons:
            record_path = '{}/{}'.format(path, person)
            records = listdir(record_path)
            data.extend([
                cv2.imread('{}/{}'.format(record_path, record), cv2.IMREAD_GRAYSCALE) 
                for record in records
            ])
            labels.extend([person] * len(records))
        return np.array(data), np.array(labels)


    

    

    