import os
import cv2
import numpy as np
from os import listdir
import skimage
from tabulate import tabulate
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms 
from Operations import *
import fingerprint_feature_extractor as fpe
from fingerprint_enhancer.fingerprint_image_enhancer import FingerprintImageEnhancer


class ImageUtils: 
    
    @staticmethod
    def image_information(image, show_cdf=False):
        """
        This function shows the image and its histogram, computes the image contrast, entropy and the average intensity.
        """

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        
        contrast = ImageUtils.image_contrast(image)
        entropy = ImageUtils.image_entropy(image)
        average_intensity = ImageUtils.image_average_intensity(image)
        
        print(f'Contrast: {contrast:.2f}')
        print(f'Entropy: {entropy:.2f}')
        print(f'Average Intensity: {average_intensity:.2f}')

        plt.subplots(1, 2, figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(hist)
        if show_cdf: plt.plot(cdf_normalized, color='r', alpha=0.5)
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        plt.title('Histogram')
        plt.tight_layout()
        plt.show()

        return contrast, entropy, average_intensity


    @staticmethod
    def images_information(images, titles, show_cdf=False):
        """
        This function shows the image and its histogram, computes the image contrast, entropy and the average intensity.
        """
        size = len(images)
        fig, axes = plt.subplots(size, 2, figsize=(10, 4*size))
        table_header = ['Image','Contrast', 'Entropy', 'Average Intensity']
        table_data = []
        for i in range(size):

            hist = cv2.calcHist([images[i]], [0], None, [256], [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
        

            axes[i, 0].set_title(titles[i])
            axes[i, 0].axis('off')
            axes[i, 0].imshow(images[i], cmap='gray')
            axes[i, 1].grid(color = 'black', linestyle = '--', linewidth = 0.5)
            axes[i, 1].set_title('Histogram')
            axes[i, 1].set_xlabel('Intensity level')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].plot(hist)
            if show_cdf: axes[i, 1].plot(cdf_normalized, color='r', alpha=0.5)


            table_data.append([
                titles[i], 
                f'{ImageUtils.image_contrast(images[i]):.2f}', 
                f'{ImageUtils.image_entropy(images[i]):.2f}', 
                f'{ImageUtils.image_average_intensity(images[i]):.2f}'
            ])

            
        table_string = tabulate(table_data, headers=table_header, tablefmt="grid")
        print(table_string)
        
        plt.tight_layout()
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
    def image_histogram(image):
        """
        This function computes the histogram of an image.
        """
        hist = image.ravel()
        hist, _ = np.histogram(image.ravel(), bins=256, range=[0, 256])
        hist = hist / hist.sum()

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
    def apply_morphological_operation(image, operation=None, kernel=(3, 3), morphElement=cv2.MORPH_RECT):
        """
        This function performs morphological operations on an image.
        """
        if operation == MorphologicalOperations.EROSION:
            structElem = cv2.getStructuringElement(morphElement, kernel)
            result = cv2.erode(image, structElem)
        elif operation == MorphologicalOperations.DILATION:
            structElem = cv2.getStructuringElement(morphElement, kernel)
            result = cv2.dilate(image, structElem)
        elif operation == MorphologicalOperations.OPENING:
            structElem = cv2.getStructuringElement(morphElement, kernel)
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, structElem)
        elif operation == MorphologicalOperations.CLOSING:
            structElem = cv2.getStructuringElement(morphElement, kernel)
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, structElem)
        else:
            raise ValueError('Invalid operation. Choose a valid Morphological Operation.')

        return result

    @staticmethod
    def thresholding(image, threshold=127, method=ThresholdingMethods.BINARY_OTSU):
        """
        This function performs thesholding on an image.
        """
        result = None

        if method is ThresholdingMethods.OTSU:
            threshold, result = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        elif method is ThresholdingMethods.BINARY_OTSU:
            threshold, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method is ThresholdingMethods.BINARY:
            if isinstance(threshold, int):
                _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        else:
            print('Invalid method. Choose a valid thresholding method.')

        return result
        
    @staticmethod
    def contrast_stretching(image, min=None, max=None, lookUpTable=None):
        """
        This function performs contrast stretching on an image.
        """
        if lookUpTable is None:
            if min is None: min = np.min(image)
            if max is None: max = np.max(image)
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip((i - min) * 255 / (max - min), 0, 255)

        
        return cv2.LUT(image, lookUpTable), lookUpTable
    

    @staticmethod
    def histogram_equalization(image, method=EquializationMethods.CLASSIC, clipLimit=2.0, tileGridSize=(8, 8)):
        if method == EquializationMethods.CLASSIC:
            return cv2.equalizeHist(image)
        elif method == EquializationMethods.CLAHE:
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            return clahe.apply(image)
    
    @staticmethod
    def histogram_specification(image, image_ref):
        return match_histograms(image, image_ref).astype(np.uint8)
    

    @staticmethod
    def pseudo_coloring(image, colorMap='jet'):
        if colorMap == 'jet':
            return cv2.applyColorMap(image, cv2.COLORMAP_JET)
        elif colorMap == 'hot':
            return cv2.applyColorMap(image, cv2.COLORMAP_HOT)
        elif colorMap == 'bone':
            return cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        elif colorMap == 'cool':
            return cv2.applyColorMap(image, cv2.COLORMAP_COOL)
        elif colorMap == 'copper':
            return cv2.applyColorMap(image, cv2.COLORMAP_COPPER)
        elif colorMap == 'rainbow':
            return cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)
        elif colorMap == 'turbo':
            return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
        elif colorMap == 'viridis':
            return cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
        elif colorMap == 'plasma':
            return cv2.applyColorMap(image, cv2.COLORMAP_PLASMA)
        elif colorMap == 'ocean':
            return cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)
        elif colorMap == 'magma':
            return cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
        elif colorMap == 'spring':
            return cv2.applyColorMap(image, cv2.COLORMAP_SPRING)
        else:
            raise ValueError('Invalid method. Choose a valid pseudo coloring method.')
        

    @staticmethod
    def smoth_image(image, kernel=3, sigma=1.0):
        return cv2.GaussianBlur(image, (kernel, kernel), sigma)
    
    @staticmethod
    def compute_mask(original, smothed):
        mask = np.zeros_like(original)
        mask[original > smothed] = 1
        return mask
    
    @staticmethod
    def enhance_image(image, kernel=3, k=1.0):
        smothed = ImageUtils.smoth_image(image, kernel=kernel, sigma=k)
        mask = ImageUtils.compute_mask(image, smothed)

        # k > 1: Maior peso da máscara na imagem
        # k < 1: Menor menor peso da máscara na imagem
        # k = 1: Máscara com mesmo peso da imagem
        return (image + k * mask).astype(np.uint8), mask.astype(np.uint8)


    @staticmethod
    def remove_salt_and_pepper_noise(image, kern=3):
        return cv2.medianBlur(image, kern)
        # return skimage.filters.rank.median(image)

    @staticmethod
    def remove_gaussian_noise(image, kernel=3, sigma=1.0):
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), sigma)


    @staticmethod
    def print_standard_metrics(original, noisy):
        mse = ImageUtils.compute_MSE(original, noisy)
        mae = ImageUtils.compute_MAE(original, noisy)

        table_header = ['Mean Square Error (MSE)', 'Mean Absolute Error (MAE)']
        table_data = [[f'{mse:.2f}', f'{mae:.2f}']]
            
        table_string = tabulate(table_data, headers=table_header, tablefmt="grid")
        print(table_string)


    @staticmethod
    def compute_MSE(original, noisy):
        return np.mean((original - noisy) ** 2)
    
    @staticmethod
    def compute_PSNR(original, noisy):
        mse = ImageUtils.compute_MSE(original, noisy)
        return 10 * np.log10(np.max(original) ** 2 / mse)
    
    @staticmethod
    def compute_MAE(original, noisy):
        return np.mean(np.abs(original - noisy))
    

    @staticmethod
    def thinning(image):
        return cv2.ximgproc.thinning(image)
    

    def enhance_fingerprint_image(): 
        # Smoothing


        # Thresholding

        # Edge detection

        # Ridge detection

        # Minutiae detection (ridge terminations and bifurcations)
        pass



    @staticmethod
    def skeletonize_image(binary_image):
        """Aplica o filtro de esqueleto à imagem binária."""
        skeleton = skimage.morphology.skeletonize(binary_image)
        return skeleton.astype(np.uint8)
    

    @staticmethod
    def enhance(image, saveResult=False, folder=None, image_name=None):

        image_enhancer = FingerprintImageEnhancer()

        image_enhancer.enhance(image, invert_output=True)

        image_enhancer.save_enhanced_image("enhanced/" + image_name+'.png')

        #move result to folder 
        if saveResult and folder is not None and image_name is not None:
            if not os.path.exists(f"{folder}enhanced"):
                os.makedirs(f"{folder}enhanced")

            os.replace("enhanced/" + image_name+".png", f'{folder}enhanced/{image_name}.png')



    @staticmethod
    def extract_minutiae(image, showResult=False, saveResult=False, folder=None, image_name=None):
        FeaturesTerminations, FeaturesBifurcations = fpe.extract_minutiae_features(image, spuriousMinutiaeThresh=3, invertImage=False, showResult=showResult, saveResult=saveResult)

        #move result to folder 
        if saveResult and folder is not None and image_name is not None:

            if not os.path.exists(f"{folder}"):
                os.makedirs(f"{folder}")

            os.replace('result.png', f'{folder}{image_name}.bmp')
            
        return FeaturesTerminations, FeaturesBifurcations


    
    @staticmethod
    def edge_detection(image, method=SharpeningOperations.ROBERTS, kernel=None):
        if method == SharpeningOperations.CANNY:
            result =  cv2.Canny(image, 75, 150)
        elif method == SharpeningOperations.CUSTOM:
            result =  cv2.filter2D(image, -1, kernel)
        elif method == SharpeningOperations.SOBEL:
            result =  skimage.filters.sobel(image)
        elif method == SharpeningOperations.ROBERTS:
            result =  skimage.filters.roberts(image)
        elif method == SharpeningOperations.PREWITT:
            result =  skimage.filters.prewitt(image)
        elif method == SharpeningOperations.LOG:
            result = cv2.GaussianBlur(image, (3, 3), 0)
            result =  cv2.Laplacian(result, cv2.CV_16S, ksize=3)
            result = cv2.convertScaleAbs(result)
        
        if result.dtype != np.uint8:
            result = (result - result.min()) / (result.max() - result.min()) * 255
            result = result.astype(np.uint8)
        
        return result
    

    @staticmethod
    def get_images_from_folder(source_directory, n_images=None, imgs_format = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'gif', 'giff'], contains=None):
        images = []
        for file in os.listdir(source_directory):
            if file.split('.')[-1] in imgs_format:
                if contains is not None:
                    if contains in file:
                        images.append(os.path.join(source_directory, file))
                else:
                    images.append(os.path.join(source_directory, file))

        if n_images is not None:
            return images[:n_images]
        else:
            return images
        

    @staticmethod
    def select_image_from_folder(source_directory, image_name=None):
        images = ImageUtils.get_images_from_folder(source_directory)
        if image_name is not None:
            for image in images:
                if image_name in image:
                    return image
        else:
            print(f"Folder {source_directory} does not contain image {image_name}")
            return None

    
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


    

    

    