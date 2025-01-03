import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

class Auxiliar:
    @staticmethod
    def image_information(image_path, show_image=True, show_histogram=True, calculate_contrast=True, calculate_entropy=True, calculate_average_intensity=True):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        if show_image:
            plt.imshow(img, cmap='gray')
            plt.title('Imagem')
            plt.show()

        if show_histogram:
            plt.hist(img.ravel(), bins=256, range=[0, 256])
            plt.title('Histograma')
            plt.show()

        if calculate_contrast:
            contrast = img.max() - img.min()
            print(f"Contraste: {contrast}")

        if calculate_entropy:
            entropy = measure.shannon_entropy(img)
            print(f"Entropia: {entropy}")

        if calculate_average_intensity:
            average_intensity = np.mean(img)
            print(f"Intensidade média: {average_intensity}")

    @staticmethod
    def logical_operations(img1_path, img2_path, operation="AND"):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            print("Erro ao carregar as imagens!")
            return

        if operation == "AND":
            result = cv2.bitwise_and(img1, img2)
        elif operation == "OR":
            result = cv2.bitwise_or(img1, img2)
        elif operation == "XOR":
            result = cv2.bitwise_xor(img1, img2)
        else:
            print("Operação desconhecida!")
            return

        plt.imshow(result, cmap='gray')
        plt.title(f'Operação Lógica: {operation}')
        plt.show()

    @staticmethod
    def arithmetic_operations(img1_path, img2_path, operation="add"):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            print("Erro ao carregar as imagens!")
            return

        if operation == "add":
            result = cv2.add(img1, img2)
        elif operation == "subtract":
            result = cv2.subtract(img1, img2)
        elif operation == "multiply":
            result = cv2.multiply(img1, img2)
            result = np.clip(result, 0, 255)  # Limitando os valores para evitar overflow
        elif operation == "divide":
            img2 = np.where(img2 == 0, 1, img2)  # Evita divisão por zero substituindo zeros por uns
            result = cv2.divide(img1, img2)
            result = np.clip(result, 0, 255)  # Limitando os valores para evitar overflow
        else:
            print("Operação desconhecida!")
            return

        result = result.astype(np.uint8)  # Garantir que o tipo de dado seja uint8
        plt.imshow(result, cmap='gray')
        plt.title(f'Operação Aritmética: {operation}')
        plt.show()

    @staticmethod
    def binarize_image(image_path, method='binary', threshold=127):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        if method == 'binary':
            _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        elif method == 'adaptive':
            binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            print("Método de limiarização desconhecido!")
            return

        plt.imshow(binary_img, cmap='gray')
        plt.title('Imagem Binarizada')
        plt.show()

    @staticmethod
    def contrast_stretching(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        # Estiramento de contraste
        min_val = np.min(img)
        max_val = np.max(img)
        contrast_stretched_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Mostrar a imagem original e a imagem com estiramento de contraste
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        plt.imshow(contrast_stretched_img, cmap='gray')
        plt.title('Estiramento de Contraste')
        plt.show()

        # Comparar a entropia e o contraste
        original_entropy = measure.shannon_entropy(img)
        stretched_entropy = measure.shannon_entropy(contrast_stretched_img)
        original_contrast = img.max() - img.min()
        stretched_contrast = contrast_stretched_img.max() - contrast_stretched_img.min()

        print(f"Entropia Original: {original_entropy}, Entropia Estendida: {stretched_entropy}")
        print(f"Contraste Original: {original_contrast}, Contraste Estendido: {stretched_contrast}")

    @staticmethod
    def histogram_equalization(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        equalized_img = cv2.equalizeHist(img)

        # Mostrar a imagem original e a imagem equalizada
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        plt.imshow(equalized_img, cmap='gray')
        plt.title('Equalização de Histograma')
        plt.show()

    @staticmethod
    def histogram_specification(image_path, reference_image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or ref_img is None:
            print("Erro ao carregar as imagens!")
            return

        # Especificação de histograma
        spec_img = cv2.calcHist([img], [0], None, [256], [0, 256])
        ref_hist = cv2.calcHist([ref_img], [0], None, [256], [0, 256])

        # Cumulative distribution function
        cdf_img = spec_img.cumsum()
        cdf_ref = ref_hist.cumsum()

        # Normalização
        cdf_img = (255.0 / cdf_img[-1]) * cdf_img
        cdf_ref = (255.0 / cdf_ref[-1]) * cdf_ref

        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            closest_val = np.argmin(np.abs(cdf_ref - cdf_img[i]))
            lookup_table[i] = closest_val

        specified_img = cv2.LUT(img, lookup_table)

        # Mostrar a imagem original, a imagem referência e a imagem especificada
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagem Original')

        plt.subplot(1, 3, 2)
        plt.imshow(ref_img, cmap='gray')
        plt.title('Imagem Referência')

        plt.subplot(1, 3, 3)
        plt.imshow(specified_img, cmap='gray')
        plt.title('Especificação de Histograma')
        plt.show()

    @staticmethod
    def sharpening(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        # Kernel para afilamento
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])

        sharpened_img = cv2.filter2D(img, -1, kernel)

        # Mostrar a imagem original e a imagem afilada
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        plt.imshow(sharpened_img, cmap='gray')
        plt.title('Afilamento')
        plt.show()

    @staticmethod
    def binarize_sharpened_image(image_path, threshold=127):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        # Kernel para afilamento
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
        [0, -1, 0]])

        sharpened_img = cv2.filter2D(img, -1, kernel)

        # Mostrar a imagem original e a imagem afilada
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        plt.imshow(sharpened_img, cmap='gray')
        plt.title('Afilamento')
        plt.show()

        return sharpened_img

    @staticmethod
    def binarize_sharpened_image(image_path, threshold=127):
        sharpened_img = Auxiliar.sharpening(image_path)
        if sharpened_img is None:
            return

        _, binary_img = cv2.threshold(sharpened_img, threshold, 255, cv2.THRESH_BINARY)

        # Mostrar a imagem afilada e a imagem binarizada
        plt.subplot(1, 2, 1)
        plt.imshow(sharpened_img, cmap='gray')
        plt.title('Imagem Afilada')

        plt.subplot(1, 2, 2)
        plt.imshow(binary_img, cmap='gray')
        plt.title('Imagem Binarizada')
        plt.show()

    @staticmethod
    def apply_pseudo_color(image_path, colormap=cv2.COLORMAP_JET):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        pseudo_color_img = cv2.applyColorMap(img, colormap)

        # Mostrar a imagem original e a imagem com pseudo-cor
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        plt.imshow(pseudo_color_img)
        plt.title('Pseudo-Cor')
        plt.show()

    @staticmethod
    def analyze_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        # Mostrar a imagem e seu histograma
        plt.subplot(1, 2, 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        if len(img.shape) == 2:
            plt.hist(img.ravel(), bins=256, range=[0, 256])
        else:
            for i, color in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
        plt.title('Histograma')
        plt.show()

    @staticmethod
    def restore_image(image_path, method='median', kernel_size=3):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        if method == 'median':
            restored_img = cv2.medianBlur(img, kernel_size)
        elif method == 'gaussian':
            restored_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif method == 'bilateral':
            restored_img = cv2.bilateralFilter(img, kernel_size, 75, 75)
        else:
            print("Método desconhecido!")
            return

        # Mostrar a imagem original e a imagem restaurada
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        plt.imshow(restored_img, cmap='gray')
        plt.title('Imagem Restaurada')
        plt.show()

        return img, restored_img

    @staticmethod
    def calculate_metrics(original_img, restored_img):
        MSE = np.mean((original_img - restored_img) ** 2)
        MAE = np.mean(np.abs(original_img - restored_img))

        print(f"MSE: {MSE}")
        print(f"MAE: {MAE}")

    @staticmethod
    def minutiae_detection(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro ao carregar a imagem!")
            return

        # Binarização da imagem
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Extrair bordas usando Canny
        edges = cv2.Canny(binary_img, 100, 200)

        # Mostrar a imagem binarizada e as bordas
        plt.subplot(1, 2, 1)
        plt.imshow(binary_img, cmap='gray')
        plt.title('Imagem Binarizada')

        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Bordas')
        plt.show()

        return edges

# Exemplo de uso
# Inicializando a classe Auxiliar
auxiliar = Auxiliar()

# Caminho para a imagem
image_path = 'imgs/lena.jpeg'

# Exemplo de uso para obter informações da imagem
auxiliar.image_information(image_path, show_image=True, show_histogram=True, calculate_contrast=True, calculate_entropy=True, calculate_average_intensity=True)

# Exemplo de uso para operações lógicas
auxiliar.logical_operations(image_path, image_path, operation="AND")

# Exemplo de uso para operações aritméticas
auxiliar.arithmetic_operations(image_path, image_path, operation="add")

# Exemplo de uso para binarização de imagem
auxiliar.binarize_image(image_path, method='otsu')

# Exemplo de uso para estiramento de contraste
auxiliar.contrast_stretching(image_path)

# Exemplo de uso para equalização de histograma
auxiliar.histogram_equalization(image_path)

# Exemplo de uso para especificação de histograma
auxiliar.histogram_specification(image_path, 'imgs/lena.jpeg')

# Exemplo de uso para afilamento de imagem
auxiliar.sharpening(image_path)

# Exemplo de uso para binarização de imagem afilada
auxiliar.binarize_sharpened_image(image_path)

# Exemplo de uso para aplicar pseudo-cor
auxiliar.apply_pseudo_color(image_path, colormap=cv2.COLORMAP_JET)

# Exemplo de uso para análise de imagem e histograma
auxiliar.analyze_image(image_path)

# Exemplo de uso para restauração de imagem e cálculo de métricas
original_img, restored_img = auxiliar.restore_image(image_path, method='median')
auxiliar.calculate_metrics(original_img, restored_img)

# Exemplo de uso para detecção de minúcias
auxiliar.minutiae_detection(image_path)
