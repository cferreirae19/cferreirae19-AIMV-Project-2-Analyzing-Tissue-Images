import cv2
import numpy as np
import math
import os
import timeit


def segment_biopsy(image_file):
    # Read the image
    img = cv2.imread(image_file)

    b, g, r = cv2.split(img)

    # Calcular la varianza de cada canal de color
    variance_b = np.var(b)
    variance_g = np.var(g)
    variance_r = np.var(r)

    color_variance = (variance_b + variance_g + variance_r) / 3.0

    if (color_variance > 1500):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = os.path.basename(image_file)
    img_name = img_name.split(".")[0]
    img_area = img.shape[0] * img.shape[1]

    K = 2
    kernel_size = 3
    size_threshold = 1180 #1700  #0.000135

    # Reshape the image to a data matrix
    data = np.float32(img.reshape(-1, 3))

    # STEP 1: Kmeans with K=2 to separate tissue from not-tissue -------------------------------------------
    # Apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 3, flags)

    centers = np.uint8(centers)

    luminances = []

    for centroid in centers:
        if(color_variance > 1500):
            H, L, S = centroid
            luminances.append(L)
        else:
            R, G, B = centroid
            luminances.append(0.2126 * R + 0.7152 * G + 0.0722 * B)

    # Find the index of the centroid with the lowest and highest luminance
    min_luminance_index = np.argmin(luminances)
    max_luminance_index = np.argmax(luminances)

    # Create the mask
    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels == min_luminance_index] = 150  # Set pixels in the cluster with the lowest luminance to (0, 0, 150)
    mask[labels == max_luminance_index] = 75
    mask = mask.reshape(img.shape[0], img.shape[1])
    
    # -------------------------------------------------------------------------------------------------------
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Puedes ajustar el tamaño del kernel según tus necesidades
    # mask = cv2.dilate(mask, kernel, iterations=1)

    _, binary_mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))

    if contours:
        minarea = cv2.contourArea(contours[0])
        if minarea == 0:
            minarea = 0.5
    else:
        minarea = None
    # print(minarea)

    threshold_circularity = 0.5
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # Calcular la relación perímetro/área
        circularity = (4 * math.pi * area) / ((perimeter * perimeter) + 0.0000001)
        # print(circularity)

        if (circularity > threshold_circularity and area < (minarea)*(size_threshold)):
            # print(f'{area}/{img_area}')
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
            # cv2.drawContours(mask, [contour], 0, 0, 2)

    # ===== TESTING =====
    # Convert image to RGB
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract the blue channel
    blue_channel = rgb_image[:, :, 2]
    
    # Try different values
    blue_threshold_value = 75
    
    # Set pixels in the mask to 0 where there is a background
    mask[(blue_channel <= blue_threshold_value) & (mask!=255) & (mask!=150)] = 0
    # ===================

    output_directory = "SegImages"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cv2.imwrite(f'{output_directory}/{img_name}_masked.png', mask)

    # print('Image Size in pixels :  ', img.shape)
    # print('Image Size in mm: ', img.shape[0] / 2100.0, ' x ', img.shape[1] / 2100.0)

    count_stroma = np.sum(mask == 75)
    count_tissue = np.sum(mask == 150)
    count_nuclei = np.sum(mask == 255)
    count_total = count_stroma + count_tissue + count_nuclei

    # Imprimir los resultados
    print (f'\nImage: {img_name}')
    print(f'stroma: {count_stroma/ (2100.0 * 2100.0)} mm2 ({count_stroma*100/count_total}%)')
    print(f'tissue: {count_tissue/ (2100.0 * 2100.0)} mm2 ({count_tissue*100/count_total}%)')
    print(f'nuclei: {count_nuclei/ (2100.0 * 2100.0)} mm2 ({count_nuclei*100/count_total}%)')

    if(color_variance > 1500):
        img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    concatenation_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if img.shape[0] == concatenation_mask.shape[0]:
        # Concatenar las imágenes en horizontal
        img_comp = np.hstack((img, concatenation_mask))

    comp_directory = "CompImages"
    if not os.path.exists(comp_directory):
        os.makedirs(comp_directory)

    cv2.imwrite(f'{comp_directory}/{img_name}_comp.png', img_comp)

    return mask


def measure_execution_time():
    # run same code 5 times to get measurable data
    n = 5

    # Calcular el tiempo de ejecución promedio
    result = timeit.timeit(stmt='segment_biopsy(image_file)',
                           globals=globals(),
                           number=n)

    # Imprimir el tiempo de ejecución promedio
    print(f"Tiempo de ejecución promedio es {result / n} segundos")


def mask_all(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.tif')]

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        segment_biopsy(image_path)


if __name__ == "__main__":

    # README !!!
    # Comentar o descomentar las líneas según la prueba que se quiera realizar

    # 1) MASK ONE -------------------------------------------
    #image_file = 'CompImages/3919_1_comp.png'
    #mask = segment_biopsy(image_file)
    # -------------------------------------------------------

    # 2) MASK ALL -------------------------------------------
    directory_path = 'biopsia'
    mask_all(directory_path)
    # -------------------------------------------------------

    # 3) MEASURE TIME ---------------------------------------
    # measure_execution_time()
    # -------------------------------------------------------
