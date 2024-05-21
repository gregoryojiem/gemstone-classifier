import cv2
import numpy as np
from skimage import exposure, measure, morphology

debug = True


def preprocess_image(input_img):
    """
    This function uses edge-based object segmentation techniques to isolate a gem from its background. It also resizes
    the image to 224x224 for use in training a classifier.

    :param input_img: The input gem image. Must be 1920x1080 and ideally the gem and background are distinct
    :return: 224x224 RGB image of the input gem with background pixels set to [0, 0, 0].
    """
    # Convert image to grayscale, and normalize the lighting using histogram equalization
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    equalized_img = exposure.equalize_adapthist(gray_img)
    blurred_img = cv2.blur(equalized_img, (8, 8))

    # Edge detection w/ sobel filter, and then binarization
    sobel_vt_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64) / 4.0
    sobel_hz_filter = sobel_vt_filter.T
    dldy = cv2.filter2D(blurred_img, -1, sobel_vt_filter)
    dldx = cv2.filter2D(blurred_img, -1, sobel_hz_filter)

    edge_mag = np.sqrt(dldx ** 2 + dldy ** 2)
    _, binarized_img = cv2.threshold(edge_mag, 0.033, 1, cv2.THRESH_BINARY)

    # Remove background noise using morphology and by ignoring small pixel groups
    struct_el = morphology.disk(3)
    cleaned_img = morphology.opening(binarized_img, struct_el)

    labeled_img, obj_count = measure.label(cleaned_img, return_num=True, connectivity=2)
    props = measure.regionprops(labeled_img)

    gem_edges_img = np.zeros(cleaned_img.shape, dtype=np.uint8)
    for prop in props:
        if prop.area > 250:
            gem_edges_img[labeled_img == prop.label] = 1

    # Crop out the wall edges, and pad image for a centered square crop
    crop_amount = 10
    walls_cropped_img = gem_edges_img[crop_amount:-crop_amount, crop_amount:-crop_amount]
    final_img_size = (672, 672)
    padded_gem_mask = np.pad(walls_cropped_img, ((0, 672), (0, 672)), mode='constant')

    # Compute convex hull and centroid
    gem_conv_hull = morphology.convex_hull_image(padded_gem_mask)
    centroid = measure.regionprops(gem_conv_hull.astype(np.uint8))[0].centroid

    # Centered crop
    x_min = int(round(centroid[1]) - final_img_size[1] / 2)
    y_min = int(round(centroid[0]) - final_img_size[0] / 2)
    normalize_crop = (slice(y_min, y_min + final_img_size[0]), slice(x_min, x_min + final_img_size[1]))
    cropped_gem_mask = gem_conv_hull[normalize_crop]

    # Use mask to crop gem from original image, and resize to 224x224
    cropped_input_img = input_img[crop_amount:-crop_amount, crop_amount:-crop_amount]
    padded_input_img = np.pad(cropped_input_img, ((0, 672), (0, 672), (0, 0)), mode='constant')
    cropped_input_img = padded_input_img[normalize_crop]
    segmented_gem = cropped_input_img * np.expand_dims(cropped_gem_mask, axis=-1).astype(np.uint8)
    gem_img = cv2.resize(segmented_gem, (224, 224), interpolation=cv2.INTER_NEAREST)

    # Display the original and final images
    if debug:
        cv2.imshow("Original", cv2.resize(input_img, (384, 216))) # Resize original for a better comparison
        cv2.imshow("Final Image", gem_img)
        cv2.waitKey(0)

    return gem_img

