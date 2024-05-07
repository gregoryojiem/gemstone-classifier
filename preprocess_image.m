function gem_img = preprocess_image(input_img)
%PREPROCESS_IMAGE Takes in an image, returns a segmented gem image
% Given that gems are radically different in shape, color, size, 
% reflectivity, etc. this approach segments by performing edge detection,
% isolating gem edges, and taking the convex hull of those edges.

% Perform the basic preprocessing needed to normalize the image
gray_img = im2double(rgb2gray(input_img)); 
equalized_img = adapthisteq(gray_img);
blurred_img = imfilter(equalized_img, fspecial('average', 8));

% Make a sobel filter and use it to find the edge magnitues of img
sobel_vt_filter = [
    1, 2, 1;
    0, 0, 0;
    -1, -2, -1]/4;
sobel_hz_filter = sobel_vt_filter';

dldy = imfilter(blurred_img, sobel_vt_filter, 'same', 'repl' );
dldx = imfilter(blurred_img, sobel_hz_filter, 'same', 'repl' );

edge_mag = sqrt(dldx.^2 + dldy.^2);

% Binarize the edge magnitudes to ignore weaker edges
binarized_img = imbinarize(edge_mag, 0.033);

% Remove some of the remaining background noise with morphology
struct_el = strel('disk', 3);
cleaned_img = imopen(binarized_img, struct_el);

% Occasionally, a small amount of noise still remains. 
% We can fix this by removing all small pixel groups < 250 pixels

% Label all 8-conn objects and get their area in pixels
[labeled_img, obj_count] = bwlabel(cleaned_img);
props = regionprops(labeled_img, 'Area');

% Make an empty image and write the larger pixel groups to it
gem_edges_img = zeros(size(cleaned_img));
for i = 1:obj_count
    if props(i).Area > 250
        gem_edges_img = gem_edges_img + (labeled_img == i);
    end
end

% Crop out the wall edges
img_size = size(gem_edges_img);
crop_amount = 10;
x_min = crop_amount;
y_min = crop_amount;
width = img_size(2) - crop_amount * 2;
height = img_size(1) - crop_amount * 2;
wall_crop = [x_min, y_min, width, height];
walls_cropped_img = imcrop(gem_edges_img, wall_crop);

% Pad out the image so we can get a centered square crop of the gem
final_img_size = [672, 672];
padded_gem_mask = padarray(walls_cropped_img, final_img_size);

% Compute convex hull, at this point only gem edges should be remaining
% Get the centroid of filled in gem shape
gem_conv_hull = bwconvhull(padded_gem_mask);
centroid = regionprops(gem_conv_hull, 'Centroid').Centroid;

% Crop out a 672x672 region, with the gem being centered using centroid
x_min = round(centroid(1)) - final_img_size(1)/2;
y_min = round(centroid(2)) - final_img_size(2)/2;
width = final_img_size(1) - 1;
height = final_img_size(2) - 1;
normalize_crop = [x_min, y_min, width, height];
cropped_gem_mask = imcrop(gem_conv_hull, normalize_crop);

% Use the binary image as a mask to crop out the gem from original image
cropped_input_img = imcrop(input_img, wall_crop);
padded_input_img = padarray(cropped_input_img, final_img_size);
cropped_input_img = imcrop(padded_input_img, normalize_crop);

% Now we have a 672x672 RGB image with the gem centered
% The final step is to remove the background using the mask
segmented_gem = cropped_input_img .* uint8(cropped_gem_mask);

% Resize to 224x224 for our classifiers and return the image
gem_img = imresize(segmented_gem, 1/3, 'nearest');
end

