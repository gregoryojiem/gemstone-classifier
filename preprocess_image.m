function [gem_img, gem_mask] = preprocess_image(input_img)
%PREPROCESS_IMAGE This function takes an input image containing a gem and
% segements it out from the bg. Given that gems are radically different in
% shape, color, size, reflectivity, etc. this approach segments by using 
% edge detection and then the convex hull of the isolated edges.
gray_img = im2double(rgb2gray(input_img)); 
equalized_img = adapthisteq(gray_img);
blurred_img = imfilter(equalized_img, fspecial('average', 8));

sobel_vt_filter = [ 1, 2, 1 ;
0, 0, 0 ;
-1, -2, -1 ]/4;

sobel_hz_filter = sobel_vt_filter';

dldy = imfilter(blurred_img, sobel_vt_filter, 'same', 'repl' );
dldx = imfilter(blurred_img, sobel_hz_filter, 'same', 'repl' );
edge_mag = sqrt(dldx.^2 + dldy.^2);
binarized_img = imbinarize(edge_mag, 0.033);

struct_el = strel('disk', 3);
cleaned_img = imopen(binarized_img, struct_el);

[labeled_img, obj_count] = bwlabel(cleaned_img);
props = regionprops(labeled_img, 'Area');
    
gem_edges_img = zeros(size(cleaned_img));

for i = 1:obj_count
    if props(i).Area > 250
        gem_edges_img = gem_edges_img + (labeled_img == i);
    end
end

img_size = size(gem_edges_img);
crop_amount = 10;
x_min = crop_amount;
y_min = crop_amount;
width = img_size(2) - crop_amount * 2;
height = img_size(1) - crop_amount * 2;
wall_crop = [x_min, y_min, width, height];
walls_cropped_img = imcrop(gem_edges_img, wall_crop);
final_img_size = [672, 672];
padded_gem_mask = padarray(walls_cropped_img, final_img_size);

gem_conv_hull = bwconvhull(padded_gem_mask);
centroid = regionprops(gem_conv_hull, 'Centroid').Centroid;
x_min = round(centroid(1)) - final_img_size(1)/2;
y_min = round(centroid(2)) - final_img_size(2)/2;
width = final_img_size(1) - 1;
height = final_img_size(2) - 1;
normalize_crop = [x_min, y_min, width, height];
cropped_gem_mask = imcrop(gem_conv_hull, normalize_crop);

cropped_input_img = imcrop(input_img, wall_crop);
padded_input_img = padarray(cropped_input_img, final_img_size);
cropped_input_img = imcrop(padded_input_img, normalize_crop);

segmented_gem = cropped_input_img .* uint8(cropped_gem_mask);

gem_img = imresize(segmented_gem, 1/3, 'nearest');
gem_mask = imresize(cropped_gem_mask, 1/3, 'nearest');
end

