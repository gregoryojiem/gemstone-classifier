function feature_vector = generate_features(gem_img, gem_mask)
%GENERATE_FEATURES Generates features to help classify an image
% Uses the gem image and the mask image to calculate various features for 
% training a decision tree, features like average color, the size of the 
% gem compared to the background, eccentricity of the gem, etc. 

% Get the average lightness, a*, and b* values of the image
[avg_l_val, avg_a_val, avg_b_val] = get_avg_lab_values(gem_img, gem_mask);

% Calculate the ratio of foreground (gem pixels) to background pixels
num_of_pixels = numel(gem_mask)/3;
num_of_gem_pixels = sum(gem_mask(:));
size_ratio = numel(num_of_gem_pixels)/num_of_pixels;

% Calculate eccentricity
eccentricity = regionprops(gem_mask, "Eccentricity").Eccentricity;

% Store the gem pixels in a Nx3 form for kmeans
[l_vals, a_vals, b_vals] = imsplit(rgb2lab(gem_img)); 
img_data = [l_vals(gem_mask), a_vals(gem_mask), b_vals(gem_mask)];

% Perform Kmeans with 3 clusters to find 3 different dominant colors
% We have to set rng to 1 to ensure clusters are consistent between runs
rng(1);
num_of_clusters = 3;
[idx, C] = kmeans(img_data, num_of_clusters);

% Sort the clusters by their size
cluster_sizes = accumarray(idx(:), 1);
[~, sorted_idx] = sort(cluster_sizes, 'descend');

% Get the centroids (color) of the clusters
dominant_color = C(sorted_idx(1), :);
secondary_color = C(sorted_idx(2), :);
tertiary_color = C(sorted_idx(3), :);

% Normalize the colors 
[dom_l_val, dom_b_val, dom_c_val] = normalize_lab_values(dominant_color);
[sec_l_val, sec_b_val, sec_c_val] = normalize_lab_values(secondary_color);
[ter_l_val, ter_b_val, ter_c_val] = normalize_lab_values(tertiary_color);

% Add all the features together and return them
feature_vector = [
    avg_l_val...
    avg_a_val...
    avg_b_val...
    dom_l_val...
    dom_b_val...
    dom_c_val...
    sec_l_val...
    sec_b_val...
    sec_c_val...
    ter_l_val...
    ter_b_val...
    ter_c_val...
    size_ratio...
    eccentricity];
end