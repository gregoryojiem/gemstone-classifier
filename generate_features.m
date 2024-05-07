function feature_vector = generate_features(gem_img)
%GENERATE_FEATURES Generates features to help classify an image
% Uses the gem image and the mask image to calculate various features for 
% training a decision tree, features like average color, the size of the 
% gem compared to the background, eccentricity of the gem, etc. 

% Convert the image to CIELAB space and split it into lightness, a*, and b*
lab_img = rgb2lab(gem_img);
[l_vals, a_vals, b_vals] = imsplit(lab_img); 

% Get a mask of the gem
gem_mask = bwconvhull(rgb2gray(gem_img) > 0);

% Isolate the gem pixels using the mask, so we only find their average
l_vals = l_vals(gem_mask);
a_vals = a_vals(gem_mask);
b_vals = b_vals(gem_mask);

% Get the average lightness value, and the average a* and b* values
avg_l = sum(l_vals)/size(l_vals, 1);
avg_a = sum(a_vals)/size(a_vals, 1);
avg_b = sum(b_vals)/size(b_vals, 1);

% Calculate the ratio of foreground (gem pixels) to background pixels
num_of_pixels = numel(gem_mask);
num_of_gem_pixels = sum(gem_mask(:));
size_ratio = num_of_gem_pixels/num_of_pixels;

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
[sorted_cluster_sizes, sorted_idx] = sort(cluster_sizes, 'descend');

% Get the centroids (color) of the clusters
dominant_color = C(sorted_idx(1), :);
secondary_color = C(sorted_idx(2), :);
tertiary_color = C(sorted_idx(3), :);

% End member analysis
% Define the exemplars
white_avg = [1.00, 1.00, 1.00];  % White
gray_avg  = [0.50, 0.50, 0.50];  % Gray
black_avg = [0.00, 0.00, 0.00];  % Black
red_avg   = [1.00, 0.00, 0.00];  % Red
green_avg = [0.00, 1.00, 0.00];  % Green
blue_avg  = [0.00, 0.00, 1.00];  % Blue
cyan_avg  = [0.00, 1.00, 1.00];  % Cyan
magenta_avg= [1.00, 0.00, 1.00];  % Magenta
yellow_avg = [1.00, 1.00, 0.00];  % Yellow
brown_avg  = [0.35, 0.22, 0.15];  % Brown
orange_avg = [1.00, 0.60, 0.00];  % Orange
pink_avg   = [1.00, 0.72, 0.84];  % Pink

% Compute Euclidean distance to each class exemplar.
gem_img = im2double(gem_img);
D1 = sqrt((gem_img(:,:,1)-white_avg(1)).^2   + (gem_img(:,:,2)-white_avg(2)).^2    + (gem_img(:,:,3)-white_avg(3)).^2);
D2 = sqrt((gem_img(:,:,1)-gray_avg(1)).^2    + (gem_img(:,:,2)-gray_avg(2)).^2     + (gem_img(:,:,3)-gray_avg(3)).^2);
D3 = sqrt((gem_img(:,:,1)-black_avg(1)).^2   + (gem_img(:,:,2)-black_avg(2)).^2    + (gem_img(:,:,3)-black_avg(3)).^2);
D4 = sqrt((gem_img(:,:,1)-red_avg(1)).^2     + (gem_img(:,:,2)-red_avg(2)).^2      + (gem_img(:,:,3)-red_avg(3)).^2);
D5 = sqrt((gem_img(:,:,1)-green_avg(1)).^2   + (gem_img(:,:,2)-green_avg(2)).^2    + (gem_img(:,:,3)-green_avg(3)).^2);
D6 = sqrt((gem_img(:,:,1)-blue_avg(1)).^2    + (gem_img(:,:,2)-blue_avg(2)).^2     + (gem_img(:,:,3)-blue_avg(3)).^2);
D7 = sqrt((gem_img(:,:,1)-cyan_avg(1)).^2    + (gem_img(:,:,2)-cyan_avg(2)).^2     + (gem_img(:,:,3)-cyan_avg(3)).^2);
D8 = sqrt((gem_img(:,:,1)-magenta_avg(1)).^2 + (gem_img(:,:,2)-magenta_avg(2)).^2  + (gem_img(:,:,3)-magenta_avg(3)).^2);
D9 = sqrt((gem_img(:,:,1)-yellow_avg(1)).^2  + (gem_img(:,:,2)-yellow_avg(2)).^2   + (gem_img(:,:,3)-yellow_avg(3)).^2);
D10 = sqrt((gem_img(:,:,1)-brown_avg(1)).^2  + (gem_img(:,:,2)-brown_avg(2)).^2    + (gem_img(:,:,3)-brown_avg(3)).^2);
D11 = sqrt((gem_img(:,:,1)-orange_avg(1)).^2 + (gem_img(:,:,2)-orange_avg(2)).^2   + (gem_img(:,:,3)-orange_avg(3)).^2);
D12 = sqrt((gem_img(:,:,1)-pink_avg(1)).^2   + (gem_img(:,:,2)-pink_avg(2)).^2     + (gem_img(:,:,3)-pink_avg(3)).^2);

% Build distance matrix with a channel for each color we're classifying
distances = D1;
distances(:,:,2) = D2;
distances(:,:,3) = D3;
distances(:,:,4) = D4;
distances(:,:,5) = D5;
distances(:,:,6) = D6; 
distances(:,:,7) = D7;
distances(:,:,8) = D8;
distances(:,:,9) = D9;
distances(:,:,10) = D10;
distances(:,:,11) = D11; 
distances(:,:,12) = D12;

% Find the id of the closest distance
[~, min_id] = min(distances, [], 3);

% Get the sums of the various colors of pixels
bg_pixels = (num_of_pixels - num_of_gem_pixels);
white_percentage  = nnz(min_id(:) == 1) / num_of_pixels;
black_percentage  = (nnz(min_id(:) == 3) - bg_pixels) / num_of_pixels;
gray_percentage   = nnz(min_id(:) == 2) / num_of_pixels; 
red_percentage    = nnz(min_id(:) == 4) / num_of_pixels;
green_percentage  = nnz(min_id(:) == 5) / num_of_pixels;
blue_percentage   = nnz(min_id(:) == 6) / num_of_pixels;
cyan_percentage   = nnz(min_id(:) == 7) / num_of_pixels;
magenta_percentage= nnz(min_id(:) == 8) / num_of_pixels;
yellow_percentage = nnz(min_id(:) == 9) / num_of_pixels;
brown_percentage  = nnz(min_id(:) == 10) / num_of_pixels;
orange_percentage = nnz(min_id(:) == 11) / num_of_pixels;
pink_percentage   = nnz(min_id(:) == 12) / num_of_pixels;

% Add all the features together and return them
feature_vector = [
    avg_l...
    avg_a...
    avg_b...
    white_percentage...
    black_percentage...
    gray_percentage...
    cyan_percentage... 
    magenta_percentage...
    yellow_percentage...
    red_percentage... 
    green_percentage...
    blue_percentage...
    brown_percentage... 
    orange_percentage...
    pink_percentage...
    dominant_color(1)...
    dominant_color(2)...
    dominant_color(3)...
    sorted_cluster_sizes(1)...
    secondary_color(1)...
    secondary_color(2)...
    secondary_color(3)...
    sorted_cluster_sizes(2)...
    tertiary_color(1)...
    tertiary_color(2)...
    tertiary_color(3)...
    sorted_cluster_sizes(3)...
    eccentricity...
    size_ratio
];
end