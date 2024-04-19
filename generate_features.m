function feature_vector = generate_features(gem_img, gem_mask)
%GENERATE_FEATURES Generates features to help classify an image
% Uses the gem image and the mask image to calculate various features for 
% training a decision tree, features like average color, the size of the 
% gem compared to the background, eccentricity of the gem, etc. 

% Convert the image to CIELAB space and split it into lightness, a*, and b*
lab_img = rgb2lab(gem_img);
[l_vals, a_vals, b_vals] = imsplit(lab_img); 

% Normalize all of the lab values to be between 0 and 1
l_vals = l_vals(gem_mask)/100;
a_vals = (a_vals(gem_mask)/128 + 1)/2;
b_vals = (b_vals(gem_mask)/128 + 1)/2;

% Get the average lightness value, and the average a* and b* values
avg_lightness = sum(l_vals)/size(l_vals, 1);
avg_a_val = sum(a_vals)/size(a_vals, 1);
avg_b_val = sum(b_vals)/size(b_vals, 1);

% Calculate the ratio of foreground (gem pixels) to background pixels
num_of_pixels = numel(lab_img)/3;
size_ratio = numel(l_vals)/num_of_pixels;

% Calculate eccentricity
eccentricity = regionprops(gem_mask, "Eccentricity").Eccentricity;

% Add all the features together and return
feature_vector = [
    avg_lightness...
    avg_a_val...
    avg_b_val...
    size_ratio...
    eccentricity];
end