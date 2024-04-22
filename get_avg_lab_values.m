function [avg_l_norm, avg_a_norm, avg_b_norm] = get_avg_lab_values(gem_img, gem_mask)
%GET_AVG_LAB_VALUES Finds the average CIELAB values of an image

% Convert the image to CIELAB space and split it into lightness, a*, and b*
lab_img = rgb2lab(gem_img);
[l_vals, a_vals, b_vals] = imsplit(lab_img); 

% Isolate the gem pixels using the mask, so we only find their average
l_vals = l_vals(gem_mask);
a_vals = a_vals(gem_mask);
b_vals = b_vals(gem_mask);

% Get the average lightness value, and the average a* and b* values
avg_l = sum(l_vals)/size(l_vals, 1);
avg_a = sum(a_vals)/size(a_vals, 1);
avg_b = sum(b_vals)/size(b_vals, 1);

% Normalize the LAB values and return them
[avg_l_norm, avg_a_norm, avg_b_norm] = normalize_lab_values([avg_l, avg_a, avg_b]);
end