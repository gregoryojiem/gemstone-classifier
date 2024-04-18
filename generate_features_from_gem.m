function feature_vector = generate_features_from_gem(file_name)
%UNTITLED Summary of this function goes here
[processed_img, gem_mask] = preprocess_image(imread(file_name));
imshow(processed_img);
feature_vector = processed_img;
end