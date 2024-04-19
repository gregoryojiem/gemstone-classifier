function [samples, labels] = generate_training_data(main_folder_name)
%GENERATE_TRAINING_DATA Generates a list of training samples with classes
% Given a main folder containing subdirectories of different classes, this
% function generates features for each image inside of each subdirectory,
% and returns those features + class labels.
main_folder = dir(main_folder_name);
write_normalized_images_to_file = false;
num_of_features = 5;
num_of_samples = 1000;

% Set up features and classifications arrays, preallocate 1000 entries
samples = zeros(num_of_samples, num_of_features);
labels = cell(num_of_samples, 1);

% Keeps track of how many examples we've processed so far
sample_count = 1;

% Loop over each subfolder for each type of gem
% Start at i=3 to avoid looping over current/parent directories
for i=3 : length(main_folder)
   subfolder_name = strcat(main_folder_name, "/", main_folder(i).name);
   gem_images = dir(subfolder_name);
    
   % Loop over the training images and generate normalized training images
   for j=3 : length(gem_images)
       file_name = strcat(subfolder_name, "/", gem_images(j).name);
       path_to_folder = strcat("training_data", "/", main_folder(i).name);

       % Get the path where the gem/mask would be stored if saved before
       path_to_gem = strcat(path_to_folder, "/", num2str(sample_count), ".png");
       path_to_mask = strcat(path_to_folder, "/", num2str(sample_count), "_mask", ".png");

       % If the training image has been saved before, reuse it
       % Otherwise, process it from scratch.
       if exist(path_to_gem, 'file')
           normalized_img = im2double(imread(path_to_gem));
           gem_mask = imread(path_to_mask);
       else
           [normalized_img, gem_mask] = preprocess_image(file_name);
       end

       % If the flag is true write the image/mask to file for debugging
       if write_normalized_images_to_file && ~exist(path_to_folder, 'dir')
            mkdir(path_to_folder);
       end

       if write_normalized_images_to_file
         imwrite(normalized_img, path_to_gem);
         imwrite(gem_mask, path_to_mask);
       end

       % Generate and save features for the normalized image
       samples(sample_count, :) = generate_features(normalized_img, gem_mask);
       labels{sample_count} = main_folder(i).name;
       sample_count = sample_count + 1;
   end
end

% Resize features and labels to their correct size
samples = samples(1:sample_count-1, :);
labels = labels(1:sample_count-1);
end

