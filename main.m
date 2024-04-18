function main
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Get the directory with all of the training data
main_folder_name = "NAZCA_SCANNED_GEMS";
main_folder = dir(main_folder_name);

% Loop over each subfolder for each type of gem
% Start at i=3 to avoid looping over current/parent directories
for i=3 : length(main_folder)
   subfolder_name = strcat(main_folder_name, "/", main_folder(i).name);
   gem_images = dir(subfolder_name);

   % Loop over the training images and generate features for each
   for j=3 : length(gem_images)
       file_name = strcat(subfolder_name, "/", gem_images(j).name);
       path_to_folder = strcat("test_images", "/", main_folder(i).name);
       if ~exist(path_to_folder, 'dir')
            mkdir(path_to_folder);
       end
       imwrite(generate_features_from_gem(file_name), strcat(path_to_folder, "/", gem_images(j).name));
   end
end
 
end