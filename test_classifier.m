function test_classifier(filename)
%TEST_CLASSIFIER Tests a decision tree or resnet50 network and displays
%basic accuracy stats

% Check if the directory exists
if ~exist(filename, 'dir')
    error('There is no directory at the given filename.');
end

% Variables to keep track of accuracy
correct_count = 0;
total_count = 0;

% Loop over each subfolder for each type of gem
% Start at i=3 to avoid looping over current/parent directories
main_folder = dir(filename);
for i=3 : length(main_folder)
   subfolder = main_folder(i).name;
   subfolder_path = strcat(filename, "/", subfolder);
   gem_images = dir(subfolder_path);
    
   % Loop over the testing images
   for j=3 : length(gem_images)
       % Get the path of the image to classify
       gem_path = strcat(subfolder_path, "/", gem_images(j).name);
       
       % Classify the image and update classification statistics
       classification = classify(gem_path);
       if strcmp(classification, subfolder)
           correct_count = correct_count + 1;
       end
       total_count = total_count + 1;
   end
end

% Display stats about accuracy
disp(strcat('Accuracy: ', num2str(correct_count/total_count*100), "%"))
end