function test_classifier(filename)
%TEST_CLASSIFIER Tests a decision tree or resnet50 network and displays
%basic accuracy stats

% Check if the directory exists
if ~exist(filename, 'dir')
    error('There is no directory at the given filename.');
end

% Variables to keep track of accuracy
number_of_test_samples = 300;
correct_count = 0;
total_count = 0;
label_count = 1;
true_labels = cell(number_of_test_samples, 1);
predicted_labels = {};
class_labels = {};

% Loop over each subfolder for each type of gem
% Start at i=3 to avoid looping over current/parent directories
main_folder = dir(filename);
for i=3 : length(main_folder)
   subfolder = string(main_folder(i).name);
   subfolder_path = strcat(filename, "/", subfolder);
   gem_images = dir(subfolder_path);
    
   % If the class label hasn't been seen before add it to class label list
   if length(gem_images) > 2 && ~ismember(subfolder, class_labels)
       class_labels{label_count} = char(subfolder);
       label_count = label_count + 1;
   end

   % Loop over the testing images
   for j=3 : length(gem_images)
       % Get the path of the image to classify
       gem_path = strcat(subfolder_path, "/", gem_images(j).name);
       
       % Classify the image
       classification = char(classify(gem_path));

       % If the classification class label hasn't been seen before, add to
       % class label list
       if ~ismember(classification, class_labels)
           class_labels{label_count} = classification;
           label_count = label_count + 1;
       end

       % Update classification statistics 
       if strcmp(classification, subfolder)
           correct_count = correct_count + 1;
       end
       total_count = total_count + 1;
       true_labels{total_count} = char(subfolder);
       predicted_labels{total_count} = classification;
   end
end

% Adjust preallocated size to actual size
true_labels = true_labels(1:total_count-1);
predicted_labels = predicted_labels(1:total_count-1);

% Sort class labels in descending alphabetical order
class_labels = sort(class_labels);

% Display stats about accuracy
disp(strcat('Accuracy: ', num2str(correct_count/total_count*100), "%"));
confusion_matrix = confusionmat(true_labels, predicted_labels);
confusionchart(confusion_matrix, class_labels(1:label_count-1));
end