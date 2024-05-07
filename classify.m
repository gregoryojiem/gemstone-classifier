function classification = classify(filename)
%CLASSIFY Loads in a pre-trained classifier and classifies the image given
%by filename
if ~exist(filename, "file")
    error('Please enter a valid path to an image');
end

% Perform pre-processing on the image
gem_img = preprocess_image(imread(filename));

% Load in a classifier; this can be a neural network or a decision tree
% decision_tree.mat and resnet.mat are defined and can be used here
load decision_tree.mat;

% Check what type the classifier is, and predict the image class
if exist('net', 'var') 
    scores = predict(net, single(gem_img));
    [label,score] = scores2label(scores,class_labels);
    label = strtrim(string(label));
elseif exist('dt', 'var') 
    feature_vector = generate_features(gem_img);
    predicted_class = predict(dt, feature_vector);
    label = strtrim(string(predicted_class));
end 

% Set the classification equal to the predicted class label and display it
classification = label;
disp(label);
end

