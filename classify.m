function classification = classify(filename)
%CLASSIFY Loads in a pre-trained classifier and classifies the image given
%by filename
if ~exist(filename, "file")
    error('Please enter a valid path to an image')
end

% Preprocess the image and generate features from it
[gem_img, gem_mask] = preprocess_image(filename);
feature_vector = generate_features(gem_img, gem_mask);

% Load in the classifier and get a classification
decision_tree = load("classifier.mat").dt;
predicted_class = predict(decision_tree, feature_vector);
classification = predicted_class{1};
disp(classification);
end

