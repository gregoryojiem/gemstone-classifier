function train_classifier(classifier_type, output_filename)
%TRAIN_CLASSIFIER Generates training data, uses it to train a classifier,
% and saves the classifier to a file

% Get the directory with all of the training data
raw_data_folder = "NAZCA_SCANNED_GEMS";
training_data_folder = "training_data";
testing_data_folder = "testing_data";

% Check to make sure input arguments were given correctly
if nargin < 2
    error('Input arguments are required: <classifier-type> <output-file>')
end

if classifier_type ~= "dt" && classifier_type ~= "resnet50"
    error('Classifier type must be dt for a decision tree, or resnet50')
end

% Set rng to 1 to ensure consistent results
rng(1);

% Load in the training images as a datastore
% Get a list of the labels of the images
% Split the data into a training, and a testing set
imds = imageDatastore(raw_data_folder, 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');
class_labels = categories(imds.Labels);
[imds_training_set, imds_testing_set] = splitEachLabel(imds,0.8);

% If normalized training images don't exist, generate them
% imds is stored with NAZCA_SCANNED_GEMS as the root folder
% We rename it and fix the file structure
if ~exist(training_data_folder, 'dir') 
    
    % Add a transform to the imds to make it pre-process the image
    % before we write it to a file
    imds_transform = transform(imds_training_set, @(x) preprocess_image(x));

    % Write the pre-processed images to file
    % Parallel processing is enabled but can be disabled if necessary
    writeall(imds_transform, training_data_folder,...
        'OutputFormat', 'png', 'UseParallel', true);

    % Fix file structure
    movefile training_data/NAZCA_SCANNED_GEMS...
        training_data/training_data % rename the folder
    movefile training_data/training_data ./ % move folder up a level
end

% Load in the pre-processed images
imds_training_set = imageDatastore(training_data_folder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% If testing data doesn't exist
% Save the testing set to a folder so it can be used in other functions
if ~exist(testing_data_folder, 'dir') 
    writeall(imds_testing_set, "testing_data", 'UseParallel', true);
    movefile testing_data/NAZCA_SCANNED_GEMS testing_data/testing_data % rename
    movefile testing_data/testing_data ./ % move folder level up a level
end

% Train a classifier and save it to a file
disp('Training classifier');
if classifier_type == "dt"
    disp('Generating training data');

    % Add a transform to call generate_features on each processed image
    imds_transform = transform(imds_training_set, @(x) generate_features(x));

    % Transform all the images to features and get their labels
    training_data = readall(imds_transform, UseParallel=true); 
    labels = char(imds_training_set.Labels);

    disp('Begining decision tree training');

    % Make the decision tree and optimize all of the hyperparameters 
    dt = fitctree(training_data, labels, 'OptimizeHyperparameters', 'all');

    % Display the tree and its splits
    view(dt,'Mode', 'graph');

    % Calculate statistics about its accuracy, perform cross-validation
    % This step isn't necessary but it's useful for debugging
    resub_loss = resubLoss(dt);
    cv_tree = crossval(dt);
    cv_loss = kfoldLoss(cv_tree);
    disp(['Cross validation loss:', num2str(cv_loss)]);
    disp(['Resubstitution loss:', num2str(resub_loss)]);
    disp('Saving classifier to file');

    % Save the decision tree to the output filename
    save(output_filename, 'dt');
else
    % Split the training set into a training set and a validation set
    % 0.8125 splits the overall data into 0.65 training, 0.15 validation
    [imds_training_set, imds_validation_set] = ...
        splitEachLabel(imds_training_set, 0.8125);

    % Make an augmenter to improve robustness by adding reflections and
    % translations
    augmenter = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandYReflection=true);
    
    % Augment the training set
    aug_training_set = augmentedImageDatastore(...
        [224, 224, 3], imds_training_set, 'DataAugmentation', augmenter);

    % Load in a pretrained resnet50 NN with an output layer = num_classes
    % This follows the new format for loading in a neural net in
    % MATLAB2024, so that version is required. It's not recommended
    % anymore, but its possible to instead follow the same code but change
    % net to net = resnet50, and adjust the output layers.
    num_classes = numel(class_labels);
    net = imagePretrainedNetwork('resnet50', NumClasses=num_classes);

    % Set up the training options for the neural net
    % It typically won't finish all epochs, validation patience is set to 3
    % and frequency to 1 so it stops when validation improvement plateaus
    % MiniBatchSize can be adjusted depending on available GPU memory
    options = trainingOptions("adam", ...
        MaxEpochs=10, ...
        MiniBatchSize=64, ...
        InitialLearnRate=0.0001, ...
        ValidationData=imds_validation_set, ...
        ValidationFrequency=1, ...
        ValidationPatience=3,...
        Plots="training-progress", ...
        Metrics="accuracy", ...
        Verbose=false);

    % Train the network with training options using cross-entropy loss
    net = trainnet(aug_training_set, net, "crossentropy", options);

    % Save the network to the given output filename
    save(output_filename, 'net', 'class_labels');

    % See how the network performs on the small testing set reserved
    % We add a transform to make sure the images are pre-processed first
    imds_test_transform = transform(imds_testing_set, @(x) preprocess_image(x));
    test_results = minibatchpredict(net, imds_test_transform);
    accuracy = mean(imds_testing_set.Labels == scores2label(test_results, class_labels))
end
end