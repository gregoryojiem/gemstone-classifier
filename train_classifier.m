function train_classifier(classifier_type, output_filename)
%TRAIN_CLASSIFIER Generates training data, uses it to train a classifier,
% and saves the classifier to a file

% Get the directory with all of the training data
raw_data_folder = "NAZCA_SCANNED_GEMS";
training_data_folder = "training_data";

% Check to make sure input arguments were given correctly
if nargin < 2
    error('Input arguments are required: <classifier-type> <output-file>')
end

if classifier_type ~= "dt" && classifier_type ~= "resnet50"
    error('Classifier type must be dt for a decision tree, or resnet50')
end

% Train a classifier and save it to a file
disp('Training classifier');
if classifier_type == "dt"
    disp('Generating training data');

    % Generate the training data for a classifier
    [training_data, labels] = generate_training_data(raw_data_folder);

    % Make the decision tree and optimize all of the hyperparameters 
    dt = fitctree(training_data, labels,'OptimizeHyperparameters','all');

    % Display the tree and its splits
    view(dt,'Mode','graph');

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
    % If normalized training images don't exist, generate them
    if ~exist(training_data_folder, 'dir') 
        generate_training_data(raw_data_folder);
    end

    % Load in the training images as a datastore
    % Get a list of the labels of the images
    % Split the data into a training, validation, and testing set
    imds = imageDatastore(training_data_folder, 'IncludeSubfolders',true,...
        'LabelSource','foldernames');
    class_labels = categories(imds.Labels);
    [imds_training_set,imds_validation_set,imds_testing_set] = ...
        splitEachLabel(imds,0.7,0.15,'randomized');

    % Load in a pretrained resnet50 NN with an output layer = num_classes
    num_classes = numel(class_labels);
    net = imagePretrainedNetwork('resnet50', NumClasses=num_classes);

    % Make an augmenter to improve robustness by adding reflections and
    % translations
    augmenter = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandYReflection=true, ...
        RandXTranslation=[-25, 25], ...
        RandYTranslation=[-25, 25]);
    
    % Augment the training set
    aug_training_set = augmentedImageDatastore(...
        [224, 224, 3], imds_training_set, 'DataAugmentation',augmenter);
    
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
    net = trainnet(aug_training_set,net,"crossentropy",options);

    % See how the network performs on the small testing set reserved
    test_results = minibatchpredict(net,imds_testing_set);
    accuracy = mean(imds_testing_set.Labels == scores2label(test_results,class_names))

    % Save the network to the given output filename
    save(output_filename, 'net', 'class_labels');
end
end