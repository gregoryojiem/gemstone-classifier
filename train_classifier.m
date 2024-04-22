function train_classifier(classifier_type, output_filename)
%TRAIN_CLASSIFIER Generates training data, uses it to train a classifier,
% and saves the classifier to a file

% Get the directory with all of the training data
main_folder_name = "NAZCA_SCANNED_GEMS";

% Check to make sure input arguments were given correctly
if nargin < 2
    error('Input arguments are required: <classifier-type> <output-file>')
end

if classifier_type ~= "dt" && classifier_type ~= "resnet50"
    error('Classifier type must be dt for a decision tree, or resnet50')
end

% Generate the training data for a classifier
disp('Generating training data');
[training_data, labels] = generate_training_data(main_folder_name);

% Train a classifier and save it to a file
disp('Training classifier');
if classifier_type == "dt"
    dt = fitctree(training_data, labels,'OptimizeHyperparameters','all');
    view(dt,'Mode','graph');
    resub_loss = resubLoss(dt);
    cv_tree = crossval(dt);
    cv_loss = kfoldLoss(cv_tree);
    disp(['Cross validation loss:', num2str(cv_loss)]);
    disp(['Resubstitution loss:', num2str(resub_loss)]);
    disp('Saving classifier to file');
    save(output_filename, 'dt');
else
    %TODO RESNET 50
end
end