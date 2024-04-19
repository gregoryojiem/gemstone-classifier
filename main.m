function main
%MAIN Driver function that makes the calls to generate training data and
%train a classifier

% Get the directory with all of the training data
main_folder_name = "NAZCA_SCANNED_GEMS";

% Generate the training data for a classifier
[samples, labels] = generate_training_data(main_folder_name)

% Train a classifier
% TODO

% Save classifier to a file
% TODO
end