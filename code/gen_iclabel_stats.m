%% ICLabel classification statistics for standalone EEG (5-utterance pipeline)
%  Computes per-subject IC counts by ICLabel category and group summary.
%  Input:  *_precut_ICA.set files from prep_st_eeg/
%  Output: printed summary statistics for manuscript reporting
%
%  ICLabel categories: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other
%  Each IC is assigned to the category with the highest posterior probability.

clear; clc;

%% Path setup
if exist('../../data/prep_st_eeg/', 'dir')
    data_path = '../../data/prep_st_eeg/';
elseif exist('data/prep_st_eeg/', 'dir')
    data_path = 'data/prep_st_eeg/';
else
    error('Cannot find prep_st_eeg data. Set data_path to your prep_st_eeg directory.');
end

%% Find all pre-ICA files
files = dir(fullfile(data_path, '*_precut_ICA.set'));
num_subjects = length(files);
fprintf('Found %d subjects with ICA data\n', num_subjects);

cat_names = {'Brain', 'Muscle', 'Eye', 'Heart', 'Line', 'Channel', 'Other'};
num_cats = 7;

%% Collect IC distributions
subject_ids = cell(num_subjects, 1);
ic_counts = zeros(num_subjects, num_cats);
ic_totals = zeros(num_subjects, 1);

for s = 1:num_subjects
    name = files(s).name(1:5);
    subject_ids{s} = name;
    EEG = pop_loadset('filename', files(s).name, 'filepath', data_path, 'loadmode', 'info');

    if ~isfield(EEG.etc, 'ic_classification') || ...
       ~isfield(EEG.etc.ic_classification, 'ICLabel')
        fprintf('  %s: NO ICLabel data, skipping\n', name);
        continue;
    end

    cls = EEG.etc.ic_classification.ICLabel.classifications;
    n_ic = size(cls, 1);
    ic_totals(s) = n_ic;

    [~, max_cat] = max(cls, [], 2);
    for c = 1:num_cats
        ic_counts(s, c) = sum(max_cat == c);
    end

    fprintf('  %s: %2d ICs | Brain=%2d Eye=%d Muscle=%d Heart=%d Line=%d Chan=%d Other=%d\n', ...
        name, n_ic, ic_counts(s,1), ic_counts(s,3), ic_counts(s,2), ...
        ic_counts(s,4), ic_counts(s,5), ic_counts(s,6), ic_counts(s,7));
end

%% Summary statistics
valid = ic_totals > 0;
N = sum(valid);
fprintf('\n========================================\n');
fprintf('ICLabel Summary (N=%d, 5-utterance pipeline)\n', N);
fprintf('========================================\n');
fprintf('Total ICs per subject: %.1f +/- %.1f\n', ...
    mean(ic_totals(valid)), std(ic_totals(valid)));
for c = 1:num_cats
    m = mean(ic_counts(valid, c));
    s = std(ic_counts(valid, c));
    pct = mean(ic_counts(valid, c) ./ ic_totals(valid)) * 100;
    fprintf('  %-8s: %5.1f +/- %4.1f  (%4.1f%%)\n', cat_names{c}, m, s, pct);
end
fprintf('========================================\n');
