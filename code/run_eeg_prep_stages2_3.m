%% run_eeg_prep_stages2_3.m - Run stage2 (ICA, parfor 16) + stage3 for both ST and SI
%% Run after stage1 completes:
%% matlab -batch "addpath('/path/to/eeglab'); cd('/path/to/code'); run_eeg_prep_stages2_3"

% Start parallel pool (use available cores, up to 16)
n_workers = min(16, feature('numcores'));
p = gcp('nocreate');
if isempty(p)
    parpool('local', n_workers);
elseif p.NumWorkers ~= n_workers
    delete(p);
    parpool('local', n_workers);
end

fprintf('===== ST 5u stage2 (ICA) =====\n');
standalone_eeg_prep_stage2;

fprintf('===== SI 5u stage2 (ICA) =====\n');
simultaneous_eeg_prep_stage2;

% Close parpool before stage3 (serial)
delete(gcp('nocreate'));

fprintf('===== ST 5u stage3 (clean) =====\n');
standalone_eeg_prep_stage3;

fprintf('===== SI 5u stage3 (clean) =====\n');
simultaneous_eeg_prep_stage3;

% Summary
st_done = length(dir(fullfile('../../data/prep_st_eeg/', '*_clean.set')));
si_done = length(dir(fullfile('../../data/prep_si_eeg/', '*_clean.set')));
fprintf('\n===== ALL DONE =====\n');
fprintf('ST 5u: %d clean files (63ch, ECG removed)\n', st_done);
fprintf('SI 5u: %d clean files (63ch, ECG removed)\n', si_done);
