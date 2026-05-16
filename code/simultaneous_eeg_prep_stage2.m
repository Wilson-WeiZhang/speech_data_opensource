%% simultaneous_eeg_prep_stage2.m â€” SI EEG 5u: ICA + ICLabel + MARA
%% Run: matlab -batch "addpath('/path/to/eeglab'); cd('/path/to/code'); simultaneous_eeg_prep_stage2"

if ~exist('eeglab', 'file')
    addpath('/path/to/eeglab2024');  % <-- SET YOUR EEGLAB PATH
    eeglab nogui;
end

data_dir = '../../data/prep_si_eeg/';
files = dir(fullfile(data_dir, '*_processed_trials.set'));
fprintf('Found %d files for ICA\n', length(files));

parfor i = 1:length(files)
    name = files(i).name(1:5);
    out_file = fullfile(data_dir, [name '_precut_ICA.set']);
    if exist(out_file, 'file')
        fprintf('SKIP %s\n', name);
        continue;
    end

    fprintf('  %d/%d: %s\n', i, length(files), name);
    EEG = pop_loadset('filename', files(i).name, 'filepath', data_dir);

    EEG = pop_runica(EEG, 'icatype', 'runica', 'interrupt', 'on');
    EEG = pop_iclabel(EEG, 'default');

    ALLEEG = []; CURRENTSET = [];
    [ALLEEG, EEG, CURRENTSET] = processMARA(ALLEEG, EEG, CURRENTSET);

    EEG = pop_saveset(EEG, 'filename', [name '_precut_ICA.set'], 'filepath', data_dir);
    fprintf('  %s done\n', name);
end

fprintf('\n===== SI 5u c2 complete =====\n');
