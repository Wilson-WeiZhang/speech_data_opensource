%% simultaneous_eeg_prep_stage3.m â€” SI EEG 5u: clean (automatic brain > 0.9)
%% Logic: remove ECG only after IC rejection, retaining 63 scalp EEG channels.
%% Run: matlab -batch "addpath('/path/to/eeglab'); cd('/path/to/code'); simultaneous_eeg_prep_stage3"

if ~exist('eeglab', 'file')
    addpath('/path/to/eeglab2024');  % <-- SET YOUR EEGLAB PATH
    eeglab nogui;
end

data_dir = '../../data/prep_si_eeg/';
d = dir(fullfile(data_dir, '*_precut_ICA.set'));
fprintf('Found %d precut_ICA files\n', length(d));

threshold = 0.9;
skipped = {};

for sub = 1:length(d)
    name = d(sub).name(1:5);
    out_file = fullfile(data_dir, [name '_clean.set']);
    if exist(out_file, 'file')
        fprintf('SKIP %s (exists)\n', name);
        continue;
    end

    fprintf('=== %d/%d: %s ===\n', sub, length(d), name);
    EEG = pop_loadset('filename', d(sub).name, 'filepath', data_dir);

    % ICLabel: brain > 0.9
    t_brain = find(EEG.etc.ic_classification.ICLabel.classifications(:, 1) > threshold)';

    if isempty(t_brain)
        fprintf('  NO brain components > %.1f â€” SKIPPED\n', threshold);
        skipped{end+1} = name;
        continue;
    end

    % Artifacts > 0.9
    t_mus   = find(EEG.etc.ic_classification.ICLabel.classifications(:, 2) > threshold)';
    t_eye   = find(EEG.etc.ic_classification.ICLabel.classifications(:, 3) > threshold)';
    t_heart = find(EEG.etc.ic_classification.ICLabel.classifications(:, 4) > threshold)';
    t_line  = find(EEG.etc.ic_classification.ICLabel.classifications(:, 5) > threshold)';
    t_chan  = find(EEG.etc.ic_classification.ICLabel.classifications(:, 6) > threshold)';

    % Fallback: if no IC exceeds heart threshold, remove the IC with the
    % highest heart probability among the first 5 ICs (cardiac artifact is
    % typically captured by early ICs in EEG recordings).
    if isempty(t_heart)
        [~, t_heart] = max(EEG.etc.ic_classification.ICLabel.classifications(1:min(5,end), 4));
    end

    all_artifacts = unique([t_mus, t_eye, t_heart, t_line, t_chan]);
    brain = setdiff(t_brain, all_artifacts);

    % Remove low-brain (< 0.1)
    low_brain = find(EEG.etc.ic_classification.ICLabel.classifications(:, 1) < 0.1)';
    low_brain = setdiff(low_brain, all_artifacts);
    brain = setdiff(brain, low_brain);

    if isempty(brain)
        fprintf('  NO brain components after filtering â€” SKIPPED\n');
        skipped{end+1} = name;
        continue;
    end

    fprintf('  Brain components: %d (of %d total)\n', length(brain), size(EEG.icawinv, 2));

    EEG = pop_subcomp(EEG, brain, 0, 1);
    EEG = pop_select(EEG, 'nochannel', {'ECG'});
    EEG = eeg_checkset(EEG);
    EEG = pop_reref(EEG, []);

    EEG = pop_saveset(EEG, 'filename', [name '_clean.set'], 'filepath', data_dir);
    fprintf('  Saved (%d ch, %d epochs)\n', EEG.nbchan, EEG.trials);
end

fprintf('\n===== SI 5u c3 complete =====\n');
if ~isempty(skipped)
    fprintf('Skipped subjects (no brain > %.1f): %s\n', threshold, strjoin(skipped, ', '));
end
