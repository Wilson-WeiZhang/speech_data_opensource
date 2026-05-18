%% simultaneous_eeg_prep_stage1.m - SI EEG: 5-utterance version
%% BCG corrected -> resample 250 -> 1-30 Hz -> epoch covert 5utt -> save
%% Run: matlab -batch "addpath('/path/to/eeglab'); cd('/path/to/code'); simultaneous_eeg_prep_stage1"

addpath('/path/to/eeglab2024');  % <-- SET YOUR EEGLAB PATH
eeglab nogui;

data_dir = '../../data/bcg_si_eeg/';       % <-- SET YOUR RAW DATA PATH
output_dir = '../../data/prep_si_eeg/';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

subj_dirs = dir(fullfile(data_dir, 'S0*'));
subj_dirs = subj_dirs([subj_dirs.isdir]);
fprintf('Found %d SI subjects\n', length(subj_dirs));

valid_event_types = {'S  1', 'S  2', 'S  3', 'S  4', 'S  5'};
event_types = {'C 1', 'C 2', 'C 3', 'C 4', 'C 5'};
pre_event_time = 0.5;
post_event_time = 1.5;

for subj = 1:length(subj_dirs)
    name = subj_dirs(subj).name;
    vhdr_file = dir(fullfile(data_dir, name, '*_BCG_Correction.vhdr'));
    if isempty(vhdr_file)
        fprintf('SKIP %s: no BCG file\n', name);
        continue;
    end

    out_file = fullfile(output_dir, [name '_processed_trials.set']);
    if exist(out_file, 'file')
        fprintf('SKIP %s (exists)\n', name);
        continue;
    end

    fprintf('=== %d/%d: %s ===\n', subj, length(subj_dirs), name);

    EEG = pop_loadbv(fullfile(data_dir, name), vhdr_file(1).name);
    EEG = pop_resample(EEG, 250);
    EEG = pop_eegfiltnew(EEG, 1, 30, [], 0, [], 0);  % 1-30 Hz, no notch

    % Keep only valid stimulus events
    valid_indices = find(ismember({EEG.event.type}, valid_event_types));
    if isempty(valid_indices)
        all_types = unique({EEG.event.type});
        fprintf('  WARNING: no stimulus events. Types: %s\n', strjoin(all_types, ', '));
        continue;
    end
    EEG.event = EEG.event(valid_indices);

    % Detect blocks
    latencies = [EEG.event.latency];
    latency_diffs = diff(latencies) / EEG.srate;
    block_breaks = find(latency_diffs > 20);

    block_numbers = ones(1, length(EEG.event));
    current_block = 1;
    for i = 1:length(block_breaks)
        block_numbers(block_breaks(i)+1:end) = current_block + 1;
        current_block = current_block + 1;
    end

    if max(block_numbers) ~= 10
        fprintf('  WARNING: Expected 10 blocks, found %d\n', max(block_numbers));
    end

    % Label O/C
    for j = 1:length(EEG.event)
        EEG.event(j).block_number = block_numbers(j);
        if mod(block_numbers(j), 2) == 1
            EEG.event(j).type = ['O' EEG.event(j).type(3:end)];
        else
            EEG.event(j).type = ['C' EEG.event(j).type(3:end)];
        end
    end

    % Remove duplicates (< 5s apart)
    to_remove = [];
    for j = 2:length(EEG.event)
        if (EEG.event(j).latency - EEG.event(j-1).latency) < 5 * EEG.srate
            to_remove = [to_remove, j];
        end
    end
    EEG.event(to_remove) = [];

    % Expand: 5 utterances per trial, 2s apart
    expanded_events = struct('type', {}, 'latency', {}, 'urevent', {}, 'block_number', {});
    for j = 1:length(EEG.event)
        blk = EEG.event(j).block_number;
        for k = 1:5
            new_event.type = [EEG.event(j).type '_u_' num2str(k) '_b_' num2str(blk)];
            new_event.latency = EEG.event(j).latency + (k-1) * 2 * EEG.srate;
            new_event.urevent = EEG.event(j).urevent;
            new_event.block_number = blk;
            expanded_events(end+1) = new_event;
        end
    end
    EEG.event = expanded_events;
    EEG = eeg_checkset(EEG, 'eventconsistency');

    % Epoch: covert only, all 5 utterances
    all_event_types = {};
    for j = 1:length(event_types)
        for k = 1:5
            for blk = [2 4 6 8 10]
                all_event_types{end+1} = [event_types{j} '_u_' num2str(k) '_b_' num2str(blk)];
            end
        end
    end

    EEG = pop_epoch(EEG, all_event_types, [-pre_event_time post_event_time], ...
        'newname', [name '_epochs'], 'epochinfo', 'yes');

    fprintf('  %d covert epochs (expected 500)\n', EEG.trials);
    EEG = pop_saveset(EEG, 'filename', [name '_processed_trials.set'], 'filepath', output_dir);
end

fprintf('\n===== SI 5u c1 complete =====\n');
