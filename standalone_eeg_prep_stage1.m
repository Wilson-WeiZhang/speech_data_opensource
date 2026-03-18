%% standalone_eeg_prep_stage1.m — ST EEG: 5-utterance version
%% Raw → resample 250 → 1-100 Hz → notch 49-51 → epoch covert 5utt → save
%% Run: matlab -batch "addpath('/path/to/eeglab'); cd('/path/to/code'); standalone_eeg_prep_stage1"

addpath('/path/to/eeglab2024');  % <-- SET YOUR EEGLAB PATH
eeglab nogui;

data_dir = '../../data/raw_st_eeg/';       % <-- SET YOUR RAW DATA PATH
output_dir = '../../data/prep_st_5u/';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

file_list = dir(fullfile(data_dir, '*.vhdr'));
fprintf('Found %d raw ST files\n', length(file_list));

valid_event_types = {'S  1', 'S  2', 'S  3', 'S  4', 'S  5'};
event_types = {'C 1', 'C 2', 'C 3', 'C 4', 'C 5'};
pre_event_time = 0.5;
post_event_time = 1.5;

for subj = 1:length(file_list)
    name = file_list(subj).name(1:5);
    out_file = fullfile(output_dir, [name '_processed_trials.set']);
    if exist(out_file, 'file')
        fprintf('SKIP %s (exists)\n', name);
        continue;
    end

    fprintf('=== %d/%d: %s ===\n', subj, length(file_list), name);

    EEG = pop_loadbv(data_dir, file_list(subj).name);
    EEG = pop_resample(EEG, 250);
    EEG = pop_eegfiltnew(EEG, 1, 100, [], 0, [], 0);
    EEG = pop_eegfiltnew(EEG, 49, 51, [], 1, [], 0);  % notch

    % Keep only valid stimulus events
    valid_indices = find(ismember({EEG.event.type}, valid_event_types));
    EEG.event = EEG.event(valid_indices);

    % Detect blocks by latency gaps > 20s
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

    % Add block number to events
    for j = 1:length(EEG.event)
        EEG.event(j).block_number = block_numbers(j);
    end

    % Remove duplicate events (< 5s apart) — keeps 1 marker per trial
    to_remove = [];
    for j = 2:length(EEG.event)
        if (EEG.event(j).latency - EEG.event(j-1).latency) < 5 * EEG.srate
            to_remove = [to_remove, j];
        end
    end
    EEG.event(to_remove) = [];

    % Label O/C based on block (odd=overt, even=covert)
    for j = 1:length(EEG.event)
        if mod(EEG.event(j).block_number, 2) == 1
            EEG.event(j).type = ['O' EEG.event(j).type(3:end)];
        else
            EEG.event(j).type = ['C' EEG.event(j).type(3:end)];
        end
    end

    % Expand events: 5 utterances per trial, 2s apart
    expanded_events = struct('type', {}, 'latency', {}, 'urevent', {}, 'block_number', {});
    for j = 1:length(EEG.event)
        blk = EEG.event(j).block_number;
        if ismember(EEG.event(j).type, event_types) || ismember(EEG.event(j).type, ...
                {'O 1','O 2','O 3','O 4','O 5'})
            for k = 1:5
                new_event.type = [EEG.event(j).type '_u_' num2str(k) '_b_' num2str(blk)];
                new_event.latency = EEG.event(j).latency + (k-1) * 2 * EEG.srate;
                new_event.urevent = EEG.event(j).urevent;
                new_event.block_number = blk;
                expanded_events(end+1) = new_event;
            end
        end
    end
    EEG.event = expanded_events;
    EEG = eeg_checkset(EEG, 'eventconsistency');

    % Epoch: covert only (blocks 2,4,6,8,10), all 5 utterances
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

fprintf('\n===== ST 5u c1 complete =====\n');
