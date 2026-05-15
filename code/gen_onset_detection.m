%% gen_onset_detection_opensource.m — Audio Onset/Offset Detection Pipeline
%% Reproduces onset_detection_results.mat from raw audio recordings.
%%
%% This script documents the complete processing pipeline from raw audio
%% (HDF5) to the shared onset_detection_results.mat file. The pipeline
%% has three stages:
%%
%%   Stage 1: Convert HDF5 audio to per-subject MAT files
%%   Stage 2: Hilbert envelope extraction, outlier trial rejection,
%%            and representative trial selection per phrase
%%   Stage 3: Threshold-based speech onset/offset detection
%%
%% Requirements:
%%   - MATLAB (tested R2025b)
%%   - Signal Processing Toolbox (for hilbert)
%%   - Raw audio HDF5 file: dso_audio_sr48000.H5
%%       Structure: /S00XX/audio  [672000 x 100]  (samples x trials)
%%                  /S00XX/label  [100 x 1]        (phrase label 0-4)
%%
%% Audio recording parameters:
%%   - Sampling rate: 48,000 Hz
%%   - Trial duration: 14 s (672,000 samples)
%%   - 100 trials per subject (20 trials x 5 phrases)
%%   - Each trial contains 5 utterance windows at 2, 4, 6, 8, 10 s
%%   - Labels 0-4 map to: Go there, Distract target, Follow me,
%%                         Explore here, Terminate
%%
%% Output:
%%   onset_detection_results.mat — struct array with fields:
%%     subject        — subject ID string (e.g. 'S0009')
%%     label          — phrase label (0-4)
%%     trial          — trial index within that phrase
%%     window         — utterance window (always 1 here; first utterance)
%%     onset_sample   — onset position in samples (within 2.0-3.5 s window)
%%     offset_sample  — offset position in samples (within 2.0-3.5 s window)
%%
%%   To convert to milliseconds: onset_ms = onset_sample / 48000 * 1000
%%
%% Usage:
%%   1. Set paths below
%%   2. Run: matlab -batch "gen_onset_detection_opensource"

%% ==================== USER CONFIG ====================
h5_file     = '';                          % path to dso_audio_sr48000.H5
work_dir    = '';                          % working directory for intermediate files
out_dir     = '';                          % output directory for final .mat
fs          = 48000;                       % audio sampling rate (Hz)
skip_subjects = {'S0013'};                 % subjects excluded from audio analysis

%% ==================== SETUP ====================
if isempty(h5_file) || ~exist(h5_file, 'file')
    error('Set h5_file to the path of dso_audio_sr48000.H5');
end
mat_dir = fullfile(work_dir, 'audio_mat');       % Stage 1 output
env_dir = fullfile(work_dir, 'audio_envelope');  % Stage 2 output
if ~exist(mat_dir, 'dir'), mkdir(mat_dir); end
if ~exist(env_dir, 'dir'), mkdir(env_dir); end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

phrase_names = {'Go there', 'Distract target', 'Follow me', ...
                'Explore here', 'Terminate'};

%% ==================== STAGE 1: HDF5 → MAT ====================
%% Each subject's audio is stored in the HDF5 file as:
%%   /S00XX/audio  [672000 x 100]  — raw waveform (14 s at 48 kHz)
%%   /S00XX/label  [100 x 1]       — phrase label per trial (0-4)
%%
%% Output: one MAT file per subject containing:
%%   audio  [100 x 672000]  — transposed for MATLAB convention (trials x samples)
%%   label  [100 x 1]
%%   fs     — sampling rate

fprintf('\n===== STAGE 1: HDF5 to MAT conversion =====\n');
info = h5info(h5_file);
n_groups = length(info.Groups);
fprintf('Found %d subjects in HDF5\n', n_groups);

for i = 1:n_groups
    group_name = info.Groups(i).Name;       % e.g. '/S0009'
    subject_id = group_name(2:end);         % remove leading '/'

    if ismember(subject_id, skip_subjects)
        fprintf('  %s: SKIPPED (excluded)\n', subject_id);
        continue;
    end

    audio_raw = h5read(h5_file, [group_name '/audio']);  % [672000 x 100]
    label     = h5read(h5_file, [group_name '/label']);  % [100 x 1]
    audio     = audio_raw';                              % [100 x 672000]

    save(fullfile(mat_dir, [subject_id '.mat']), 'audio', 'label', 'fs', '-v7.3');
    fprintf('  %s: audio %s, labels %s\n', subject_id, ...
        mat2str(size(audio)), mat2str(size(label)));
end

%% ==================== STAGE 2: Envelope + QC ====================
%% For each subject:
%%   1. Compute Hilbert envelope for every trial
%%   2. Reject outlier trials via pairwise correlation (3-sigma rule)
%%   3. For each phrase (label 0-4), keep all valid trials
%%
%% Outlier detection:
%%   - Compute mean pairwise Pearson correlation for each trial
%%   - Threshold = max(1st percentile, mean - 3*std, 0.1)
%%   - Trials below threshold are rejected
%%
%% Output per subject: envelope MAT file with struct 'results' containing:
%%   trials_by_label  — all valid trial envelopes per phrase

fprintf('\n===== STAGE 2: Hilbert envelope + outlier rejection =====\n');
mat_files = dir(fullfile(mat_dir, 'S*.mat'));

for file_idx = 1:length(mat_files)
    subject_file = mat_files(file_idx).name;
    subject_id = subject_file(1:end-4);
    if ismember(subject_id, skip_subjects), continue; end
    fprintf('  %d/%d: %s', file_idx, length(mat_files), subject_id);

    data = load(fullfile(mat_dir, subject_file));
    audio = data.audio;
    label = data.label;
    [n_trials, n_samples] = size(audio);

    % Hilbert envelope for all trials
    envelope_trials = zeros(n_trials, n_samples);
    for trial = 1:n_trials
        envelope_trials(trial, :) = abs(hilbert(audio(trial, :)));
    end

    % Outlier trial detection (correlation-based)
    valid_trials = detect_outlier_trials_correlation(envelope_trials);
    fprintf(' — %d/%d valid\n', length(valid_trials), n_trials);

    % Organize by phrase label
    trials_by_label = struct();
    for label_val = 0:4
        label_mask = (label == label_val);
        valid_label_trials = intersect(valid_trials, find(label_mask));
        trials_by_label.(sprintf('label_%d', label_val)) = ...
            envelope_trials(valid_label_trials, :);
    end

    results = struct();
    results.subject_id    = subject_id;
    results.fs            = fs;
    results.trials_by_label = trials_by_label;
    results.valid_trials  = valid_trials;
    results.n_total_trials = n_trials;

    save(fullfile(env_dir, [subject_id '_envelope.mat']), 'results', '-v7.3');
end

%% ==================== STAGE 3: Onset/Offset Detection ====================
%% For each valid trial, extract the first utterance window (2.0–3.5 s)
%% and detect speech onset and offset using a threshold-based algorithm.
%%
%% Algorithm (detect_speech_onset_offset):
%%   1. Estimate baseline from lowest 10% of envelope samples
%%   2. Set threshold = baseline + 5% × (peak - baseline)
%%   3. Onset: scan forward; require envelope > threshold sustained
%%      across 6 context windows (100–600 ms) with increasing amplitude
%%   4. Offset: scan backward with same context-window criterion
%%
%% The context-window requirement ensures that transient spikes are not
%% mistaken for speech onset — the signal must stay above threshold for
%% at least 600 ms with an upward trend.

fprintf('\n===== STAGE 3: Onset/offset detection =====\n');

% First utterance window: 2.0–3.5 s
time_window = [2.0, 3.5];
win_samples = round(time_window * fs);

env_files = dir(fullfile(env_dir, 'S*_envelope.mat'));
onset_results = [];

for file_idx = 1:length(env_files)
    subject_file = env_files(file_idx).name;
    subject_id = subject_file(1:end-13);
    if ismember(subject_id, skip_subjects), continue; end
    fprintf('  %d/%d: %s\n', file_idx, length(env_files), subject_id);

    data = load(fullfile(env_dir, subject_file));
    results = data.results;

    for label_val = 0:4
        label_field = sprintf('label_%d', label_val);
        if ~isfield(results.trials_by_label, label_field), continue; end

        trials_data = results.trials_by_label.(label_field);
        if isempty(trials_data), continue; end
        [n_trials, n_samples] = size(trials_data);

        for trial_idx = 1:n_trials
            envelope_trial = trials_data(trial_idx, :);

            start_s = win_samples(1);
            end_s   = min(win_samples(2), n_samples);
            if start_s > n_samples || end_s <= start_s, continue; end

            window_envelope = envelope_trial(start_s:end_s);
            [onset_sample, offset_sample] = ...
                detect_speech_onset_offset(window_envelope, fs);

            row.subject       = subject_id;
            row.label         = label_val;
            row.trial         = trial_idx;
            row.window        = 1;
            row.onset_sample  = onset_sample;
            row.offset_sample = offset_sample;
            onset_results = [onset_results; row];
        end
    end
end

%% Save results
save(fullfile(out_dir, 'onset_detection_results.mat'), 'onset_results');
fprintf('\nSaved onset_detection_results.mat (%d entries)\n', length(onset_results));

%% Summary statistics
all_onsets  = [onset_results.onset_sample];
all_offsets = [onset_results.offset_sample];
v_on  = ~isnan(all_onsets);
v_off = ~isnan(all_offsets);

onset_ms  = all_onsets(v_on)  / fs * 1000;
offset_ms = all_offsets(v_off) / fs * 1000;

fprintf('\nGrand mean onset:  %.1f +/- %.1f ms  (N = %d)\n', ...
    mean(onset_ms), std(onset_ms), sum(v_on));
fprintf('Grand mean offset: %.1f +/- %.1f ms  (N = %d)\n', ...
    mean(offset_ms), std(offset_ms), sum(v_off));
fprintf('\n===== Pipeline complete =====\n');

%% ==================== HELPER FUNCTIONS ====================

function valid_trials = detect_outlier_trials_correlation(envelope_trials)
%DETECT_OUTLIER_TRIALS_CORRELATION  Reject outlier trials via pairwise correlation.
%
%   For each trial, compute its mean Pearson correlation with all other
%   trials. Trials with correlation below threshold are rejected.
%
%   Threshold = max(1st percentile of correlations,
%                   mean - 3*std of correlations,
%                   0.1)

    [n_trials, ~] = size(envelope_trials);
    correlations = zeros(n_trials, 1);

    for i = 1:n_trials
        trial_i = envelope_trials(i, :);
        r_vals = zeros(n_trials - 1, 1);
        k = 0;
        for j = 1:n_trials
            if i == j, continue; end
            if any(isnan(trial_i)) || any(isnan(envelope_trials(j, :)))
                continue;
            end
            r = corrcoef(trial_i, envelope_trials(j, :));
            k = k + 1;
            r_vals(k) = r(1, 2);
        end
        if k > 0
            correlations(i) = mean(r_vals(1:k));
        else
            correlations(i) = NaN;
        end
    end

    valid_mask = ~isnan(correlations);
    if sum(valid_mask) > 5
        threshold = max(prctile(correlations, 1), ...
                        mean(correlations) - 3 * std(correlations));
        threshold = max(threshold, 0.1);
        valid_mask = valid_mask & (correlations >= threshold);
    end

    valid_trials = find(valid_mask);
end

function [onset_sample, offset_sample] = detect_speech_onset_offset(envelope_data, fs)
%DETECT_SPEECH_ONSET_OFFSET  Threshold-based speech boundary detection.
%
%   1. Baseline = mean of lowest 10% of envelope samples
%   2. Threshold = baseline + 5% * (peak - baseline)
%   3. Onset: first sample where envelope exceeds threshold AND the signal
%      remains above threshold (with increasing trend) across 6 context
%      windows of 100, 200, 300, 400, 500, 600 ms
%   4. Offset: same criterion scanning backward from end
%
%   Returns sample indices relative to the input segment (NaN if not found).

    % Baseline from lowest 10% of samples
    n_baseline = round(0.1 * fs);  % ~4800 samples
    sorted_data = sort(envelope_data);
    baseline_mean = mean(sorted_data(1:min(n_baseline, length(sorted_data))));

    % Threshold: 5% above baseline toward peak
    peak_value = max(envelope_data);
    threshold = baseline_mean + (peak_value - baseline_mean) * 0.05;

    % Context windows: 100 ms to 600 ms in 100 ms steps
    context_windows = round((0.1:0.1:0.6) * fs);

    % Forward scan for onset
    onset_sample = NaN;
    for i = 1:length(envelope_data)
        if envelope_data(i) > threshold
            all_ok = true;
            for cw = context_windows
                end_idx = min(i + cw - 1, length(envelope_data));
                seg = envelope_data(i:end_idx);
                if ~(mean(seg) > threshold && max(seg) > envelope_data(i))
                    all_ok = false;
                    break;
                end
            end
            if all_ok
                onset_sample = i;
                break;
            end
        end
    end

    % Backward scan for offset
    offset_sample = NaN;
    for i = length(envelope_data):-1:1
        if envelope_data(i) > threshold
            all_ok = true;
            for cw = context_windows
                start_idx = max(i - cw + 1, 1);
                seg = envelope_data(start_idx:i);
                if ~(mean(seg) > threshold && max(seg) > envelope_data(i))
                    all_ok = false;
                    break;
                end
            end
            if all_ok
                offset_sample = i;
                break;
            end
        end
    end
end
