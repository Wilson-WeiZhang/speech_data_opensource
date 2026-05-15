%% gen_fig_audio_opensource.m — Audio Quality Figures
%% Reproduces audio QC panels from the DSO Data Descriptor manuscript.
%%
%% Panel A: Speech onset/offset distribution (histogram)
%%   Input: audio_timing.csv (shared with dataset, 53 subjects)
%%
%% Panel B: Example waveform + Hilbert envelope (5 phrases, sample subject)
%%   Input: raw audio .mat file for a sample subject (not shared; requires
%%          original HDF5 or per-subject MAT from Stage 1 of the pipeline)
%%
%% audio_timing.csv columns:
%%   subject_id, label (0-4), phrase, trial, utterance (1-5),
%%   onset_ms, offset_ms, duration_ms
%%   Times are relative to the utterance window start.
%%
%% Usage:
%%   1. Set paths below
%%   2. Run: matlab -batch "gen_fig_audio_opensource"

%% ==================== USER CONFIG ====================
csv_file    = '../../data_sharing/audio_timing.csv';  % shared onset/offset CSV
audio_file  = '';                                   % sample subject raw audio .mat (optional; not shared)
out_dir     = '../../fig/f5';
dpi         = 300;
fs_audio    = 48000;  % audio sampling rate (Hz)

%% ==================== SETUP ====================
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

phrase_names = {'Go There', 'Distract Target', 'Follow Me', 'Explore Here', 'Terminate'};
colors = [76, 114, 176;    % Go There       — blue
          85, 168, 104;    % Distract Target — green
          139, 195, 74;    % Follow Me       — yellow-green
          255, 193, 7;     % Explore Here    — gold
          221, 132, 82] / 255;  % Terminate  — orange

%% ==================== PANEL A: Onset/Offset Histogram ====================
fprintf('=== Panel A: Loading audio_timing.csv ===\n');
T = readtable(csv_file);
fprintf('  %d rows, %d subjects\n', height(T), length(unique(T.subject_id)));

onsets_ms  = T.onset_ms;
offsets_ms = T.offset_ms;

% Remove NaN
valid = ~isnan(onsets_ms) & ~isnan(offsets_ms);
onsets_ms  = onsets_ms(valid);
offsets_ms = offsets_ms(valid);

fprintf('  Onset:  %.0f +/- %.0f ms\n', mean(onsets_ms), std(onsets_ms));
fprintf('  Offset: %.0f +/- %.0f ms\n', mean(offsets_ms), std(offsets_ms));

% Plot — aspect ratio: y is half the height relative to x
fig = figure('visible', 'off', 'Position', [100 100 900 250], 'Color', 'w');

subplot(1, 2, 1);
histogram(onsets_ms, 30, 'FaceColor', [0.3 0.5 0.7], 'EdgeColor', 'w', 'FaceAlpha', 0.8);
xline(mean(onsets_ms), '--r', 'LineWidth', 1.5);
xlabel('Speech Onset (ms)', 'FontSize', 14);
ylabel('Count', 'FontSize', 14);
set(gca, 'LineWidth', 1.5, 'TickDir', 'out', 'FontSize', 11);
pbaspect([2 1 1]);  % x:y = 2:1
box on;

subplot(1, 2, 2);
histogram(offsets_ms, 30, 'FaceColor', [0.7 0.3 0.3], 'EdgeColor', 'w', 'FaceAlpha', 0.8);
xline(mean(offsets_ms), '--r', 'LineWidth', 1.5);
xlabel('Speech Offset (ms)', 'FontSize', 14);
ylabel('Count', 'FontSize', 14);
set(gca, 'LineWidth', 1.5, 'TickDir', 'out', 'FontSize', 11);
pbaspect([2 1 1]);  % x:y = 2:1
box on;

exportgraphics(fig, fullfile(out_dir, 'audio_onset_offset_hist.png'), 'Resolution', dpi);
exportgraphics(fig, fullfile(out_dir, 'audio_onset_offset_hist.pdf'), 'ContentType', 'vector');
close(fig);
fprintf('  Saved audio_onset_offset_hist.png + .pdf\n');

%% ==================== PANEL B: Waveform + Envelope ====================
%% Requires raw audio .mat file (not shared due to privacy).
%% The .mat file contains:
%%   audio  [100 x 672000]  — raw waveform (trials x samples, 48 kHz)
%%   label  [100 x 1]       — phrase label per trial (0-4)
%%   fs     — sampling rate
%%
%% Each subplot shows one phrase: normalized raw waveform (gray) overlaid
%% with symmetric Hilbert envelope (colored), for the first trial of that
%% phrase from the sample subject.

if ~isempty(audio_file) && exist(audio_file, 'file')
    fprintf('\n=== Panel B: Waveform + Hilbert envelope ===\n');
    A = load(audio_file);

    fig = figure('visible', 'off', 'Position', [100 100 900 700], 'Color', 'w');

    % First utterance window: 2.0–3.5 s
    win_start = round(2.0 * fs_audio) + 1;
    win_end   = round(3.5 * fs_audio);

    for p = 1:5
        label_idx = find(A.label == (p-1), 1);
        if isempty(label_idx), continue; end

        seg = double(A.audio(label_idx, win_start:win_end));
        seg = seg / max(abs(seg));  % normalize to [-1, 1]
        t_ms = (0:length(seg)-1) / fs_audio * 1000;

        % Hilbert envelope + 10 ms smoothing
        env = abs(hilbert(seg));
        env = movmean(env, round(0.01 * fs_audio));

        subplot(5, 1, p);
        plot(t_ms, seg, 'Color', [0.5 0.5 0.5 0.7], 'LineWidth', 0.3);
        hold on;
        plot(t_ms, env, 'Color', colors(p, :), 'LineWidth', 2);
        plot(t_ms, -env, 'Color', colors(p, :), 'LineWidth', 2);  % symmetric
        hold off;

        title(phrase_names{p}, 'FontSize', 11, 'FontWeight', 'bold', ...
              'HorizontalAlignment', 'left', 'Units', 'normalized', 'Position', [0 1.05 0]);
        ylim([-1.2 1.2]);
        set(gca, 'YTick', [-1 0 1], 'LineWidth', 1, 'TickDir', 'out');
        box off;

        if p == 3
            ylabel('Normalized amplitude', 'FontSize', 12);
        end
        if p < 5
            set(gca, 'XTickLabel', []);
        end
    end
    xlabel('Time (ms)', 'FontSize', 12);
    xlim([0 1500]);

    exportgraphics(fig, fullfile(out_dir, 'audio_waveform_envelope.png'), 'Resolution', dpi);
    exportgraphics(fig, fullfile(out_dir, 'audio_waveform_envelope.pdf'), 'ContentType', 'vector');
    close(fig);
    fprintf('  Saved audio_waveform_envelope.png + .pdf\n');
else
    fprintf('\nSkipping Panel B: no audio_file specified or file not found.\n');
    fprintf('  Set audio_file to a per-subject .mat (e.g. S0009.mat) to generate.\n');
end

%% ==================== DONE ====================
fprintf('\n===== Audio figures complete =====\n');
