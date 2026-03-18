%% gen_fig3_opensource.m — Figure 3: Phrase-Specific ERPs
%% Reproduces Figure 3 from the DSO Data Descriptor manuscript.
%%
%% Figure layout (2 rows x 2 columns):
%%   Col 1: Standalone EEG (ST)       Col 2: Simultaneous EEG (SI)
%%   Row 1: GFP — 5 phrases           Row 1: GFP — 5 phrases
%%   Row 2: Cz  — 5 phrases           Row 2: Cz  — 5 phrases
%%
%% Each panel shows grand-average waveforms (5 command phrases) with SEM
%% shading across subjects. Data are baseline z-scored and smoothed (20 ms
%% moving average) for visualization.
%%
%% Requirements:
%%   - MATLAB (tested R2025b)
%%   - EEGLAB (tested 2024.2)
%%   - Preprocessed data: *_clean56.set files in prep_st_5u/ and prep_si_5u/
%%
%% Input data:
%%   data_root/prep_st_5u/  — Standalone EEG, 56-ch, covert epochs (N=58)
%%   data_root/prep_si_5u/  — Simultaneous EEG, 56-ch, covert epochs (N=51)
%%
%% Output:
%%   out_dir/erp_5phrases_gfp_st_covert.png
%%   out_dir/erp_5phrases_gfp_si_covert.png
%%   out_dir/erp_5phrases_cz_st_covert.png
%%   out_dir/erp_5phrases_cz_si_covert.png
%%
%% Usage:
%%   1. Set paths below (data_root, eeglab_path, out_dir)
%%   2. Run: matlab -batch "gen_fig3_opensource"

%% ==================== USER CONFIG ====================
data_root   = '../../data';                    % parent of prep_st_5u/ and prep_si_5u/
eeglab_path = '/path/to/eeglab2024.2';     % EEGLAB root directory
out_dir     = '../../fig/f3';                  % output directory
dpi         = 300;                           % export resolution

%% ==================== SETUP ====================
addpath(eeglab_path); eeglab nogui;
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

st_dir = fullfile(data_root, 'prep_st_5u');
si_dir = fullfile(data_root, 'prep_si_5u');

% The 5 covert speech command phrases
condition_names = {'Go there', 'Distract target', 'Follow me', 'Explore here', 'Terminate'};
colors = [97, 108, 140; 86, 140, 135; 178, 213, 155; 242, 222, 121; 217, 95, 24] / 255;

modalities = {'st', 'si'};
mod_dirs   = {st_dir, si_dir};
mod_labels = {'Standalone EEG', 'Simultaneous EEG'};

% Channel modes to plot
ch_modes  = {'GFP', 'Cz'};
ch_tags   = {'gfp', 'cz'};
smooth_win = 5;  % 5 samples @ 250 Hz = 20 ms moving average

%% ==================== MAIN LOOP ====================
for m = 1:2
    files = dir(fullfile(mod_dirs{m}, '*_clean56.set'));
    n_subj = length(files);
    fprintf('\n=== %s: %d subjects ===\n', mod_labels{m}, n_subj);

    % Get dimensions from first file
    EEG0 = pop_loadset('filename', files(1).name, 'filepath', mod_dirs{m});
    times  = EEG0.times / 1000;           % ms -> seconds
    n_pnts = EEG0.pnts;
    bl_idx = times < 0;                    % baseline: pre-stimulus samples
    ch_labels = {EEG0.chanlocs.labels};
    cz_idx = find(strcmp(ch_labels, 'Cz'));

    % Storage: n_subj x 2 modes (GFP, Cz) x 5 phrases x n_pnts
    subject_data = nan(n_subj, 2, 5, n_pnts);

    for s = 1:n_subj
        fprintf('  %d/%d: %s\n', s, n_subj, files(s).name(1:5));
        EEG = pop_loadset('filename', files(s).name, 'filepath', mod_dirs{m});

        %% Parse phrase number from epoch labels
        % Label format: "C 4_u_1_b_2" (C=covert, 4=phrase, u=utterance, b=block)
        cond_trials = cell(5, 1);
        for e = 1:EEG.trials
            evt = EEG.epoch(e).eventtype;
            if iscell(evt), evt = evt{1}; end
            if ischar(evt) || isstring(evt)
                evt = strtrim(evt);
                if length(evt) >= 3
                    digits = regexp(evt(2:end), '\d', 'once');
                    if ~isempty(digits)
                        phrase = str2double(evt(1 + digits));
                        if phrase >= 1 && phrase <= 5
                            cond_trials{phrase}(end+1) = e;
                        end
                    end
                end
            end
        end

        %% Compute per-phrase waveforms, then baseline z-score
        for ph = 1:5
            if ~isempty(cond_trials{ph})
                % Average across trials for this phrase
                erp_ph = mean(EEG.data(:, :, cond_trials{ph}), 3);  % n_ch x n_pnts

                % Mode 1: GFP = std across channels at each time point
                gfp = std(erp_ph, 0, 1);
                bl_mu = mean(gfp(bl_idx));
                bl_sd = std(gfp(bl_idx));
                if bl_sd > 0
                    gfp = (gfp - bl_mu) / bl_sd;
                end
                subject_data(s, 1, ph, :) = gfp;

                % Mode 2: Cz electrode
                cz_erp = erp_ph(cz_idx, :);
                bl_mu = mean(cz_erp(bl_idx));
                bl_sd = std(cz_erp(bl_idx));
                if bl_sd > 0
                    cz_erp = (cz_erp - bl_mu) / bl_sd;
                end
                subject_data(s, 2, ph, :) = cz_erp;
            end
        end
    end

    %% Plot each mode
    for mode = 1:2
        % Grand mean + SEM across subjects
        gm = squeeze(nanmean(subject_data(:, mode, :, :), 1));   % 5 x n_pnts
        gs = squeeze(nanstd(subject_data(:, mode, :, :), 0, 1)) / sqrt(n_subj);

        % Smooth with moving average
        for cond = 1:5
            gm(cond, :) = movmean(gm(cond, :), smooth_win);
            gs(cond, :) = movmean(gs(cond, :), smooth_win);
        end

        fig = figure('Position', [100, 100, 1400, 700], 'Color', 'white');
        hold on;

        % SEM shading
        for cond = 1:5
            upper = gm(cond, :) + gs(cond, :);
            lower = gm(cond, :) - gs(cond, :);
            fill([times, fliplr(times)], [upper, fliplr(lower)], ...
                colors(cond, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
                'HandleVisibility', 'off');
        end

        % Mean lines
        for cond = 1:5
            plot(times, gm(cond, :), '-', 'Color', colors(cond, :), 'LineWidth', 3, ...
                'DisplayName', condition_names{cond});
        end

        % Reference lines
        yl = ylim;
        plot([0 0], yl, 'k--', 'LineWidth', 3, 'HandleVisibility', 'off');
        yline(0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, 'HandleVisibility', 'off');
        ylim(yl);

        % Formatting
        font_size = 20;
        xlabel('Event-locked time / s', 'FontSize', font_size);
        if mode == 1
            ylabel('GFP (z-score)', 'FontSize', font_size);
        else
            ylabel('Amplitude (z-score)', 'FontSize', font_size);
        end
        xlim([-0.5, 1.5]);
        title(sprintf('%s — %s, 5 Phrases (N=%d)', ...
            mod_labels{m}, ch_modes{mode}, n_subj), 'FontSize', font_size + 2);
        set(gca, 'TickDir', 'out', 'Box', 'off', 'FontSize', font_size, 'LineWidth', 3);
        ax = gca;
        ax.XAxis.LineWidth = 3;
        ax.YAxis.LineWidth = 3;
        legend('Location', 'northeast', 'FontSize', 14, 'Box', 'off');
        hold off;

        % Save
        fname = sprintf('erp_5phrases_%s_%s_covert', ch_tags{mode}, modalities{m});
        exportgraphics(fig, fullfile(out_dir, [fname '.png']), 'Resolution', dpi);
        close(fig);
        fprintf('  Saved %s.png\n', fname);
    end
end

%% ==================== DONE ====================
fprintf('\n===== Figure 3 complete =====\n');
fprintf('Output directory: %s\n', out_dir);
fprintf('Files:\n');
fprintf('  erp_5phrases_gfp_st_covert.png  — GFP, Standalone\n');
fprintf('  erp_5phrases_gfp_si_covert.png  — GFP, Simultaneous\n');
fprintf('  erp_5phrases_cz_st_covert.png   — Cz,  Standalone\n');
fprintf('  erp_5phrases_cz_si_covert.png   — Cz,  Simultaneous\n');
fprintf('\nManual step: arrange the 4 panels into a 2x2 grid in your figure editor.\n');
