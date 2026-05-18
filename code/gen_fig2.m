%% gen_fig2.m - Figure 2: EEG Signal Quality
%% Reproduces Figure 2 from the DSO Data Descriptor manuscript.
%%
%% Figure layout (2 rows x 2 columns):
%%   Col 1: Simultaneous EEG (SI)    Col 2: Standalone EEG (ST)
%%   Row 1: Grand-mean PSD - 3 topomaps (Delta/Theta/Alpha) + channel PSD
%%   Row 2: Sample subject PSD - same layout as Row 1
%%
%% Requirements:
%%   - MATLAB (tested R2025b)
%%   - EEGLAB (tested 2024.2)
%%   - Preprocessed data: *_clean.set files in prep_st_eeg/ and prep_si_eeg/
%%
%% Input data:
%%   data_root/prep_st_eeg/  - Standalone EEG, 63-ch, 5-utterance epochs (N=58)
%%   data_root/prep_si_eeg/  - Simultaneous EEG, 63-ch, 5-utterance epochs (N=51)
%%
%% Output:
%%   out_dir/grandmean_st_psd.png  - Row 1 right panel
%%   out_dir/grandmean_si_psd.png  - Row 1 left panel
%%   out_dir/<sample>_st_psd.png   - Row 2 right panel
%%   out_dir/<sample>_si_psd.png   - Row 2 left panel
%%
%% Usage:
%%   1. Set paths below (data_root, eeglab_path, out_dir)
%%   2. Run: matlab -batch "gen_fig2"

%% ==================== USER CONFIG ====================
% -- Set these paths to match your environment --
data_root   = '../../data';                    % parent of prep_st_eeg/ and prep_si_eeg/
eeglab_path = '/path/to/eeglab2024.2';     % EEGLAB root directory
out_dir     = '../../fig/f2';                  % output directory
sample_subj = 'S0016';                      % sample subject ID for rows 2-3
dpi         = 300;                           % export resolution

%% ==================== SETUP ====================
addpath(eeglab_path); eeglab nogui;
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

st_dir = fullfile(data_root, 'prep_st_eeg');
si_dir = fullfile(data_root, 'prep_si_eeg');

freq_bands = [1 4; 4 8; 8 12];             % Delta, Theta, Alpha (Hz)
band_labels = {
    'Delta (1-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-12 Hz)'
};

%% ==================== STEP 1: Grand-Mean PSD & Topomaps ====================
% Compute per-subject PSD using EEGLAB spectopo, then average across subjects.

modalities = {'si', 'st'};
mod_dirs   = {si_dir, st_dir};
mod_labels = {'Simultaneous EEG', 'Standalone EEG'};
expected_n = [51, 58];

gm = struct();  % store grand-mean results

for m = 1:2
    tag = modalities{m};
    files = dir(fullfile(mod_dirs{m}, '*_clean.set'));
    n_subj = length(files);
    fprintf('=== %s: %d subjects ===\n', mod_labels{m}, n_subj);
    assert(n_subj == expected_n(m), ...
        '%s expected %d clean files, found %d', mod_labels{m}, expected_n(m), n_subj);

    all_psd_ch = [];   % n_subj x n_ch x n_freqs
    all_topo   = [];   % n_subj x n_ch x 3 bands

    for i = 1:n_subj
        fprintf('  %d/%d: %s\n', i, n_subj, files(i).name(1:5));
        EEG = pop_loadset('filename', files(i).name, 'filepath', mod_dirs{m});
        assert(EEG.nbchan == 63, '%s has %d channels, expected 63', files(i).name, EEG.nbchan);

        % Per-channel PSD via spectopo (returns dB: 10*log10(uV^2/Hz))
        [spectra, freqs] = spectopo(EEG.data, 0, EEG.srate, ...
            'freqrange', [1 30], 'plot', 'off');

        if i == 1
            all_psd_ch = zeros(n_subj, size(spectra, 1), length(freqs));
            all_topo   = zeros(n_subj, size(spectra, 1), 3);
            gm.(tag).freqs = freqs;
            gm.(tag).chanlocs = EEG.chanlocs;
        end

        all_psd_ch(i, :, :) = spectra;

        % Band-averaged power per channel (for topoplots)
        for b = 1:3
            fidx = freqs >= freq_bands(b,1) & freqs <= freq_bands(b,2);
            all_topo(i, :, b) = mean(spectra(:, fidx), 2);
        end
    end

    gm.(tag).psd_ch = squeeze(mean(all_psd_ch, 1));   % n_ch x n_freqs
    gm.(tag).topo   = squeeze(mean(all_topo, 1));      % n_ch x 3
    gm.(tag).n      = n_subj;
    gm.(tag).all_topo = all_topo;
    gm.(tag).all_psd_ch = all_psd_ch;
end

%% ==================== STEP 2: Sample Subject PSD & Topomaps ====================
% Load sample subject from both modalities.

sample = struct();
for m = 1:2
    tag = modalities{m};
    fname = sprintf('%s_clean.set', sample_subj);
    EEG = pop_loadset('filename', fname, 'filepath', mod_dirs{m});
    assert(EEG.nbchan == 63, '%s has %d channels, expected 63', fname, EEG.nbchan);

    [spectra, freqs] = spectopo(EEG.data, 0, EEG.srate, ...
        'freqrange', [1 30], 'plot', 'off');

    topo_dat = zeros(size(spectra, 1), 3);
    for b = 1:3
        fidx = freqs >= freq_bands(b,1) & freqs <= freq_bands(b,2);
        topo_dat(:, b) = mean(spectra(:, fidx), 2);
    end

    sample.(tag).spectra  = spectra;
    sample.(tag).freqs    = freqs;
    sample.(tag).topo     = topo_dat;
    sample.(tag).chanlocs = EEG.chanlocs;
end

%% ==================== STEP 3: Determine Global Colorbar Limits ====================
% Shared color limits across all 4 PSD topoplot figures for visual consistency.

all_vals = [gm.st.topo(:); gm.si.topo(:); sample.st.topo(:); sample.si.topo(:)];
clim_global = [min(all_vals), max(all_vals)];

%% ==================== STEP 4: Plot 4 PSD Panels (Rows 1-2) ====================
% Each panel: 3 topomaps on top + per-channel PSD curves below.

plots = {
    gm.si.psd_ch,       gm.si.freqs, gm.si.topo,     gm.si.chanlocs, ...
        sprintf('%s Grand Mean (N=%d)', mod_labels{1}, gm.si.n), 'grandmean_si_psd'
    gm.st.psd_ch,       gm.st.freqs, gm.st.topo,     gm.st.chanlocs, ...
        sprintf('%s Grand Mean (N=%d)', mod_labels{2}, gm.st.n), 'grandmean_st_psd'
    sample.si.spectra,   sample.si.freqs, sample.si.topo, sample.si.chanlocs, ...
        sprintf('%s %s', sample_subj, mod_labels{1}), sprintf('%s_si_psd', sample_subj)
    sample.st.spectra,   sample.st.freqs, sample.st.topo, sample.st.chanlocs, ...
        sprintf('%s %s', sample_subj, mod_labels{2}), sprintf('%s_st_psd', sample_subj)
};

for p = 1:4
    psd_ch   = plots{p, 1};
    freqs    = plots{p, 2};
    topo_dat = plots{p, 3};
    chlocs   = plots{p, 4};
    ttl      = plots{p, 5};
    fname    = plots{p, 6};

    fig = figure('Position', [50 50 1000 600], 'Color', 'w', 'Visible', 'off');

    % --- Bottom: PSD curves (one line per channel) ---
    axes('Position', [0.08 0.08 0.88 0.48]);
    hold on;
    ch_colors = lines(size(psd_ch, 1));
    for ch = 1:size(psd_ch, 1)
        plot(freqs, psd_ch(ch, :), 'Color', ch_colors(ch, :), 'LineWidth', 0.8);
    end
    xlabel('Frequency (Hz)', 'FontSize', 12);
    ylabel('Log Power Spectral Density 10*log_{10}(\muV^2/Hz)', 'FontSize', 10);
    xlim([1 30]);
    % Mark band center frequencies
    for b = 1:3
        xline(mean(freq_bands(b,:)), '-k', 'LineWidth', 1);
    end
    set(gca, 'FontSize', 10, 'LineWidth', 1.2, 'Box', 'on');

    % --- Top: 3 topoplots (Delta, Theta, Alpha) ---
    topo_pos = [0.02 0.62 0.28 0.35;
                0.32 0.62 0.28 0.35;
                0.62 0.62 0.28 0.35];
    for b = 1:3
        axes('Position', topo_pos(b, :));
        topoplot(topo_dat(:, b), chlocs, 'electrodes', 'on', 'style', 'both', ...
            'maplimits', clim_global);
        title(band_labels{b}, 'FontSize', 11);
    end

    % Shared colorbar
    cb = colorbar('Position', [0.91 0.67 0.015 0.25], 'FontSize', 9);
    cb.Label.String = 'dB';
    caxis(clim_global);

    exportgraphics(fig, fullfile(out_dir, [fname '.png']), 'Resolution', dpi);
    close(fig);
    fprintf('  Saved %s.png\n', fname);
end

%% ==================== DONE ====================
fprintf('\n===== Figure 2 complete =====\n');
fprintf('Output directory: %s\n', out_dir);
fprintf('Files:\n');
fprintf('  grandmean_si_psd.png  - Row 1 left  (Simultaneous, grand mean)\n');
fprintf('  grandmean_st_psd.png  - Row 1 right (Standalone, grand mean)\n');
fprintf('  %s_si_psd.png   - Row 2 left  (Simultaneous, sample)\n', sample_subj);
fprintf('  %s_st_psd.png   - Row 2 right (Standalone, sample)\n', sample_subj);
fprintf('\nManual step: arrange the 4 panels into a 2x2 grid in your figure editor.\n');
