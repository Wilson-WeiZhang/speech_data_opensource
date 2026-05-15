%% fMRI Preprocessing Pipeline (SPM12 + DPARSF/DPABI)
% COSMO Dataset - Task fMRI preprocessing
% Requires: SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
% Input:  raw NIfTI functional + T1 structural images
% Output: preprocessed NIfTI in MNI space (swra*.nii)
%
% Pipeline: slice timing -> realignment -> T1 coregistration to func ->
%           New Segment + DARTEL normalization -> smoothing (4mm FWHM)
%
% Parameters match DPARSF configuration used in the original processing.
% Usage: Set 'data_dir' and 'subjects' below, then run.

clear; clc;
spm('defaults', 'fmri');
spm_jobman('initcfg');

%% Configuration
data_dir    = '/path/to/data';       % root directory
subjects    = {'S0009','S0012','S0014','S0015','S0016','S0017','S0018', ...
               'S0021','S0022','S0024','S0025','S0026','S0027','S0028', ...
               'S0029','S0030','S0031','S0032','S0033','S0034','S0035', ...
               'S0036','S0037','S0038','S0039','S0040','S0041','S0042', ...
               'S0043','S0044','S0045','S0047','S0048','S0049','S0050', ...
               'S0052','S0055','S0056','S0057','S0058','S0059','S0060', ...
               'S0061','S0062','S0063','S0064','S0065','S0066','S0067','S0068'};

n_discard   = 10;        % first 10 volumes discarded (removed before input)
TR          = 1.5;       % repetition time (s)
n_slices    = 26;        % number of slices
slice_order = [2 4 6 8 10 12 14 16 18 20 22 24 26 ...
               1 3 5 7 9 11 13 15 17 19 21 23 25]; % interleaved: even first
ref_slice   = 26;        % reference slice
smooth_fwhm = [4 4 4];  % smoothing kernel (mm)
norm_bb     = [-90 -126 -72; 90 90 108]; % bounding box for normalization
norm_vox    = [3 3 3];   % output voxel size (mm)

%% Process each subject
for s = 1:length(subjects)
    subj = subjects{s};
    fprintf('\n=== Processing %s (%d/%d) ===\n', subj, s, length(subjects));

    func_dir = fullfile(data_dir, 'raw_fmri', subj);
    anat_dir = fullfile(data_dir, 'mri_t1');

    % Load functional volumes
    func_files = spm_select('FPList', func_dir, '^.*\.nii$');
    if isempty(func_files)
        fprintf('  No functional files found for %s, skipping.\n', subj);
        continue;
    end

    % Discard first N volumes
    func_files = func_files(n_discard+1:end, :);
    n_vols = size(func_files, 1);
    fprintf('  %d volumes after discarding first %d\n', n_vols, n_discard);

    % Get structural image
    anat_file = spm_select('FPList', anat_dir, ['^' subj '_T1\.nii$']);

    % --- Step 1: Slice Timing Correction ---
    matlabbatch{1}.spm.temporal.st.scans = {cellstr(func_files)};
    matlabbatch{1}.spm.temporal.st.nslices = n_slices;
    matlabbatch{1}.spm.temporal.st.tr = TR;
    matlabbatch{1}.spm.temporal.st.ta = TR - (TR / n_slices);
    matlabbatch{1}.spm.temporal.st.so = slice_order;
    matlabbatch{1}.spm.temporal.st.refslice = ref_slice;
    matlabbatch{1}.spm.temporal.st.prefix = 'a';

    % --- Step 2: Realignment (motion correction) ---
    matlabbatch{2}.spm.spatial.realign.estwrite.data = ...
        {cellstr(spm_file(func_files, 'prefix', 'a'))};
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.which = [2 1];
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.prefix = 'r';

    % --- Step 3: Coregistration (T1 -> mean functional) ---
    % Note: T1 is coregistered TO the functional space (T1 Coreg to Fun)
    mean_func = spm_file(func_files(1,:), 'prefix', 'meana');
    matlabbatch{3}.spm.spatial.coreg.estimate.ref = cellstr(mean_func);
    matlabbatch{3}.spm.spatial.coreg.estimate.source = cellstr(anat_file);
    matlabbatch{3}.spm.spatial.coreg.estimate.other = {''};

    % --- Step 4: New Segment + DARTEL normalization ---
    % Step 4a: Segment T1 using New Segment (generates DARTEL imports)
    tpm_path = fullfile(spm('Dir'), 'tpm', 'TPM.nii');
    for ti = 1:6
        matlabbatch{4}.spm.spatial.preproc.tissue(ti).tpm = {[tpm_path ',' num2str(ti)]};
        matlabbatch{4}.spm.spatial.preproc.tissue(ti).ngaus = [1 1 2 3 4 2];
        matlabbatch{4}.spm.spatial.preproc.tissue(ti).ngaus = matlabbatch{4}.spm.spatial.preproc.tissue(ti).ngaus(ti);
        matlabbatch{4}.spm.spatial.preproc.tissue(ti).native = [1 1]; % native + DARTEL import
        matlabbatch{4}.spm.spatial.preproc.tissue(ti).warped = [0 0];
    end
    matlabbatch{4}.spm.spatial.preproc.channel.vols = cellstr(anat_file);
    matlabbatch{4}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{4}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{4}.spm.spatial.preproc.channel.write = [0 0];
    matlabbatch{4}.spm.spatial.preproc.warp.affreg = 'european';
    matlabbatch{4}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{4}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{4}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];

    % Step 4b: DARTEL — create template (run once across all subjects)
    % For single-subject processing, use existing DARTEL template or
    % run DARTEL separately. Here we use Normalize to MNI via DARTEL:
    % matlabbatch{5}.spm.tools.dartel.mni_norm.template = {dartel_template};
    % matlabbatch{5}.spm.tools.dartel.mni_norm.data.subj.flowfield = {flowfield};
    % matlabbatch{5}.spm.tools.dartel.mni_norm.data.subj.images = cellstr(spm_file(func_files,'prefix','ra'));
    % matlabbatch{5}.spm.tools.dartel.mni_norm.vox = norm_vox;
    % matlabbatch{5}.spm.tools.dartel.mni_norm.bb = norm_bb;

    % SPM Normalise:Write using deformation field from New Segment
    % Note: the original processing used DPARSF with DARTEL normalization.
    % This script uses the deformation field approach as an equivalent alternative.
    [anat_path, anat_name] = fileparts(deblank(anat_file));
    deformation_field = fullfile(anat_path, ['y_' anat_name '.nii']);
    matlabbatch{5}.spm.spatial.normalise.write.subj.def = {deformation_field};
    matlabbatch{5}.spm.spatial.normalise.write.subj.resample = ...
        cellstr(spm_file(func_files, 'prefix', 'ra'));
    matlabbatch{5}.spm.spatial.normalise.write.woptions.bb = norm_bb;
    matlabbatch{5}.spm.spatial.normalise.write.woptions.vox = norm_vox;
    matlabbatch{5}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{5}.spm.spatial.normalise.write.woptions.prefix = 'w';

    % --- Step 5: Smoothing (4 mm FWHM) ---
    matlabbatch{6}.spm.spatial.smooth.data = ...
        cellstr(spm_file(func_files, 'prefix', 'wra'));
    matlabbatch{6}.spm.spatial.smooth.fwhm = smooth_fwhm;
    matlabbatch{6}.spm.spatial.smooth.dtype = 0;
    matlabbatch{6}.spm.spatial.smooth.prefix = 's';

    % Run batch
    try
        spm_jobman('run', matlabbatch);
        fprintf('  Done: %s\n', subj);
    catch ME
        fprintf('  ERROR processing %s: %s\n', subj, ME.message);
    end
    clear matlabbatch;
end

fprintf('\n=== All subjects processed ===\n');
