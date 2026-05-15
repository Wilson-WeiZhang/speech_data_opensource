"""
Fig 4: fMRI data quality — FD distribution + tSNR map
Computes framewise displacement (FD) from SPM realignment parameters and
temporal signal-to-noise ratio (tSNR) from preprocessed 4D NIfTI files.
Output: fig/fig4_fmri_qc.png/.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import os
import glob
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

# === Config ===
data_root = '../../data'                  # <-- SET YOUR DATA ROOT
fmri_dir  = os.path.join(data_root, 'prep_fmri')   # contains S00xx/ subdirs with swra*.nii
out_dir   = '../../fig'

# === Step 1: Compute FD from realignment parameters ===
# SPM writes rp_*.txt (6 columns: x y z pitch roll yaw) alongside preprocessed files.
# FD (Power et al. 2012): sum of absolute derivatives of 6 params (rotations in mm at 50mm radius).

def compute_fd(rp_file, radius=50.0):
    """Compute framewise displacement from SPM realignment parameters."""
    rp = np.loadtxt(rp_file)  # [n_vols x 6]: x y z pitch roll yaw
    rp_mm = rp.copy()
    rp_mm[:, 3:] *= radius  # convert rotations (rad) to mm at sphere surface
    diff = np.diff(rp_mm, axis=0)
    fd = np.sum(np.abs(diff), axis=1)
    return fd

subj_dirs = sorted(glob.glob(os.path.join(fmri_dir, 'S0*')))
subj_dirs = [d for d in subj_dirs if os.path.isdir(d)]

mean_fds = []
subj_ids = []
for sd in subj_dirs:
    sid = os.path.basename(sd)
    rp_files = glob.glob(os.path.join(sd, 'rp_*.txt'))
    if not rp_files:
        continue
    fd = compute_fd(rp_files[0])
    mean_fds.append(np.mean(fd))
    subj_ids.append(sid)

mean_fds = np.array(mean_fds)
n_subj = len(mean_fds)
print(f'FD computed for {n_subj} subjects')
sorted_fd = np.sort(mean_fds)

# === Step 2: Compute group-mean tSNR ===
# tSNR = mean(BOLD) / std(BOLD) across time, per voxel, then average across subjects.

tsnr_maps = []
for sd in subj_dirs:
    sid = os.path.basename(sd)
    nii_files = glob.glob(os.path.join(sd, 'swra*.nii'))
    if not nii_files:
        continue
    img = nib.load(nii_files[0])
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        continue  # skip if not 4D
    mu = data.mean(axis=3)
    sd_map = data.std(axis=3)
    sd_map[sd_map == 0] = np.inf
    tsnr_map = mu / sd_map
    tsnr_maps.append(tsnr_map)

group_tsnr = np.mean(tsnr_maps, axis=0) if tsnr_maps else np.zeros((1,1,1))
print(f'tSNR computed for {len(tsnr_maps)} subjects')

# === Figure: a (FD bar) on top, b (tSNR 3x5) on bottom ===
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[0.8, 1.2], hspace=0.25)

# --- Panel (a): FD sorted bar ---
ax_a = fig.add_subplot(gs[0])
colors = ['#4393C3' if v < 0.2 else ('#FDAE61' if v < 0.5 else '#D73027') for v in sorted_fd]
ax_a.bar(range(1, len(sorted_fd)+1), sorted_fd, color=colors, width=0.8)
ax_a.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.8)
ax_a.axhline(0.2, color='gray', linestyle=':', linewidth=1, alpha=0.6)
ax_a.text(len(sorted_fd)+1.5, 0.5, '0.5 mm', color='red', fontsize=9, va='center')
ax_a.text(len(sorted_fd)+1.5, 0.2, '0.2 mm', color='gray', fontsize=9, va='center')
ax_a.set_xlabel('Subjects (sorted)', fontsize=11)
ax_a.set_ylabel('Mean FD (mm)', fontsize=11)
ax_a.set_title(f'Head Motion During fMRI Acquisition (N={n_subj})', fontsize=12, fontweight='bold')
ax_a.set_ylim(0, max(0.55, sorted_fd.max() * 1.1))
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
mu, sd = np.mean(mean_fds), np.std(mean_fds)
n_above = int(np.sum(mean_fds > 0.5))
ax_a.text(0.02, 0.95, f'Mean FD = {mu:.2f} \u00b1 {sd:.2f} mm\n{n_above} subjects > 0.5 mm',
          transform=ax_a.transAxes, fontsize=9, va='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax_a.text(-0.06, 1.05, 'a', transform=ax_a.transAxes, fontsize=16, fontweight='bold', va='top')

# --- Panel (b): tSNR axial slices, 3 rows x 5 cols ---
nz = group_tsnr.shape[2]
n_slices = 15
slices = np.linspace(int(nz * 0.2), int(nz * 0.88), n_slices).astype(int)
nrows, ncols = 3, 5

gs_inner = gridspec.GridSpecFromSubplotSpec(nrows + 1, ncols + 1, subplot_spec=gs[1],
                                            width_ratios=[1]*ncols + [0.05],
                                            height_ratios=[0.4] + [1]*nrows,
                                            wspace=0.03, hspace=0.12)

vmin, vmax = 0, 150
for i, sl in enumerate(slices):
    r, c = divmod(i, ncols)
    ax = fig.add_subplot(gs_inner[r + 1, c])
    ax.imshow(np.rot90(group_tsnr[:, :, sl]), cmap='gray', vmin=vmin, vmax=vmax,
              interpolation='bilinear', aspect='equal')
    ax.set_title(f'z={sl}', fontsize=8, pad=2)
    ax.axis('off')

# Panel b label
ax_label = fig.add_subplot(gs[1])
ax_label.axis('off')
ax_label.text(-0.06, 1.05, 'b', transform=ax_label.transAxes, fontsize=16, fontweight='bold', va='top')
ax_label.set_title(f'Group Mean tSNR Map (N={len(tsnr_maps)})', fontsize=12, fontweight='bold', pad=10)

# Colorbar
cax = fig.add_subplot(gs_inner[1:, -1])
sm = plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cb = plt.colorbar(sm, cax=cax)
cb.set_label('tSNR', fontsize=10)

# Save
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f'{out_dir}/fig4_fmri_qc.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out_dir}/fig4_fmri_qc.pdf', bbox_inches='tight', facecolor='white')
print(f'Saved to {out_dir}/fig4_fmri_qc.png/pdf')
plt.close()
