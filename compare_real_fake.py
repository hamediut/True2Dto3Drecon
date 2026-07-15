"""
Compare generated 3D volumes against the real training volume.

It reports two things, using the SAME statistics as training
(src/SMD_cal.calculate_two_point_3D):
  1) Porosity  -> volume fraction of phase == 1 (mean of the binary volume)
  2) S2(r)     -> directional two-point correlation (x, y, z) and their average,
                  plus the mean-squared-error between real and fake average S2
                  (this MSE is the exact criterion train.py uses to pick the best model).

Because the real volume (e.g. battery_separator.tif, 500x650x1300) is much bigger
than a generated 128^3 volume, we sample random 128^3 crops from the real volume so
the two are compared at the same size / same S2 length.

Example
-------
# compare all volumes produced by an inference run against the real one
python compare_real_fake.py --real battery_separator.tif \
    --fake_dir ./inference_G_XXXX_Zsize4_ball_0 \
    --output_dir ./comparison

# or compare a single generated volume
python compare_real_fake.py --real battery_separator.tif \
    --fake ./inference_G_XXXX/fake3D_00_por_0.312.tif
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')  # no display needed, just save files
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error

from src.SMD_cal import calculate_two_point_3D


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--real', required=True, type=str,
                        help='Path to the real 3D volume (.tif stack), e.g. battery_separator.tif')
    parser.add_argument('--fake', type=str,
                        help='Path to a single generated 3D volume (.tif). Use this OR --fake_dir.')
    parser.add_argument('--fake_dir', type=str,
                        help='Folder containing generated 3D volumes (*.tif). Use this OR --fake.')
    parser.add_argument('--max_fakes', type=int, default=None,
                        help='If set, use at most this many generated volumes from --fake_dir.')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Size of the random cubic crops taken from the real volume. '
                             'Default: auto (matches the generated volume size).')
    parser.add_argument('--n_real_crops', type=int, default=None,
                        help='Number of random crops from the real volume to average S2 over. '
                             'Default: same as the number of generated volumes (min 4).')
    parser.add_argument('--output_dir', type=str, default='./comparison',
                        help='Where to save the comparison plot and CSVs.')
    parser.add_argument('--seed', type=int, default=33, help='RNG seed for reproducible real crops.')
    return parser.parse_args()


def load_binary(path):
    """Read a .tif and binarize it (phase == 1 where value >= 1)."""
    im = tifffile.imread(path)
    return np.where(im >= 1, 1, 0).astype(np.uint8)


def sample_real_crops(vol, size, n, rng):
    """Sample n random cubic crops of edge `size` from a 3D volume."""
    x_max, y_max, z_max = vol.shape
    if min(vol.shape) < size:
        raise SystemExit(f'Real volume {vol.shape} is smaller than crop_size {size} on some axis.')
    crops = np.zeros((n, size, size, size), dtype=np.uint8)
    for i in range(n):
        x = int(rng.integers(0, x_max - size + 1))
        y = int(rng.integers(0, y_max - size + 1))
        z = int(rng.integers(0, z_max - size + 1))
        crops[i] = vol[x:x + size, y:y + size, z:z + size]
    return crops


def main():
    args = parse_args()
    if not args.fake and not args.fake_dir:
        raise SystemExit('Provide either --fake <file> or --fake_dir <folder>.')
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load generated volumes ----------------------------------------------
    if args.fake_dir:
        fake_paths = sorted(glob.glob(os.path.join(args.fake_dir, '*.tif')) +
                            glob.glob(os.path.join(args.fake_dir, '*.tiff')))
    else:
        fake_paths = [args.fake]
    if args.max_fakes:
        fake_paths = fake_paths[:args.max_fakes]
    if not fake_paths:
        raise SystemExit('No generated .tif files found.')
    print(f'Loading {len(fake_paths)} generated volume(s)...')
    fakes = np.stack([load_binary(p) for p in fake_paths], axis=0)  # (m, S, S, S)
    if fakes.ndim != 4 or not (fakes.shape[1] == fakes.shape[2] == fakes.shape[3]):
        raise SystemExit(f'Expected cubic 3D generated volumes, got array of shape {fakes.shape}.')
    size = args.crop_size if args.crop_size else fakes.shape[1]

    # ---- load real volume and sample matching crops ---------------------------
    print(f'Loading real volume: {args.real}')
    real_vol = load_binary(args.real)
    if real_vol.ndim != 3:
        raise SystemExit(f'Real volume must be 3D, got shape {real_vol.shape}.')
    n_real = args.n_real_crops if args.n_real_crops else max(len(fake_paths), 4)
    print(f'Sampling {n_real} random {size}^3 crops from real volume {real_vol.shape} ...')
    reals = sample_real_crops(real_vol, size, n_real, rng)

    # ---- porosity (volume fraction of phase == 1) -----------------------------
    real_por = reals.reshape(len(reals), -1).mean(axis=1)
    fake_por = fakes.reshape(len(fakes), -1).mean(axis=1)
    print('\n=== Porosity (volume fraction of phase == 1) ===')
    print(f'  Real : {real_por.mean():.4f} +/- {real_por.std():.4f}   (n={len(reals)})')
    print(f'  Fake : {fake_por.mean():.4f} +/- {fake_por.std():.4f}   (n={len(fakes)})')
    print(f'  |diff|: {abs(real_por.mean() - fake_por.mean()):.4f}')

    # ---- directional two-point correlation S2 ---------------------------------
    print('\nComputing directional S2 for real crops (numba-accelerated, may take a bit) ...')
    r_x, r_y, r_z, r_avg = calculate_two_point_3D(reals, directional=True)
    print('Computing directional S2 for generated volumes ...')
    f_x, f_y, f_z, f_avg = calculate_two_point_3D(fakes, directional=True)

    # align lengths defensively (equal when sizes match)
    def mse(a, b):
        n = min(len(a), len(b))
        return mean_squared_error(a['s2']['mean'].values[:n], b['s2']['mean'].values[:n])

    mse_avg = mse(r_avg, f_avg)
    mse_x, mse_y, mse_z = mse(r_x, f_x), mse(r_y, f_y), mse(r_z, f_z)
    print('\n=== S2 mean-squared-error (real vs fake) ===')
    print(f'  average  : {mse_avg:.3e}   <- same metric train.py minimizes')
    print(f'  x / y / z: {mse_x:.3e} / {mse_y:.3e} / {mse_z:.3e}')

    # ---- plot S2 curves -------------------------------------------------------
    def plot_set(dfx, dfy, dfz, dfavg, color, prefix):
        plt.plot(dfx.index, dfx['s2']['mean'], color=color, lw=0.9, alpha=0.5, ls=':', label=f'{prefix} x')
        plt.plot(dfy.index, dfy['s2']['mean'], color=color, lw=0.9, alpha=0.5, ls='--', label=f'{prefix} y')
        plt.plot(dfz.index, dfz['s2']['mean'], color=color, lw=0.9, alpha=0.5, ls='-.', label=f'{prefix} z')
        m, s = dfavg['s2']['mean'], dfavg['s2']['std']
        plt.plot(dfavg.index, m, color=color, lw=2.4, label=f'{prefix} avg')
        plt.fill_between(dfavg.index, m - s, m + s, color=color, alpha=0.15)

    plt.figure(figsize=(6.8, 5.2))
    plot_set(r_x, r_y, r_z, r_avg, 'tab:blue', 'Real')
    plot_set(f_x, f_y, f_z, f_avg, 'tab:red', 'Fake')
    plt.xlabel('r (px)', fontsize='large')
    plt.ylabel('$S_2(r)$', fontsize='large')
    plt.title(f'S2: real vs fake   |   MSE(avg) = {mse_avg:.2e}\n'
              f'porosity  real={real_por.mean():.3f}  fake={fake_por.mean():.3f}')
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()
    s2_png = os.path.join(args.output_dir, 's2_comparison.png')
    plt.savefig(s2_png, dpi=200)
    plt.close()
    print(f'\nSaved S2 plot: {s2_png}')

    # ---- visual mid-slice montage (qualitative check) -------------------------
    mid = size // 2
    ncol = min(4, max(len(reals), len(fakes)))
    fig, ax = plt.subplots(2, ncol, figsize=(2.4 * ncol, 5))
    if ncol == 1:
        ax = ax.reshape(2, 1)
    for k in range(ncol):
        ax[0, k].imshow(reals[k % len(reals), mid], cmap='gray'); ax[0, k].axis('off')
        ax[1, k].imshow(fakes[k % len(fakes), mid], cmap='gray'); ax[1, k].axis('off')
    ax[0, 0].set_title('Real (top row)', loc='left', fontsize='medium')
    ax[1, 0].set_title('Fake (bottom row)', loc='left', fontsize='medium')
    plt.suptitle(f'Mid-slices (z={mid})')
    plt.tight_layout()
    slices_png = os.path.join(args.output_dir, 'slices_comparison.png')
    plt.savefig(slices_png, dpi=200)
    plt.close()
    print(f'Saved slice montage: {slices_png}')

    # ---- save curves + metrics as CSV -----------------------------------------
    n = min(len(r_avg), len(f_avg))
    curves = pd.DataFrame({
        'r': r_avg.index[:n],
        'real_x': r_x['s2']['mean'].values[:n], 'real_y': r_y['s2']['mean'].values[:n],
        'real_z': r_z['s2']['mean'].values[:n], 'real_avg': r_avg['s2']['mean'].values[:n],
        'real_avg_std': r_avg['s2']['std'].values[:n],
        'fake_x': f_x['s2']['mean'].values[:n], 'fake_y': f_y['s2']['mean'].values[:n],
        'fake_z': f_z['s2']['mean'].values[:n], 'fake_avg': f_avg['s2']['mean'].values[:n],
        'fake_avg_std': f_avg['s2']['std'].values[:n],
    })
    curves.to_csv(os.path.join(args.output_dir, 's2_curves.csv'), index=False)

    summary = pd.DataFrame({
        'metric': ['porosity_real', 'porosity_fake', 'porosity_abs_diff',
                   'mse_s2_avg', 'mse_s2_x', 'mse_s2_y', 'mse_s2_z',
                   'n_real_crops', 'n_fake', 'crop_size'],
        'value': [real_por.mean(), fake_por.mean(), abs(real_por.mean() - fake_por.mean()),
                  mse_avg, mse_x, mse_y, mse_z, len(reals), len(fakes), size],
    })
    summary.to_csv(os.path.join(args.output_dir, 'metrics_summary.csv'), index=False)
    print(f'Saved CSVs: s2_curves.csv, metrics_summary.csv in {args.output_dir}')
    print('\nDone. Lower MSE and closely overlapping S2 curves + matching porosity = good reconstruction.')


if __name__ == '__main__':
    main()
