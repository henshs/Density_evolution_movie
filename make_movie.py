#!/usr/bin/env python3

import os
import sys
import time
import argparse
import multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import M_sun, G, c, m_n
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# Kuibit capabilities
from kuibit.simdir import SimDir
from kuibit.grid_data import UniformGrid

# Unit conversion constants
length_unit = G * M_sun / c**2
density_unit = M_sun / (length_unit**3)
time_unit = length_unit / c
press_unit = M_sun / (length_unit * time_unit**2)

def parse_arguments():
    msg = "This script generates a 2D evolution movie of matter density from Cactus simulation data using Kuibit."
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('--stepsize', dest='stepsize', default=1, type=int, 
                        help='Steps of the iteration (stride between frames)')
    parser.add_argument('--fps', dest='fps', default=60, type=int, 
                        help='Frames per second for the output video')
    parser.add_argument('--serial', dest='serial', default=False, action='store_true', 
                        help='Run the frame generation sequentially instead of in parallel')
    parser.add_argument('--nproc', dest='nproc', default=multiprocessing.cpu_count()-2, type=int,
                        help='Number of parallel processes to use (defaults to total CPU cores minus 2)')
    return parser.parse_args()

def get_frames(i, rho, grid, frames_dir):
    """Generates and saves a single frame plot for time index i."""
    try:
        time_val = rho.times[i]
        rho_t = rho.get_time(time_val).to_UniformGridData_from_grid(grid)
        data = rho_t * density_unit.value
        rho.clear_cache()
        
        # Plot configuration
        cmap_type = plt.get_cmap("hot")  
        interp2d_type = 'gaussian'
        
        data_max = 10 * (2.7 * 10**17)  # 5.5 * nuclear saturation density
        data_min = np.log10(10**12)     # Lower bound filter for visual clarity
        data_max = np.log10(data_max)   
        
        extent_area = (grid.x0[0], grid.x1[0], grid.x0[1], grid.x1[1])
        
        # Thread-safe figure initialization
        fig, ax = plt.subplots(figsize=(6, 5))
        
        plot = ax.imshow(
            X=np.log10(data.data),
            cmap=cmap_type,
            aspect='equal',
            interpolation=interp2d_type,
            vmin=data_min,
            vmax=data_max,
            origin='lower',
            extent=extent_area
        )
        
        time_show = format(time_val * time_unit.value * 1000, '.2f')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        
        tick_list = [-100/1.476, -50/1.476, 0, 50/1.476, 100/1.476]
        tick_label = ['-100', '-50', '0', '50', '100']
        ax.set_xticks(tick_list)
        ax.set_xticklabels(tick_label)
        ax.set_yticks(tick_list)
        ax.set_yticklabels(tick_label)
        ax.set_title(f't = {time_show} ms')
        
        bar = fig.colorbar(plot, ax=ax)
        bar.set_label(r'$Log_{10}(\rho)\:(kg \cdot m^{-3})$')
        
        fig_name = f"rho_{i:05d}.png"
        fig.savefig(os.path.join(frames_dir, fig_name), dpi=300, bbox_inches='tight')
        plt.close(fig)  # Prevents severe memory leaks in long loops
    except Exception as e:
        print(f"Error processing frame {i}: {e}", file=sys.stderr)

def main():
    args = parse_arguments()
    t1 = time.time()

    # Paths setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_dir, "your-density-data-location"))
    frames_dir = os.path.join(script_dir, 'frames_dir/')
    movie_dir = os.path.join(script_dir, 'movie_dir/')
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(movie_dir, exist_ok=True)

    print(f"Loading Cactus simulation data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    sim_dir = SimDir(data_path)
    vars2D = sim_dir.gf.xy
    rho = vars2D["rho"]

    # Grid configuration
    size = 100 / 1.476
    grid = UniformGrid([int(size * 10), int(size * 10)], x0=[-size, -size], x1=[size, size])

    # Iteration steps configuration
    step_available = len(rho.times)
    step_size = args.stepsize
    frame_indices = list(range(0, step_available, step_size))
    calculate_for_steps = len(frame_indices)

    print(f"Starting frame generation ({calculate_for_steps} frames total)...")

    if args.serial:
        for i in tqdm(frame_indices, desc="Generating Frames", unit='frame'):
            get_frames(i, rho, grid, frames_dir)
    else:
        num_cpu = args.nproc
        timeout = 9999999
        print(f"Running parallel processing using {num_cpu} CPUs.")
        with tqdm_joblib(desc="Generating Frames", total=calculate_for_steps, unit='frame'):
            Parallel(n_jobs=num_cpu, timeout=timeout)(
                delayed(get_frames)(i, rho, grid, frames_dir) for i in frame_indices
            )
        
    print("Reading frames for video rendering...")
    images = [img for img in os.listdir(frames_dir) if img.endswith(".png")]
    if not images:
        print("Error: No frames were successfully generated.", file=sys.stderr)
        sys.exit(1)
        
    images.sort() # Natural sorting safe due to zero-padded naming format format index:05d
    
    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, _ = first_frame.shape

    movie_name = 'rho-movie.mp4'
    video_path = os.path.join(movie_dir, movie_name)
    
    # H.264 compatible codec via OpenH264/FFmpeg backends inside OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))

    print("Encoding movie...")
    for image in tqdm(images, desc="Compiling Video", unit='frame'):
        img_data = cv2.imread(os.path.join(frames_dir, image))
        video.write(img_data)

    video.release()
    cv2.destroyAllWindows()

    t2 = time.time()
    total_time = time.strftime("%H:%M:%S", time.gmtime(t2 - t1))
    print(f"Success! Movie: {movie_name} successfully created in: {total_time}")

if __name__ == "__main__":
    main()

