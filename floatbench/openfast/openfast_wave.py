# pylint: skip-file
# Released post-processing pipeline of the FLOATBench labels, kept verbatim
# (only the colour import points to floatbench.colors).
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
# pylint: disable=import-error
# pylint: disable=no-member
"""Wave Parameter Analysis and OpenFAST Sampling Script."""

import csv

from typing import Tuple

import os
import math
import random
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.special import iv

from floatbench.colors import COLORS_DICT

class WaveDistributionAnalyzer:
    """
    Analyzes and visualizes wave-related environmental conditions.

    This class computes probability density functions (PDFs) for wind speed, 
    wave height, wave period, and wave direction. It also generates
    visualizations of these distributions.
    """

    def __init__(self, num_points: int = 1000):
        """
        Initializes the WaveDistributionAnalyzer instance.

        Args:
            num_points (int): Number of points for evaluating distributions.
              Defaults to 1000.
        """
        self.num_points = num_points

        # Define ranges for environmental parameters
        self.wind_speeds = np.linspace(0.01, 25, self.num_points)
        self.wave_heights = np.linspace(0.01, 14, self.num_points)
        self.wave_periods = np.linspace(0.01, 21, self.num_points)
        self.wave_directions = np.linspace(-np.pi, np.pi, self.num_points)

    def compute_exp_weibull_pdf(self, values: np.ndarray, alpha: float,
                                beta: float, delta: float) -> np.ndarray:
        """
        Computes the PDF of the exponentiated Weibull distribution.

        Args:
            values (np.ndarray): Input array.
            alpha (float): Scale parameter.
            beta (float): Shape parameter.
            delta (float): Exponentiation parameter.

        Returns:
            np.ndarray: PDF values for the input array.
        """
        exp_term = np.exp(-(values / alpha)**beta)
        return delta * (beta / alpha) * (values / alpha)**(beta - 1) * (
            1 - exp_term)**(delta - 1) * exp_term

    def compute_turbulence_intensity(self, wind_speed: np.ndarray,
                                     reference_intensity: float) -> np.ndarray:
        """
        Calculates turbulence intensity as a percentage for a given wind speed.

        Args:
            wind_speed (np.ndarray): Array of wind speeds (m/s).
            reference_intensity (float): Reference turbulence intensity.

        Returns:
            np.ndarray: Turbulence intensity values as percentages.
        """
        return reference_intensity * (0.75 * wind_speed +
                                      5.6) / wind_speed * 100

    def compute_wave_height_pdf(self, wind_speed: float) -> np.ndarray:
        """
        Computes the wave height PDF conditioned on wind speed.

        Args:
            wind_speed (float): Wind speed (m/s).

        Returns:
            np.ndarray: Probability density values for wave heights.
        """
        beta = 1.1 + 1.37 / (1 + np.exp(-0.27 * (wind_speed - 15.86)))
        alpha = (1.25 + 0.01 * wind_speed**1.98) / (2.0445**(1 / beta))
        delta = 5
        return self.compute_exp_weibull_pdf(self.wave_heights, alpha, beta,
                                            delta)

    def compute_wave_period_pdf(self, wave_height: float) -> np.ndarray:
        """
        Computes the wave period PDF conditioned on wave height.

        Args:
            wave_height (float): Significant wave height (m).

        Returns:
            np.ndarray: Probability density values for wave periods.
        """
        mu = np.log(5.94 + 9.42 * np.sqrt(wave_height / 9.81))
        sigma = 0.24 * np.exp(-0.11 * wave_height)
        return 1 / (self.wave_periods * sigma * np.sqrt(2 * np.pi)) * np.exp(
            -1 / (2 * sigma**2) * (np.log(self.wave_periods) - mu)**2)

    def compute_wave_direction_pdf(self, wind_speed: float) -> np.ndarray:
        """
        Computes the PDF for wave direction, conditioned on wind speed.

        Args:
            wind_speed (float): Wind speed (m/s).

        Returns:
            np.ndarray: Probability density values for wave directions 
              (radians).
        """
        mu = 0.24 - 0.05 * wind_speed + 0.0014 * wind_speed**2
        k = 10.04 / (1 + np.exp(-0.28 * (wind_speed - 15.89)))
        return np.exp(
            k * np.cos(self.wave_directions - mu)) / (2 * np.pi * iv(0, k))

    def plot(self, plot_dir: str, save_svg: bool = False, separate: bool = False):
        """
        Generates visualizations for wave and wind parameter distributions.

        This method creates a four-panel plot showing:
        1. Wind speed distribution and turbulence intensities.
        2. Wave height distribution conditioned on wind speed.
        3. Wave period distribution conditioned on wave height.
        4. Wave direction distribution conditioned on wind speed.
        
        Args:
            plot_dir (str): Directory to save the generated plot.
            save_svg (bool): If True, saves plots as SVG files.
            separate (bool): If True, saves height and period plots as separate figures.
        """
        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Configure the figure and grid layout
        if not separate:
            figsize = (14,3)
            ncols = 4
        else:
            figsize = (2.5,3)
            ncols = 1
        

        # Configure the figure layout
        fig1, axes = plt.subplots(1, ncols, figsize=figsize, facecolor='white')
        
        if not separate:
            # Define subplots
            fig1.subplots_adjust(wspace=0.75)
            ax1, ax2, ax3, ax4 = axes
        else:
            ax1 = axes
            

        # Colors for the plots
        
        colors = cm.jet(np.linspace(0, 1, self.num_points))

        # Define ranges for the parameters being analyzed
        wind_range = (self.wind_speeds.min(), self.wind_speeds.max())
        height_range = (self.wave_heights.min(), self.wave_heights.max())

        # Plot 1: Wind speed and turbulence intensity
        ay = ax1.twinx()
        weibull_pdf = self.compute_exp_weibull_pdf(self.wind_speeds,
                                                   alpha=12.773,
                                                   beta=2.345,
                                                   delta=0.88)
        turbulence_types = [
            self.compute_turbulence_intensity(self.wind_speeds, ref_intensity)
            for ref_intensity in [0.16, 0.14, 0.12]
        ]
        ax1.plot(self.wind_speeds,
                 weibull_pdf,
                 label="Weibull PDF",
                 color=COLORS_DICT["blue_paper"],
                 linewidth=1.5,
                 markersize=4)
        for intensity, label, color in zip(turbulence_types,
                                           ["Type A", "Type B", "Type C"],
                                           [COLORS_DICT["light_red_paper"], COLORS_DICT["red_paper"], COLORS_DICT["dark_red_paper"]]):
            ay.plot(self.wind_speeds,
                    intensity,
                    label=label,
                    linestyle="--",
                    color=color,
                    linewidth=1.5,
                    markersize=4)

        # Customize axes and appearance
        ax1.set_xlim(0, wind_range[1])
        ax1.set_ylim(0)
        ax1.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ay.set_ylim(0, 100)
        ax1.set_xlabel("Wind speed (m/s)", fontsize=10, labelpad=10)
        ax1.set_ylabel("Probability density (PDF)", fontsize=10, labelpad=10)
        ay.set_ylabel("Turbulence intensity (%)", fontsize=10, labelpad=10)
        ax1.set_title("Wind distribution", fontsize=12)
        ax1.legend(loc="lower left", frameon=False, fontsize=9)
        ay.legend(loc="upper right", frameon=False, fontsize=9)
        ay.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])
        ax1.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])

        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
                ax1.spines[spine].set_visible(True)
                ax1.spines[spine].set_linewidth(1)
                ax1.spines[spine].set_color(COLORS_DICT["light_gray_paper"])
                ay.spines[spine].set_visible(True)
                ay.spines[spine].set_linewidth(1)
                ay.spines[spine].set_color(COLORS_DICT["light_gray_paper"])


        if separate:
            plot_path = os.path.join(plot_dir, "wind_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            if save_svg:
                    svg_path = os.path.join(plot_dir, "wind_distribution.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    plt.close(fig1)
    
        
        # Plot 2: Wave height distribution
        if separate:
            fig2, axes = plt.subplots(1, ncols, figsize=figsize, facecolor='white')
            ax2 = axes
            
        for i, wind_speed in enumerate(self.wind_speeds):
            ax2.plot(self.wave_heights,
                     self.compute_wave_height_pdf(wind_speed),
                     color=colors[i],
                     linewidth=1.5,
                     markersize=4)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap="jet",
                                   norm=plt.Normalize(vmin=wind_range[0],
                                                      vmax=wind_range[1]))
        if separate:
            cbar = fig2.colorbar(sm, ax=ax2, pad=0.01)
        else:
            cbar = fig1.colorbar(sm, ax=ax2, pad=0.01)
        cbar.set_label("Wind speed (m/s)", fontsize=7, labelpad=7)
        cbar.ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=0.7,
                           labelsize=7,
                           color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)

        # Customize axes and appearance
        ax2.set_xlim(0)
        ax2.set_ylim(0)
        ax2.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ax2.set_xlabel("Wave height (m)", fontsize=10, labelpad=10)
        ax2.set_title("Wave height distribution", fontsize=12)
        ax2.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])
    
        ax2.set_ylabel("Probability density (PDF)", fontsize=10, labelpad=10)

        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
                ax2.spines[spine].set_visible(True)
                ax2.spines[spine].set_linewidth(1)
                ax2.spines[spine].set_color(COLORS_DICT["light_gray_paper"])


        if separate:
            plot_path = os.path.join(plot_dir, "wave_height_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            if save_svg:
                    svg_path = os.path.join(plot_dir, "wave_height_distribution.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    plt.close(fig2)
                    
        # Plot 3: Wave period distribution
        if separate:
            fig3, axes = plt.subplots(1, ncols, figsize=figsize, facecolor='white')
            ax3 = axes
            
        for i, wave_height in enumerate(self.wave_heights):
            ax3.plot(self.wave_periods,
                     self.compute_wave_period_pdf(wave_height),
                     color=colors[i],
                     linewidth=1.5,
                     markersize=4)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap="jet",
                                   norm=plt.Normalize(vmin=height_range[0],
                                                      vmax=height_range[1]))
        if separate:
            cbar = fig3.colorbar(sm, ax=ax3, pad=0.01)
        else:
            cbar = fig1.colorbar(sm, ax=ax3, pad=0.01)
        cbar.set_label("Wave height (m)", fontsize=7, labelpad=7)
        cbar.ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=0.7,
                           labelsize=7,
                           color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)

        # Customize axes and appearance
        ax3.set_xlim(0)
        ax3.set_ylim(0)
        ax3.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ax3.set_xlabel("Wave peak period (s)", fontsize=10, labelpad=10)
        ax3.set_title("Wave peak period distribution", fontsize=12)
        ax3.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])
        ax3.set_ylabel("Probability density (PDF)", fontsize=10, labelpad=10)
        
        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
                ax3.spines[spine].set_visible(True)
                ax3.spines[spine].set_linewidth(1)
                ax3.spines[spine].set_color(COLORS_DICT["light_gray_paper"])


        if separate:
            plot_path = os.path.join(plot_dir, "wave_period_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            if save_svg:
                    svg_path = os.path.join(plot_dir, "wave_period_distribution.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    plt.close(fig3)

        # Plot 4: Wave direction distribution
        if separate:
            fig4, axes = plt.subplots(1, ncols, figsize=figsize, facecolor='white')
            ax4 = axes
            
        for i, wind_speed in enumerate(self.wind_speeds):
            ax4.plot(self.wave_directions / np.pi * 180,
                     self.compute_wave_direction_pdf(wind_speed),
                     color=colors[i],
                     linewidth=1.5,
                     markersize=4)
        

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap="jet",
                                   norm=plt.Normalize(vmin=wind_range[0],
                                                      vmax=wind_range[1]))
        if separate:
            cbar = fig4.colorbar(sm, ax=ax4, pad=0.01)
        else:
            cbar = fig1.colorbar(sm, ax=ax4, pad=0.01)
        cbar.set_label("Wind speed (m/s)", fontsize=7, labelpad=7)
        cbar.ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=0.7,
                           labelsize=7,
                           color=COLORS_DICT["dark_gray_paper"])
        cbar.outline.set_visible(False)

        # Customize axes and appearance
        ax4.set_xlim(-180, 180)
        ax4.set_ylim(0)
        ax4.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ax4.set_xlabel("Wave mean direction (deg)", fontsize=10, labelpad=10)
        ax4.set_title("Wave mean direction distribution", fontsize=12)
        ax4.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])
        ax4.set_ylabel("Probability density (PDF)", fontsize=10, labelpad=10)

        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
                ax4.spines[spine].set_visible(True)
                ax4.spines[spine].set_linewidth(1)
                ax4.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Save the plot
        if not separate:
            plot_path = os.path.join(plot_dir, "wave_distributions.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            
            if save_svg:
                    svg_path = os.path.join(plot_dir,
                                        "wave_distributions.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    
            plt.close(fig1)
            
        else:
            plot_path = os.path.join(plot_dir, "wave_mean_direction_distributions.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)

            if save_svg:
                    svg_path = os.path.join(plot_dir,
                                        "wave_mean_direction_distributions.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    
            plt.close(fig4)
       
       



class OpenFASTWaveSampler(WaveDistributionAnalyzer):
    """
    Generates input samples of wave parameters for use in OpenFAST simulations.

    This subclass of WaveDistributionAnalyzer provides methods to sample wave
    heights and wave peak periods based on given wind speed and statistical
    distributions. These samples are designed to serve as input parameters for
    OpenFAST simulations.
    """

    def __init__(self,
                 wind_speed: float,
                 num_samples_requested: int = 10,
                 wave_seed: int = None):
        """
            Initializes the OpenFASTWaveSampler instance.

            Args:
                wind_speed (float): Mean wind speed for the analysis (m/s).
                num_samples_requested (int, optional): Minimum number of samples
                  requested by the user. The actual number of samples used will
                  be the smallest value of the form 2^k - 1 that is greater than
                  or equal to this value to preserve hierarchical consistency.
                wave_seed (int): Seed for reproducible generation of incident
                  wave parameters.
            """
        super().__init__()
        self.wind_speed = wind_speed
        self.num_samples_requested = num_samples_requested
        (self.num_samples_generate,
         self.sample_points) = self.compute_hierarchical_sample_points()
        self.wave_seed = wave_seed

    def compute_hierarchical_sample_points(self) -> tuple[int, list[float]]:
        """
        Computes a list of sample points uniformly distributed in [0, 1]
        using hierarchical dyadic partitioning. The number of points is the
        smallest 2^k - 1 greater than or equal to num_samples_requested.

        Returns:
            tuple:
                - num_points (int): Total number of generated hierarchical
                  points (2^k - 1).
                - sample_points (list[float]): List of points in [0, 1]
                  hierarchically and uniformly distributed.
        """
        level = math.ceil(math.log2(self.num_samples_requested + 1))
        num_points = 2**level - 1
        sample_points = sorted({
            i / 2**(l + 1)
            for l in range(level)
            for i in range(1, 2**(l + 1), 2)
        })
        return num_points, sample_points

    def plot_wave_sampling_structure(self, plot_dir: str, save_svg: bool = False):
        """
        Plots the structure of sampling used to generate sampling points.

        This method generates a plot that includes:
        - The hierarchical levels used for sample generation.
        - Old samples (gray circles) and new samples (red circles).
        - A dashed black line indicating the current level used for sampling.
        - Labels showing the number of accumulated samples at each level.

        Args:
            plot_dir (str): Directory to save the generated plot.
            save_svg (bool): If True, saves plots as SVG files.
        """

        # Compute sampling info
        current_level = math.ceil(math.log2(self.num_samples_requested + 1))

        # Set global style
        plt.style.use('fivethirtyeight')

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 3), facecolor='white')

        accumulated_points = set()
        new_sample_plotted = False

        for lvl in range(current_level + 2):
            step = 2**(lvl + 1)
            current_points = [i / step for i in range(1, step, 2)]
            new_points = [
                p for p in current_points if p not in accumulated_points
            ]
            accumulated_points.update(new_points)

            all_points = sorted(accumulated_points)
            y_all = [lvl + 1] * len(all_points)
            y_new = [lvl + 1] * len(new_points)

            # Plot old samples (gray circles)
            ax.scatter(all_points, y_all, color=COLORS_DICT["grey_paper"], s=60, marker='o')

            # Plot new samples (red circles)
            if new_points:
                label_new = 'New samples' if not new_sample_plotted else None
                ax.scatter(new_points,
                           y_new,
                           color=COLORS_DICT["red_paper"],
                           s=60,
                           marker='o',
                           label=label_new)
                new_sample_plotted = True

            # Add total samples text
            n = len(all_points)
            label = "sample" if n == 1 else "samples"
            ax.text(1.01, lvl + 1, f"{n} {label}", va='center', fontsize=7)

        # Highlight current level with black "x"
        ax.plot([0, 1], [current_level, current_level],
                color=COLORS_DICT["dark_gray_paper"],
                linestyle='--',
                linewidth=0.9,
                label='Current level')

        # Add manual legend entry for old samples
        ax.scatter([], [],
                   facecolors=COLORS_DICT["grey_paper"],
                   edgecolors=COLORS_DICT["grey_paper"],
                   s=60,
                   marker='o',
                   label='Old samples')

        # Customize appearance
        ax.set_title("Hierarchical Sample Structure", fontsize=12)
        ax.set_xlabel("Points in [0, 1]", fontsize=10, labelpad=10)
        ax.set_ylabel("Level", fontsize=10, labelpad=10)
        ax.set_xlim(0, 1.15)
        ax.set_ylim(0, current_level + 3)
        ax.legend(loc="lower right",
                  fontsize=9,
                  ncols=3,
                  bbox_to_anchor=(1.0, 0),
                  frameon=False)
        ax.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])
        ax.set_facecolor('white')

        # Hide spines
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)   
            ax.spines[spine].set_linewidth(1)    
            ax.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Save figure
        plot_path = os.path.join(plot_dir, "wave_sampling_structure.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        
        if save_svg:
            svg_path = os.path.join(plot_dir,
                                 "wave_sampling_structure.svg")
            plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
            
        plt.close(fig)

    def sample_wave_seeds(self) -> list[int]:
        """
        Generates a list of wave seeds for reproducibility.

        This method generates deterministic or random 32-bit signed integer
        seeds for wave sampling. If a fixed seed is provided (`self.wave_seed`),
        it is replicated for all samples; otherwise, random seeds are generated.

        Returns:
            list[int]: List of 32-bit signed integer seeds for each wave sample.
            
        Raises:
            ValueError: Wave_seed is not in valid 32-bit signed integer range.
        """
        # Total number of seeds required
        num_seeds = self.num_samples_generate**2

        if self.wave_seed is not None:
            if self.wave_seed < -2147483648 or self.wave_seed > 2147483647:
                raise ValueError("wave_seed must be a 32-bit signed integer in "
                                 "[-2147483648, 2147483647].")
            seeds = [self.wave_seed] * num_seeds
        else:
            seeds = [
                random.randint(-2147483648, 2147483647)
                for _ in range(num_seeds)
            ]

        return seeds

    def sample_wave_heights(
            self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Samples significant wave heights (Hs) based on wind speed.

        This method computes the PDF and CDF for wave heights conditioned on the 
        wind speed and deterministically generates sample points from the
        distribution using hierarchical uniform spacing.

        Returns:
            tuple:
                - pdf (np.ndarray): Probability density function values for Hs.
                - cdf (np.ndarray): Cumulative distribution function values for
                  Hs.
                - sample_points (np.ndarray): Deterministic sample points in
                  [0, 1] used for interpolation.
                - sampled_heights (list): Sampled Hs values.
        """
        # Compute PDF and normalized CDF for wave heights based on wind speed
        pdf = self.compute_wave_height_pdf(self.wind_speed)
        cdf = np.cumsum(pdf) / np.sum(pdf)

        # Use hierarchical deterministic samples
        sample_points = np.array(self.sample_points)
        sampled_heights = np.interp(sample_points, cdf, self.wave_heights)

        return pdf, cdf, sample_points, sampled_heights.tolist()

    def sample_wave_periods(
            self, wave_height: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Samples wave peak periods (Tp) based on a significant wave height (Hs).

        This method computes the PDF and CDF for wave peak periods conditioned 
        on the given Hs and deterministically generates sample points from the
        distribution using hierarchical uniform spacing.

        Args:
            wave_height (float): Significant wave height (m).

        Returns:
            tuple:
                - pdf (np.ndarray): Probability density function values for Tp.
                - cdf (np.ndarray): Cumulative distribution function values for
                  Tp.
                - sample_points (np.ndarray): Deterministic sample points in
                  [0, 1] used for interpolation.
                - sampled_periods (list): Sampled Tp values.
        """
        # Compute PDF and normalized CDF for wave periods based on wave height
        pdf = self.compute_wave_period_pdf(wave_height)
        cdf = np.cumsum(pdf) / np.sum(pdf)

        # Use hierarchical deterministic samples
        sample_points = np.array(self.sample_points)
        sampled_periods = np.interp(sample_points, cdf, self.wave_periods)

        return pdf, cdf, sample_points, sampled_periods.tolist()

    def sample_wave_parameters(
            self,
            samples_ref_df: pd.DataFrame = None) -> tuple[list, list, list]:
        """
        Samples significant wave heights (Hs) and wave peak periods (Tp).

        This method first generates samples for wave heights (Hs) and then, for
        each sampled Hs, generates corresponding samples for wave peak periods
        (Tp).
        
        Args:
            samples_ref_df (pd.DataFrame): Reference wave samples DataFrame with
                columns 'wave_hs', 'wave_tp', and 'wave_seed'.

        Returns:
            tuple:
                - heights_samples (list): List of sampled significant wave
                  heights.
                - periods_samples (list): List of lists, where each sublist
                  contains sampled Tp values corresponding to a specific Hs.
                - seeds_samples (list): Corresponding wave seeds, or sampled if
                  not provided.
                  
        Raises:
            ValueError: If no matching seed is found for a given (Hs, Tp) pair
              in the reference DataFrame.
        """
        _, _, _, heights_samples = self.sample_wave_heights()
        periods_samples = [
            self.sample_wave_periods(height)[3] for height in heights_samples
        ]

        if samples_ref_df is not None:
            seeds_samples = []

            for wave_height, period_list in zip(heights_samples,
                                                periods_samples):
                for wave_period in period_list:

                    match = samples_ref_df[
                        np.isclose(
                            samples_ref_df["wave_hs"], wave_height, atol=1e-6) &
                        np.isclose(
                            samples_ref_df["wave_tp"], wave_period, atol=1e-6)]
                    if not match.empty:
                        seeds_samples.append(match["wave_seed"].values[0])
                    else:
                        raise ValueError(
                            f"No matching seed found for height {wave_height} "
                            f"and period {wave_period} in reference file.")

            return heights_samples, periods_samples, seeds_samples

        return heights_samples, periods_samples, self.sample_wave_seeds()

    def plot_wave_height_sampling(self, plot_dir: str, save_svg: bool = False):
        """
        Visualizes the process of wave height sampling using PDF and CDF.

        This method generates a plot that includes:
        - The PDF of significant wave heights.
        - The CDF of significant wave heights.
        - Samples used for interpolating Hs values from the CDF.
        - Visual cues (lines) showing the relationship between the samples, 
        PDF, and CDF.
        
        Args:
            plot_dir (str): Directory to save the generated plot.
            save_svg (bool): If True, saves plots as SVG files.
        """
        # Generate PDF, CDF, and sampled wave heights
        pdf, cdf, sample_points, sampled_heights = self.sample_wave_heights()

        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Configure the figure and grid layout
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
        
      
        
        # Plot PDF and CDF
        ax.plot(self.wave_heights,
                pdf,
                label='PDF',
                color=COLORS_DICT["blue_paper"],
                linewidth=1.5,
                markersize=4)
        ax.plot(self.wave_heights,
                cdf,
                label='CDF',
                color=COLORS_DICT["red_paper"],
                linewidth=1.5,
                markersize=4)

        # Plot sampled points
        plt.scatter(sampled_heights,
                    sample_points,
                    color=COLORS_DICT["grey_paper"],
                    s=30,
                    label='Samples',
                    zorder=5)

        # Add connecting lines for samples
        for height, sample in zip(sampled_heights, sample_points):
            ax.plot([height, height], [0, sample],
                    color=COLORS_DICT["dark_gray_paper"],
                    linestyle='--',
                    linewidth=0.9,
                    markersize=4)
            ax.plot([0, height], [sample, sample],
                    color=COLORS_DICT["dark_gray_paper"],
                    linestyle='--',
                    linewidth=0.9,
                    markersize=4)

        # Customize axes and appearance
        ax.set_title("Wave height distribution", fontsize=12)
        ax.set_xlabel("Wave height", fontsize=10, labelpad=10)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])
        ax.set_facecolor('white')

        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)   
            ax.spines[spine].set_linewidth(1)    
            ax.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Save the plot
        plot_path = os.path.join(plot_dir, "wave_height_sampling.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        
        if save_svg:
            svg_path = os.path.join(plot_dir,
                                 "wave_height_sampling.svg")
            plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")

        plt.close(fig)

    def plot_wave_period_sampling(self, plot_dir: str, save_svg: bool = False):
        """
        Visualizes the process of wave peak periods sampling using PDF and CDF.

        This method generates a grid of subplots where each subplot corresponds
        to a specific sampled wave height.
        Each plot includes:
        - The PDF of wave peak periods.
        - The CDF of wave peak periods.
        - Samples used for interpolating Tp values from the CDF.
        - Visual cues (lines) showing the relationship between the samples,
        PDF, and CDF.

        Args:
            plot_dir (str): Directory to save the generated plot.
            save_svg (bool): If True, saves plots as SVG files.
        """
        # Generate data for wave height and wave period sampling
        _, _, _, sampled_heights = self.sample_wave_heights()

        pdf_list = []
        cdf_list = []
        sample_points_list = []
        sampled_periods_list = []

        for height in sampled_heights:
            (pdf, cdf, sample_points,
             sampled_periods) = self.sample_wave_periods(height)
            pdf_list.append(pdf)
            cdf_list.append(cdf)
            sample_points_list.append(sample_points)
            sampled_periods_list.append(sampled_periods)

        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Configure the figure layout dynamically
        n_cols = min(self.num_samples_generate, 4)
        n_rows = -(-self.num_samples_generate // n_cols)

        # Configure the figure and grid layout
        fig = plt.figure(figsize=(7.5, 4.5), facecolor='white')
        gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.3, hspace=0.55)
        axes = [
            plt.subplot(gs[i], facecolor='white')
            for i in range(self.num_samples_generate)
        ]

        
        for i, ax in enumerate(axes):
            # Plot PDF and CDF
            ax.plot(self.wave_periods,
                    pdf_list[i],
                    label="PDF",
                    color=COLORS_DICT["blue_paper"],
                    linewidth=1.5,
                    markersize=4)
            ax.plot(self.wave_periods,
                    cdf_list[i],
                    label="CDF",
                    color=COLORS_DICT["red_paper"],
                    linewidth=1.5,
                    markersize=4)

            # Plot sampled points
            ax.scatter(sampled_periods_list[i],
                       sample_points_list[i],
                       color=COLORS_DICT["grey_paper"],
                       s=30,
                       label='Samples',
                       zorder=5)

            # Add connecting lines for samples
            for period, sample in zip(sampled_periods_list[i],
                                      sample_points_list[i]):
                ax.plot([period, period], [0, sample],
                        color=COLORS_DICT["dark_gray_paper"],
                        linestyle='--',
                        linewidth=0.9,
                        markersize=4)
                ax.plot([0, period], [sample, sample],
                        color=COLORS_DICT["dark_gray_paper"],
                        linestyle='--',
                        linewidth=0.9,
                        markersize=4)

            # Customize each subplot
            ax.set_title(f"Wave height: {sampled_heights[i]:.2f} m",
                         fontsize=12)
            ax.set_xlim(0)
            ax.set_ylim(0)
            ax.set_xlabel("Wave peak period (s)", fontsize=10)
            ax.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
            ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])

            # Hide unnecessary spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(True)   
                ax.spines[spine].set_linewidth(1)    
                ax.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Add a global title and legend
        fig.suptitle("Wave peak period distribution", fontsize=12)
        fig.legend(labels=["PDF", "CDF", "Samples"],
                   loc="lower right",
                   ncol=1,
                   fontsize=9,
                   bbox_to_anchor=(1.0, -0.05),
                   frameon=False)

        # Save the plot
        plot_path = os.path.join(plot_dir, "wave_period_sampling.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        
        if save_svg:
            svg_path = os.path.join(plot_dir,
                                 "wave_period_sampling.svg")
            plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")

        plt.close(fig)

    def plot_wave_distributions(self, plot_dir: str, save_svg: bool = False,
     separate: bool = False):
        """
        Visualizes wave distributions for a specific wind speed.

        This method generates a two-panel plot:
        1. Wave height distribution conditioned on wind speed.
        2. Wave peak period distributions conditioned on sampled wave heights.

        Args:
            plot_dir (str): Directory to save the generated plot.
            save_svg (bool): If True, saves plots as SVG files.
            separate (bool): If True, saves height and period plots as separate figures.
        """
        # Sample wave heights and periods
        sampled_heights, _, _ = self.sample_wave_parameters()

        # Set global plot style
        plt.style.use('fivethirtyeight')
        
        # Colors for the plots
        colors = cm.jet(np.linspace(0, 1, self.num_samples_generate))
        
        
        if not separate:
            figsize = (6,3)
            ncols = 2
        else:
            figsize = (3,3)
            ncols = 1
        

        # Configure the figure layout
        fig1, axes = plt.subplots(1, ncols, figsize=figsize, facecolor='white')
        if not separate:
            ax1, ax2 = axes
        else:
            ax1 = axes

        # Plot 1: Wave height distribution
        ax1.plot(self.wave_heights,
                    self.compute_wave_height_pdf(self.wind_speed),
                    color = colors[0],
                    linewidth=1.5,
                    markersize=4,
                    label=f"Wind speed: {self.wind_speed:.2f} m/s")

        # Customize axes and appearance
        ax1.set_xlim(0)
        ax1.set_ylim(0)
        ax1.set_xlabel("Wave height (m)", fontsize=10)
        ax1.set_ylabel("Probability density (PDF)", fontsize=10)
        ax1.set_title("Wave height distribution", fontsize=12)
        ax1.legend(frameon=False, ncol=3, fontsize=9, loc="upper right")
        ax1.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ax1.tick_params(axis='both',
                            which='major',
                            length=4,
                            width=1,
                            labelsize=10,
                            color=COLORS_DICT["dark_gray_paper"])
        ax1.set_facecolor('white')

        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
                ax1.spines[spine].set_visible(True)
                ax1.spines[spine].set_linewidth(1)
                ax1.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Save the plot
        if separate:
            plot_path = os.path.join(plot_dir, "wave_height_distribution.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            if save_svg:
                    svg_path = os.path.join(plot_dir, "wave_height_distribution.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    plt.close(fig1)
        
        
        # Plot 2: Wave peak period distributions
        # Configure the figure layout
        if separate:
            fig2, axes = plt.subplots(1, ncols, figsize=figsize, facecolor='white')
            ax2 = axes
            
        for i, height in enumerate(sampled_heights):
                ax2.plot(self.wave_periods,
                        self.compute_wave_period_pdf(height),
                        color=colors[i],
                        linewidth=1.5,
                        markersize=4,
                        label=f"{height:.2f} m")
                
        # Customize axes and appearance
        ax2.set_xlim(0)
        ax2.set_ylim(0)
        ax2.set_xlabel("Wave peak period (s)", fontsize=10)
        ax2.set_title("Wave peak period distributions", fontsize=12)
        if separate:
            ax2.set_ylabel("Probability density (PDF)", fontsize=10)
        ax2.legend(title="Wave Height",
                    frameon=False,
                    fontsize=9,
                    title_fontsize=10,
                    loc="upper right")
        ax2.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])
        ax2.tick_params(axis='both',
                            which='major',
                            length=4,
                            width=1,
                            labelsize=10,
                            color=COLORS_DICT["dark_gray_paper"])
        ax2.set_facecolor('white')

        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
                    ax2.spines[spine].set_visible(True)   
                    ax2.spines[spine].set_linewidth(1)    
                    ax2.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Save the plot
        if not separate:
            plot_path = os.path.join(plot_dir, "wave_distributions_wind.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            
            if save_svg:
                    svg_path = os.path.join(plot_dir,
                                        "wave_peak_period_distributions.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    
            plt.close(fig1)
            
        else:
            plot_path = os.path.join(plot_dir, "wave_peak_period_distributions.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)

            if save_svg:
                    svg_path = os.path.join(plot_dir,
                                        "wave_peak_period_distributions.svg")
                    plt.savefig(svg_path, bbox_inches='tight', transparent=True, format="svg")
                    
            plt.close(fig2)
            

    def generate_waves(
        self,
        csv_dir: str = None,
        samples_ref_df: pd.DataFrame = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates waves, saves them to CSV.

        Args:
            csv_dir (str): Directory where the CSV file will be saved. 
              If None, CSV is not saved.
            samples_ref_df (pd.DataFrame): Reference wave samples DataFrame with
                columns 'wave_hs', 'wave_tp', and 'wave_seed'.

        Returns:
            Tuple of arrays: (wave_seeds, wave_heights, wave_periods).
        """

        # Sample wave parameters
        (heights_samples, periods_samples,
         seeds_samples) = self.sample_wave_parameters(samples_ref_df)

        waves = []
        # Build list of wave entries with [id, seed, height, period]
        for i, (height,
                periods) in enumerate(zip(heights_samples, periods_samples)):
            for j, period in enumerate(periods):
                wave_idx = i * len(periods) + j
                waves.append(
                    [wave_idx, seeds_samples[wave_idx], height, period])

        # Save to CSV if directory is provided
        if csv_dir is not None:
            os.makedirs(csv_dir, exist_ok=True)
            output_csv = os.path.join(csv_dir, "wave_parameters.csv")
            with open(output_csv, mode="w", newline="",
                      encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    'wave id', 'wave seed', 'wave height (m)',
                    'wave peak period (s)'
                ])
                writer.writerows(waves)

        # Convert list to NumPy array and split columns
        waves_array = np.array(waves)
        wave_seeds = waves_array[:, 1].astype(int)
        wave_heights = waves_array[:, 2].astype(float)
        wave_periods = waves_array[:, 3].astype(float)

        return wave_seeds, wave_heights, wave_periods
