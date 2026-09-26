# pylint: skip-file
# Released post-processing pipeline of the FLOATBench labels, kept verbatim
# (only the colour import points to floatbench.colors).
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-lines
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=import-error
# pylint: disable=fixme
# pylint: disable=possibly-used-before-assignment
"""Fatigue analysis of OpenFAST results."""

import os
import json
import math
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cbook import boxplot_stats
from scipy.integrate import quad
import rainflow
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from . import openfast_results
from . import openfast_wave


class Tower:
    #TODO: MOVE THIS CLASS TO THE TOWER GENERATRION
    """
    Represents the geometric configuration of a wind turbine tower.

    Can be constructed either from direct input arrays or from a structured JSON
      file.

    Attributes:
        diameter_transitions (np.ndarray): Outer diameters at height transitions
          [m].
        z_transitions (np.ndarray): Vertical coordinates at diameter transitions
          [m].
        thickness_sections (np.ndarray): Wall thickness per segment [m].
        radius_transitions (np.ndarray): Radii at height transitions [m].
        num_sections (int): Number of tower segments.
        height (float): Total tower height [m].
        mean_radius_sections (np.ndarray): Mean radius per segment [m].
        mean_z_sections (np.ndarray): Mean z-coordinate per segment [m].
        num_uniform_points (int): Number of evenly spaced z samples.
        z_uniform_points (List[float]): Uniformly spaced height values [m].
    """

    def __init__(self,
                 diameter_transitions: Optional[List[float]] = None,
                 z_transitions: Optional[List[float]] = None,
                 thickness_sections: Optional[List[float]] = None,
                 json_path: Optional[str] = None,
                 num_uniform_points: int = 30):
        """
        Initializes the Tower instance.

        Args:
            diameter_transitions: Outer diameters at each height level [m].
            z_transitions: Corresponding vertical coordinates [m].
            thickness_sections: Wall thickness for each tower segment [m].
            json_path: Path to structured JSON geometry file.
            num_uniform_points: Number of uniformly sampled points along height.
            grid: List of normalized heights for analysis.
        """
        if json_path:
            self._load_json(json_path)
        elif (diameter_transitions is not None and z_transitions is not None and
              thickness_sections is not None):
            self.diameter_transitions = np.array(diameter_transitions)
            self.z_transitions = np.array(z_transitions)
            self.thickness_sections = np.array(thickness_sections)
        else:
            raise ValueError("Provide either a JSON path or all three arrays.")

        if len(self.diameter_transitions) != len(self.z_transitions):
            raise ValueError(
                "Length mismatch: diameter_transitions and z_transitions must "
                "be equal.")
        if len(self.thickness_sections) != len(self.z_transitions) - 1:
            raise ValueError(
                "thickness_sections must have length len(z_transitions) - 1.")

        self.radius_transitions = self.diameter_transitions / 2
        self.num_sections = len(self.thickness_sections)
        self.height = self.z_transitions[-1] - self.z_transitions[0]

        self.mean_radius_sections = (self.radius_transitions[:-1] +
                                     self.radius_transitions[1:]) / 2
        self.mean_z_sections = (self.z_transitions[:-1] +
                                self.z_transitions[1:]) / 2

        self.num_uniform_points = num_uniform_points
        self.z_uniform_points = [
            self.z_transitions[0] +
            (i - 0.5) * self.height / self.num_uniform_points
            for i in range(3, self.num_uniform_points, 3)
        ]
        np.set_printoptions(precision=18, suppress=False)
        self.grid = np.round(self.z_transitions / self.height, 18)

    def _load_json(self, json_path: str) -> None:
        """Loads tower geometry from a structured JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        try:
            diameter_dict = data["diameter_transitions (m)"]
            height_dict = data["z_transitions (m)"]
            thickness_dict = data["thickness_sections (m)"]
        except KeyError as e:
            raise KeyError(f"Missing required field in JSON: {e}") from e

        diameter = [
            diameter_dict[k]
            for k in sorted(diameter_dict, key=lambda x: int(x[1:]))
        ]
        height = [
            height_dict[k]
            for k in sorted(height_dict, key=lambda x: int(x[1:]))
        ]
        thickness = [
            thickness_dict[k]
            for k in sorted(thickness_dict, key=lambda x: int(x[1:]))
        ]

        self.diameter_transitions = np.array(diameter)
        self.z_transitions = np.array(height)
        self.thickness_sections = np.array(thickness)

    def save(self, path: str) -> None:
        """Saves tower geometry to a structured JSON file."""
        if not self.radius_transitions.any() or not self.z_transitions.any(
        ) or not self.thickness_sections.any():
            raise ValueError("Tower geometry data is incomplete.")

        data = {
            "geom_id": "custom",
            "diameter_transitions (m)": {
                f"d{i}": r for i, r in enumerate(self.diameter_transitions)
            },
            "z_transitions (m)": {
                f"h{i}": z for i, z in enumerate(self.z_transitions)
            },
            "thickness_sections (m)": {
                f"t{i+1}": t for i, t in enumerate(self.thickness_sections)
            }
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def convert_ansys_format(self,
                             save_path: Optional[str] = None
                            ) -> Dict[str, List[float]]:
        """
        Converts tower geometry to ANSYS format with 3 decimal precision.

        Args:
            save_path: Optional path to save the output as a JSON file.

        Returns:
            Dictionary with:
                - 'radius (m)'      : outer radius at each transition [m],
                - 'heights (m)'     : height of each segment [m],
                - 'thickness (mm)'  : wall thickness per segment [mm].
        """
        result = {
            "geom_id":
                "custom",
            "radius (m)":
                np.round(self.diameter_transitions / 2, 3).tolist(),
            "heights (m)":
                np.round(np.diff(self.z_transitions), 3).tolist(),
            "thickness (mm)":
                np.round(self.thickness_sections * 1000, 3).tolist()
        }

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

        return result


class TowerFatigueAnalysis(openfast_results.OpenFASTOutputAnalysis):
    """
    Analyzes tower fatigue using moment time series and SN-curve models.

    Attributes:
        tower (Tower): Instance of Tower containing geometric definition.
        moment_column (str): Identifier for moment series to use (e.g., 'mfa').
        moment_data (pd.DataFrame): Time series data filtered by required
          moments and min_time.
        sn_intercepts_log10 (List[float]): Intercepts of SN-curve in base-10
          logarithmic scale.
        sn_slopes (List[float]): Slopes of the SN-curve.
        thickness_reference (float): Reference wall thickness in millimeters.
        thickness_exponent (float): Exponent for thickness correction.
        fatigue_life_threshold (float): Threshold above which second SN slope
          is used.
        min_time (float): Time after which data is considered (in seconds).
        num_bins (int): Number of histogram bins for stress range plots.
        analysis_dir (str): Path to directory for saving analysis output.
        z_samples (List[float]): Discrete heights at which analysis is
          evaluated.
        moment_column_config (dict): Mapping of moment identifiers to column
          names.
        moment_data (pd.DataFrame): Filtered time series of moments after
          min_time.
    """

    MOMENT_COLUMN_CONFIG = {
        "mfa": {
            "bottom": "tower_bottom_mfa",
            "middle": [f"tower_{i}_mfa" for i in range(1, 10)],
            "top": "tower_top_mfa"
        }
        # Additional moment columns (e.g., 'mss') can be added here.
    }

    def __init__(self,
                 output_dir: str,
                 tower: Tower,
                 moment_column: str = 'mfa',
                 sn_intercepts_log10: Optional[List[float]] = None,
                 sn_slopes: Optional[List[float]] = None,
                 thickness_reference: float = 25.0,
                 thickness_exponent: float = 0.2,
                 fatigue_life_threshold: float = 1e7,
                 min_time: float = 400.0,
                 num_bins: int = 30,
                 analysis_dir: Optional[str] = None):
        """
        Initializes the fatigue analysis object and loads filtered moment data.

        Args:
            output_dir (str): Directory containing OpenFAST results.
            tower (Tower): Tower geometry definition.
            moment_column (str): Key identifying the moment set to use ('mfa',
              etc.).
            sn_intercepts_log10 (List[float]): List of log10 intercepts for the
              SN curve.
            sn_slopes (List[float]): Corresponding slopes of the SN curve.
            thickness_reference (flaot): Reference thickness for SN correction
              (mm).
            thickness_exponent (flaot): Exponent used in SN correction.
            fatigue_life_threshold (flaot): Threshold cycle count for switching
              slope.
            min_time (flaot): Minimum simulation time to include (in seconds).
            num_bins (int): Number of histogram bins for plots.
            analysis_dir (str): Path to store generated plots and CSVs.
        """
        super().__init__(output_dir)
        self.tower = tower
        self.moment_column = moment_column
        self.sn_intercepts_log10 = sn_intercepts_log10 or [12.010, 15.350]
        self.sn_slopes = sn_slopes or [3, 5]
        self.thickness_reference = thickness_reference
        self.thickness_exponent = thickness_exponent
        self.fatigue_life_threshold = fatigue_life_threshold
        self.min_time = min_time
        self.num_bins = num_bins
        self.analysis_dir = analysis_dir or os.path.join(
            self.output_dir, "fatigue_analysis")
        self.z_samples = self.tower.z_uniform_points
        self.moment_column_config = self._get_moment_labels_config()
        self.moment_data = self._get_moment_data()

    def _get_moment_labels_config(self) -> Dict[str, Union[str, List[str]]]:
        """
        Returns the configuration dictionary for the current moment column.

        Returns:
            dict with keys 'bottom', 'middle', and 'top', which define the
              moment labels.

        Raises:
            ValueError: If the current moment_column is not supported.
        """
        if self.moment_column not in self.MOMENT_COLUMN_CONFIG:
            options = list(self.MOMENT_COLUMN_CONFIG.keys())
            raise ValueError(
                f"Unsupported moment_column: '{self.moment_column}'. "
                f"Supported options are: {options}")
        return self.MOMENT_COLUMN_CONFIG[self.moment_column]

    def _get_moment_data(self) -> pd.DataFrame:
        """
        Retrieves and filters the moment time series data.

        Returns:
            pd.DataFrame: Filtered time series data.
        """
        required_columns = ([self.moment_column_config["bottom"]] +
                            self.moment_column_config["middle"] +
                            [self.moment_column_config["top"]])

        return self.filter_by_metrics_and_time(metrics=required_columns,
                                               min_time=self.min_time,
                                               include_time=False)

    def _interpolate_moment_series(self, z: float) -> np.ndarray:
        """
        Interpolates the moment time series at a given height `z`.

        Args:
            z: Height at which to interpolate the moment.

        Returns:
            Interpolated moment time series as a NumPy array.
        """
        config = self.moment_column_config
        z_samples = self.z_samples

        if z <= z_samples[0]:
            z1, z2 = self.tower.z_transitions[0], z_samples[0]
            m1, m2 = config["bottom"], config["middle"][0]

        elif z >= z_samples[-1]:
            z1, z2 = z_samples[-1], self.tower.z_transitions[-1]
            m1, m2 = config["middle"][-1], config["top"]

        else:
            idx = np.searchsorted(z_samples, z) - 1

            z1, z2 = z_samples[idx], z_samples[idx + 1]
            m1 = config["middle"][idx]
            m2 = config["middle"][idx + 1]

        m_series1 = self.moment_data[self.get_column_from_metric(m1)].values
        m_series2 = self.moment_data[self.get_column_from_metric(m2)].values

        return m_series1 + (m_series2 - m_series1) / (z2 - z1) * (z - z1)

    def _compute_flexural_modulus_circular_section(self, r_out: float,
                                                   thickness: float) -> float:
        """
        Computes the flexural modulus for a hollow circular cross-section.

        Args:
            r_out: Outer radius of the tower section [m].
            thickness: Wall thickness [m].

        Returns:
            Flexural modulus (I / r_out) [m^3], used to compute stress from
              bending moment.
        """
        r_in = r_out - thickness
        inertia = (np.pi / 4) * (r_out**4 - r_in**4)
        return inertia / r_out

    def _compute_thickness_correction(self, thickness: float) -> float:
        """
        Computes the SN-curve thickness correction factor.

        Args:
            thickness: Wall thickness of the current section [mm].

        Returns:
            Correction factor to be applied to the stress range.
        """
        t_eff = max(thickness, self.thickness_reference)
        return (t_eff / self.thickness_reference)**self.thickness_exponent

    def compute_rainflow_cycles(
            self, moment_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes rainflow fatigue cycles from a bending moment time series.

        Args:
            moment_series: 1D NumPy array of bending moment values over time.

        Returns:
            Tuple of:
                - n: NumPy array of cycle counts (e.g., 0.5 or 1.0 per cycle).
                - A: NumPy array of stress (or moment) ranges for each cycle.
        """
        cycles = rainflow.extract_cycles(moment_series)

        cycle_counts, moment_ranges = zip(
            *[(cycle[2], cycle[0]) for cycle in cycles])
        return np.array(cycle_counts), np.array(moment_ranges)

    def compute_damage_section(self, cycle_counts: np.ndarray,
                               stress_ranges: np.ndarray,
                               thickness: float) -> float:
        """
        Computes cumulative fatigue damage for a single tower section.

        Args:
            cycle_counts: Array with the number of occurrences of each cycle
              (e.g., 0.5, 1.0).
            stress_ranges: Array of stress range (amplitudes) per cycle [MPa].
            thickness: Wall thickness of the section [mm].

        Returns:
            Total fatigue damage (unitless).
        """
        # Apply thickness correction to moment range
        correction_factor = self._compute_thickness_correction(thickness)
        corrected_ranges = stress_ranges * correction_factor

        # First-slope SN curve
        fatigue_life = 10**(self.sn_intercepts_log10[0] -
                            self.sn_slopes[0] * np.log10(corrected_ranges))

        # Apply second slope where applicable
        mask = fatigue_life > self.fatigue_life_threshold
        fatigue_life[mask] = 10**(
            self.sn_intercepts_log10[1] -
            self.sn_slopes[1] * np.log10(corrected_ranges[mask]))

        damage = np.sum(cycle_counts / fatigue_life)
        return float(damage)

    def compute_damage_all_sections(
            self,
            save_csv: bool = False,
            save_plot: bool = False,
            save_per_section: bool = False,
            results_dir: Optional[str] = None) -> List[float]:
        """
        Computes fatigue damage at all tower sections.

        Args:
            save_csv: If True, saves fatigue damage per section to a CSV file.
            save_plot: If True, generates plots for fatigue damage and rainflow
              histograms.
            save_per_section: If True, enables section-level saving.
            results_dir: Optional directory where results will be saved, if
              plotting is enabled.

        Returns:
            List of fatigue damage values (unitless) for each tower section.
        """
        damage_list = []
        section_results = []
        all_cycle_counts = []
        all_stress_ranges = []

        for i, (mean_z_section, mean_radius_section,
                thickness_section) in enumerate(zip(
                    self.tower.mean_z_sections, self.tower.mean_radius_sections,
                    self.tower.thickness_sections),
                                                start=1):

            moment_series = self._interpolate_moment_series(mean_z_section)
            section_modulus = self._compute_flexural_modulus_circular_section(
                mean_radius_section, thickness_section)

            cycle_counts, moment_ranges = self.compute_rainflow_cycles(
                moment_series)
            stress_ranges = 1e-3 * moment_ranges / section_modulus

            damage = self.compute_damage_section(cycle_counts=cycle_counts,
                                                 stress_ranges=stress_ranges,
                                                 thickness=thickness_section *
                                                 1000)

            damage_list.append(damage)

            section_results.append({
                "section_id": i,
                "mean z[m]": mean_z_section,
                "mean radius [m]": mean_radius_section,
                "thickness [m]": thickness_section,
                "section modulus [m^3]": section_modulus,
                "damage": damage
            })

            all_cycle_counts.append(cycle_counts)
            all_stress_ranges.append(stress_ranges)

        if save_csv:
            df = pd.DataFrame(section_results)
            os.makedirs(self.analysis_dir, exist_ok=True)
            if results_dir is None:
                results_dir = os.path.join(self.analysis_dir, "fatigue_by_task",
                                           self.output_id)
                os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, "tower_damage_profile.csv")
            df.to_csv(csv_path, index=False)

        if save_plot:
            max_range = np.max(np.concatenate(all_stress_ranges))
            bins = np.linspace(0, max_range, self.num_bins)
            heatmap_matrix = self.compute_rainflow_histograms(
                all_cycle_counts, all_stress_ranges, bins, save_per_section,
                results_dir)
            self.plot_tower_damage_profile(damage_list, results_dir)
            self.plot_tower_rainflow_heatmap(heatmap_matrix, bins, results_dir)

        return damage_list

    def compute_rainflow_histograms(
            self,
            all_cycle_counts: List[np.ndarray],
            all_stress_ranges: List[np.ndarray],
            bins: np.ndarray,
            save_plot: bool,
            plot_dir: Optional[str] = None) -> np.ndarray:
        """
        Computes rainflow histograms for each tower section.

        Args:
            all_cycle_counts: List of arrays with cycle counts per section.
            all_stress_ranges: List of arrays with stress ranges per section.
            bins: Histogram bin edges.
            save_plot: Whether to save individual histogram plots.
            plot_dir: Optional directory where rainflow histograms will be
              saved, if plotting is enabled.

        Returns:
            2D NumPy array (section × bins) with histogram data.
        """
        heatmap_matrix = []

        for i, (cycle_counts,
                stress_ranges) in enumerate(zip(all_cycle_counts,
                                                all_stress_ranges),
                                            start=1):
            if save_plot:
                self.plot_section_rainflow_histogram(stress_ranges, bins,
                                                     cycle_counts, i, plot_dir)

            hist, _ = np.histogram(stress_ranges,
                                   bins=bins,
                                   weights=cycle_counts)
            heatmap_matrix.append(hist)

        return np.array(heatmap_matrix)

    def plot_section_rainflow_histogram(self,
                                        stress_ranges: np.ndarray,
                                        bins: np.ndarray,
                                        cycle_counts: np.ndarray,
                                        section_index: int,
                                        plot_dir: Optional[str] = None) -> None:
        """
        Plots a histogram of rainflow fatigue cycles for a tower section.

        Args:
            stress_ranges: Array of stress range values [MPa] for each rainflow
              cycle.
            bins: Bin edges for histogram calculation (e.g., evenly spaced
              stress intervals).
            cycle_counts: Number of cycles corresponding to each stress range.
            section_index: Index of the current tower section (used for labeling
              plots).
            plot_dir: Optional file directory to save the plot. If not provided,
              uses default directory.
        """
        # Set global style for better visual appearance
        plt.style.use('fivethirtyeight')

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')

        # Plot histogram of stress ranges, weighted by cycle counts
        # Each bar height reflects the total number of cycles falling into
        # that stress range bin
        ax.hist(stress_ranges,
                bins=bins,
                weights=cycle_counts,
                edgecolor='black')

        # Label axes and title with appropriate units and section ID
        ax.set_xlabel("Stress Range [MPa]", fontsize=10, labelpad=10)
        ax.set_ylabel("Cycle Count", fontsize=10, labelpad=10)
        ax.set_title(f'Rainflow Cycle Histogram: Section {section_index}',
                     fontsize=10)

        # Configure axis ticks and grid
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=10,
                       length=5,
                       width=1)
        ax.grid(False)  # Turn off background grid for clarity

        # Set axis limits: auto-scaling X and Y based on data range
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        # Customize spines (borders of the plot area)
        # Hide top and right borders for a cleaner look
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Keep left and bottom borders, style them in black
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(0.8)

        # Determine where to save the plot
        # If no dir is provided, create a default directory
        if plot_dir is None:
            plot_dir = os.path.join(self.analysis_dir, "fatigue_by_task",
                                    self.output_id)
        plot_dir_rainflow = os.path.join(plot_dir, "sections_rainflow_cycles")
        os.makedirs(plot_dir_rainflow, exist_ok=True)
        plot_path = os.path.join(
            plot_dir_rainflow, f"section_{section_index}_rainflow_cycles.png")

        # Save the plot to disk with tight layout and transparent background
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        # Close the figure to release memory
        plt.close(fig)

    def plot_tower_rainflow_heatmap(self,
                                    heatmap_matrix: np.ndarray,
                                    bins: np.ndarray,
                                    plot_dir: Optional[str] = None) -> None:
        """
        Plots rainflow cycle distribution as a tower-section heatmap.

        Each row corresponds to a tower section, and each column represents a
          stress range bin.

        Args:
            heatmap_matrix: 2D array (sections × bins) with cycle counts per
              stress bin.
            bins: Array of bin edges for stress ranges [MPa].
            plot_dir: Optional file directory to save the plot. If not provided,
              uses default directory.
        """
        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
        ax.set_facecolor('white')

        # Define y-tick labels (one per tower section)
        z_ticks = list(range(1, heatmap_matrix.shape[0] + 1))

        # Display heatmap with logarithmic color normalization
        im = ax.imshow(heatmap_matrix,
                       norm=matplotlib.colors.LogNorm(),
                       aspect='auto',
                       interpolation='nearest',
                       cmap='jet',
                       extent=[bins[0], bins[-1], -0.5,
                               len(z_ticks) - 0.5],
                       origin='lower')

        # Axis labels
        ax.set_xlabel("Stress Range [MPa]", fontsize=10, labelpad=10)
        ax.set_ylabel("Tower Section", fontsize=10, labelpad=10)

        # Set y-ticks and labels to match section indices
        ax.set_yticks(np.arange(len(z_ticks)))
        ax.set_yticklabels(z_ticks)

        # Style ticks and remove grid
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=10,
                       length=5,
                       width=1)
        ax.grid(False)

        # Style all plot spines
        for spine in ['left', 'bottom', 'top', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(0.8)

        # Add vertical colorbar to represent log(cycle count)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Log(Cycle Count)', fontsize=10, labelpad=10)
        cbar.ax.tick_params(axis='both', which='major', labelsize=10)
        cbar.outline.set_visible(False)

        # Determine where to save the plot
        # If no dir is provided, create a default directory
        if plot_dir is None:
            plot_dir = os.path.join(self.analysis_dir, "fatigue_by_task",
                                    self.output_id)
            os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "tower_rainflow_heatmap.png")
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        # Close figure to free memory
        plt.close(fig)

    def plot_tower_damage_profile(self,
                                  damages: List[float],
                                  plot_dir: Optional[str] = None) -> None:
        """
        Plots the fatigue damage profile along the height of the tower.

        Each damage value corresponds to a specific tower section, and the plot
        shows how damage accumulates vertically along the structure.

        Args:
            damages: List of fatigue damage values (unitless), one per tower
              section.
            plot_dir: Optional file directory to save the plot. If not provided,
              uses default directory.
        """
        # Set global visual style
        plt.style.use('fivethirtyeight')

        # Create the figure and axes with white background
        fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')
        ax.set_facecolor('white')

        # Scatter plot: fatigue damage (x-axis) vs. height (y-axis)
        ax.scatter(damages,
                   self.tower.mean_z_sections,
                   color="tab:blue",
                   label="Section Damage")

        # Axis labels and formatting
        ax.set_xlabel("Fatigue Damage", fontsize=10, labelpad=10)
        ax.set_ylabel("Tower Height [m]", fontsize=10, labelpad=10)

        # Automatically set axis limits starting from 0
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        # Configure tick appearance
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=10,
                       length=5,
                       width=1)

        # Remove grid for visual clarity
        ax.grid(False)

        # Customize plot borders (spines)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(0.8)

        # Add legend for plotted data
        ax.legend(fontsize=9, frameon=False)

        # Determine where to save the plot
        # If no dir is provided, create a default directory
        if plot_dir is None:
            plot_dir = os.path.join(self.analysis_dir, "fatigue_by_task",
                                    self.output_id)
            os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "tower_damage_profile.png")

        # Save figure to file
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        # Close the figure to release memory
        plt.close(fig)


class TowerFatigueDatasetAnalysis(openfast_results.OpenFASTDatasetAnalysis):
    """
    Analyzes tower fatigue across multiple wind conditions.

    This class computes fatigue damage at each tower section for a set of
    OpenFAST simulations using a fixed tower geometry. It aggregates the results
    according to a generalized Weibull wind distribution model, allowing for the
    estimation of lifetime damage over the turbine's design life.

    Attributes:
        dataset_dir (str): Directory containing OpenFAST simulation results.
        tower (Tower): Tower instance defining the geometry.
        dataset_inputs_path (str): Path to the dataset inputs CSV file.
        moment_column (str): Column name for the bending moment (e.g., 'mfa').
        sn_intercepts_log10 (List[float]): Intercepts of the SN curve (log10).
        sn_slopes (List[float]): Slopes of the SN curve.
        thickness_reference (float): Reference wall thickness [mm] for SN
          correction.
        thickness_exponent (float): Exponent used in thickness correction.
        fatigue_life_threshold (float): Threshold cycles to switch SN curve
          slope.
        min_time (float): Start time [s] for fatigue data analysis (ignores
          initial transients).
        weibull_delta (float): Exponent to generalize the Weibull PDF (δ=1
          yields classic Weibull).
        weibull_alpha (float): Scale parameter of the Weibull distribution
          [m/s].
        weibull_beta (float): Shape parameter of the Weibull distribution.
        design_life (int): Total design life of the turbine in years, used for
          scaling accumulated fatigue.
        analysis_dir (str): Directory where CSVs and plots are saved.
        norm_bin_equal (bool): If True, assigns equal normalized probability to
          each bin.
    """

    def __init__(self,
                 dataset_dir: str,
                 tower: Tower,
                 dataset_inputs_path: str,
                 moment_column: str = 'mfa',
                 sn_intercepts_log10: Optional[List[float]] = None,
                 sn_slopes: Optional[List[float]] = None,
                 thickness_reference: float = 25.0,
                 thickness_exponent: float = 0.2,
                 fatigue_life_threshold: float = 1e7,
                 min_time: float = 400.0,
                 weibull_delta: float = 0.88,
                 weibull_alpha: float = 12.773,
                 weibull_beta: float = 2.345,
                 design_life: int = 25,
                 analysis_dir: Optional[str] = None,
                 norm_bin_equal: bool = False,
                 **kwargs):
        """
        Initializes the TowerFatigueDatasetAnalysis object.
        
        Args:
            dataset_dir (str): Directory containing OpenFAST simulation results.
            tower (Tower): Tower geometry definition.
            dataset_inputs_path (str): Path to the dataset inputs CSV file.
            moment_column (str): Column name for the bending moment (e.g.,
              'mfa').
            sn_intercepts_log10 (List[float]): Intercepts of the SN curve in
              base-10 logarithmic scale.
            sn_slopes (List[float]): Slopes of the SN curve.
            thickness_reference (float): Reference wall thickness in
              millimeters.
            thickness_exponent (float): Exponent for thickness correction.
            fatigue_life_threshold (float): Threshold above which second SN
              slope is used.
            min_time (float): Minimum simulation time to include (in seconds).
            weibull_delta (float): Exponent to generalize the Weibull PDF.
            weibull_alpha (float): Scale parameter of the Weibull distribution
                [m/s].
            weibull_beta (float): Shape parameter of the Weibull distribution.
            design_life (int): Total design life of the turbine in years, used
                for scaling accumulated fatigue.
            analysis_dir (Optional[str]): Directory where CSVs and plots are
                saved. If None, defaults to a subdirectory in dataset_dir. 
            norm_bin_equal (bool): If True, assigns equal normalized probability
                to each bin.   
        """
        super().__init__(dataset_dir=dataset_dir,
                         min_time=min_time,
                         analysis_dir=analysis_dir,
                         dataset_inputs_path=dataset_inputs_path,
                         **kwargs)
        self.analysis_dir = analysis_dir or os.path.join(
            self.dataset_dir, "fatigue_analysis")
        self.tower = tower
        self.dataset_inputs_path = dataset_inputs_path
        self.moment_column = moment_column
        self.sn_intercepts_log10 = sn_intercepts_log10 or [12.010, 15.350]
        self.sn_slopes = sn_slopes or [3, 5]
        self.thickness_reference = thickness_reference
        self.thickness_exponent = thickness_exponent
        self.fatigue_life_threshold = fatigue_life_threshold
        self.min_time = min_time
        self.weibull_delta = weibull_delta
        self.weibull_alpha = weibull_alpha
        self.weibull_beta = weibull_beta
        self.design_life = design_life
        self.norm_bin_equal = norm_bin_equal

    def get_total_simulation_cycles_for_lifetime(self) -> float:
        """
        Gets the number of simulation cycles for the turbine's full design life.

        Returns:
            float: Total number of simulation cycles required to span the full 
          design life in hours.
        """
        total_design_hours = 24 * 365 * self.design_life
        sim_duration = self.get_reference_simulation_duration()
        sim_sets_per_hour = 3600 / sim_duration

        return sim_sets_per_hour * total_design_hours

    def compute_damage_all_sections(
            self,
            save_csv: bool = False,
            save_plot: bool = False,
            save_per_section: bool = False) -> List[List[float]]:
        """
        Computes section-wise fatigue damage for each simulation in the dataset.

        For each simulation, this method initializes a TowerFatigueAnalysis
        instance using the shared tower geometry and fatigue parameters, and
        computes the fatigue damage at each tower section.

        Args:
            save_csv (bool): If True, saves a CSV with section-wise damage for 
              each case.
            save_plot (bool): If True, generates damage and rainflow plots for
              each case.

        Returns:
            List of lists containing fatigue damage per tower section, for each
              simulation case.
        """
        damages = []

        for output in self.outputs_analysis:
            # Create fatigue analysis instance for one simulation output
            os.makedirs(self.analysis_dir, exist_ok=True)
            fatigue_analysis = TowerFatigueAnalysis(
                output_dir=output.output_dir,
                tower=self.tower,
                moment_column=self.moment_column,
                sn_intercepts_log10=self.sn_intercepts_log10,
                sn_slopes=self.sn_slopes,
                thickness_reference=self.thickness_reference,
                thickness_exponent=self.thickness_exponent,
                fatigue_life_threshold=self.fatigue_life_threshold,
                min_time=self.min_time,
                analysis_dir=self.analysis_dir)

            # Compute fatigue damage for all tower sections in this case
            section_damages = fatigue_analysis.compute_damage_all_sections(
                save_csv=save_csv,
                save_plot=save_plot,
                save_per_section=save_per_section)

            damages.append(section_damages)

        return damages

    def compute_bts_wind_bin_probabilities_from_weibull(self) -> pd.DataFrame:
        """
        Computes wind bin probabilities from BTS-based wind speeds via Weibull.
        
        Returns:
            pd.DataFrame: A DataFrame with:
                - wind_speed_bin (pd.Interval): Wind speed bin interval.
                - wind_speed_mid (float): Midpoint of the bin.
                - task_id (List[str]): List of task IDs in the bin.
                - bin_prob (float): Integrated probability over the bin
                  according to the Weibull distribution.
        """
        # Group task IDs by wind speed bins based on BTS-derived wind values
        bins_df = self.group_ids_by_bts_wind_speed()

        # Generalized Weibull PDF
        weibull_pdf = openfast_wave.WaveDistributionAnalyzer(
        ).compute_exp_weibull_pdf

        # Compute integrated probability for each bin
        bins_df["bin_prob"] = bins_df["wind_speed_bin"].apply(
            lambda wind_bin: quad(weibull_pdf,
                                  wind_bin.left,
                                  wind_bin.right,
                                  args=(self.weibull_alpha, self.weibull_beta,
                                        self.weibull_delta))[0])

        return bins_df

    def expand_bts_wind_bin_probabilities_to_task_ids(self,
                                                      norm_bin_equal=False
                                                     ) -> pd.DataFrame:
        """
        Expands wind bin probabilities from BTS-based grouping to task IDs.

        Args:
            norm_bin_equal (bool, optional):
              If True, assigns equal normalized probability to each bin.
              If False (default), uses normalized Weibull-based probabilities.
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - task_id (str): Unique identifier of each simulation.
                - wind_speed_mid (float): Midpoint of the associated wind bin.
                - bin_prob (float): Non-normalized bin probability from Weibull.
                - norm_bin_prob (float): Normalized bin probability.
        """
        # Get wind bin probabilities grouped by BTS-derived wind speeds
        df = self.extract_wind_speed_and_seed_from_bts_filenames()[[
            "task_id", "wind_speed"
        ]]

        if norm_bin_equal:
            expanded_rows = []
            for task_id in df["task_id"]:
                expanded_rows.append({
                    "task_id": task_id,
                    "wind_speed_mid": "-",
                    "bin_prob": "-"
                })

            result_df = pd.DataFrame(expanded_rows)
            result_df["norm_bin_prob"] = 1 / len(self.output_ids)

            return result_df

        bins_df = self.compute_bts_wind_bin_probabilities_from_weibull()

        # Expand to one row per task ID
        expanded_rows = []
        for _, row in bins_df.iterrows():
            for task_id in row["task_id"]:
                expanded_rows.append({
                    "task_id": task_id,
                    "wind_speed_mid": row["wind_speed_mid"],
                    "bin_prob": row["bin_prob"]
                })

        result_df = pd.DataFrame(expanded_rows)

        # Normalize probabilities across all task IDs
        result_df["norm_bin_prob"] = result_df["bin_prob"] / result_df[
            "bin_prob"].sum()

        return result_df

    def group_section_damage_by_bts_wind(
            self,
            section_damage_df: pd.DataFrame,
            section_labels: list[str] = None) -> pd.DataFrame:
        """
        Groups damage values per tower section according to BTS-based wind bins.

        Args:
            section_damage_df (pd.DataFrame): DataFrame with columns:
                - 'wind_speed_mid': float, wind speed bin centers
                - 'section_*': scalar damage values for each tower section
            section_labels (list[str], optional): List of section column names
              to include. Defaults to all columns starting with 'section_'.

        Returns:
            pd.DataFrame: DataFrame grouped by wind_speed_mid with a list of
              damage values per section.
        """

        # Identify all section columns
        section_cols = [
            col for col in section_damage_df.columns
            if col.startswith("section_")
        ]

        if section_labels is None:
            section_labels = section_cols

        # Extract wind speed midpoint values and initialize grouped DataFrame
        wind_speeds = sorted(section_damage_df["wind_speed_mid"].unique())
        grouped_df = pd.DataFrame({"wind_speed_mid": wind_speeds})
        grouped_df.set_index("wind_speed_mid", inplace=True)

        # Compute grouped damage values per section
        for label in section_labels:
            grouped_df[label] = section_damage_df.groupby(
                "wind_speed_mid")[label].apply(list)

        grouped_df.reset_index(inplace=True)
        return grouped_df

    def _compute_weighted_damage_all_sections(
        self, damage_matrix: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Computes weighted damage across all tower sections.
        
        Args:
            damage_matrix (np.ndarray): Matrix of shape (n_cases, n_sections)
              with damage values per wind case.

        Returns:
            Tuple:
                - weighted_damage (np.ndarray): Array of shape (n_sections,)
                  with fatigue damage per section, weighted by wind speed
                  probability and design cycles.
                - tower_damage_summary_df (pd.DataFrame): Detailed DataFrame
                  with columns: ["task_id", "norm_bin_prob", "num_cycles",
                  "damage_weight", "section_1", ..., "section_n"]
                - tower_weighted_damage_summary_df (pd.DataFrame): Same as
                  above, but section columns are already multiplied by weight.
        """

        # Get probability per task ID and number of repetitions for lifetime
        wind_probs = self.expand_bts_wind_bin_probabilities_to_task_ids(
            self.norm_bin_equal)
        num_cycles = self.get_total_simulation_cycles_for_lifetime()

        # Add number of cycles and weight per task
        wind_probs["num_cycles"] = np.full(len(self.output_ids), num_cycles)
        wind_probs["damage_weight"] = wind_probs["num_cycles"] * wind_probs[
            "norm_bin_prob"]

        # Create raw damage summary DataFrame
        tower_damage_summary_df = wind_probs.copy()
        for i in range(damage_matrix.shape[1]):
            section_damage = damage_matrix[:, i]
            tower_damage_summary_df[f"section_{i + 1}"] = section_damage

        # Create weighted damage summary DataFrame
        tower_weighted_damage_summary_df = wind_probs.copy()
        for i in range(damage_matrix.shape[1]):
            section_damage = damage_matrix[:, i]
            weights = tower_damage_summary_df["damage_weight"]
            weighted_section_damage = section_damage * weights
            tower_weighted_damage_summary_df[
                f"section_{i + 1}"] = weighted_section_damage

        # Compute total weighted damage per section
        weighted_damage = np.sum(
            damage_matrix *
            tower_damage_summary_df["damage_weight"].values[:, np.newaxis],
            axis=0)

        return (weighted_damage, tower_damage_summary_df,
                tower_weighted_damage_summary_df)

    def compute_weighted_damage_all_sections(
            self,
            save_plot: bool = False,
            save_yaml: bool = False,
            save_csv: bool = False,
            save_per_case: bool = False,
            save_per_section: bool = False) -> np.ndarray:
        """
        Computes weighted fatigue damage for all tower sections.

        Aggregates section-wise fatigue results across multiple wind cases using
        a generalized Weibull distribution.

        Args:
            save_plot (bool): If True, generates a plot of the damage profile.
            save_yaml (bool): If True, saves fatigue parameters and results to
              a YAML file.
            save_csv (bool): If True, saves the damage profile to a CSV file.
            save_per_case (bool): If True, also saves CSVs and plots per
              simulation case.
            save_per_section (bool): If True, enables section-level saving in
              each case.

        Returns:
            np.ndarray: Weighted fatigue damage per tower section.
        """
        # Compute damage matrix (cases × sections)
        damage_matrix = np.array(
            self.compute_damage_all_sections(
                save_csv=save_csv if save_per_case else False,
                save_plot=save_plot if save_per_case else False,
                save_per_section=save_per_section))

        # Compute weighted fatigue damage
        (weighted_damage, tower_damage_df, weighted_damage_df
        ) = self._compute_weighted_damage_all_sections(damage_matrix)

        # Save CSV
        if save_csv:

            damage_dir = os.path.join(self.analysis_dir, "damage_summary")
            damage_profile_dir = os.path.join(self.analysis_dir,
                                              "damage_profile")
            damage_by_wind_dir = os.path.join(self.analysis_dir,
                                              "damage_by_windspeed")
            os.makedirs(damage_dir, exist_ok=True)
            os.makedirs(damage_profile_dir, exist_ok=True)
            os.makedirs(damage_by_wind_dir, exist_ok=True)

            # Save raw and weighted damage summaries
            tower_damage_df.to_csv(os.path.join(damage_dir,
                                                "tower_raw_damage_summary.csv"),
                                   index=False)
            weighted_damage_df.to_csv(os.path.join(
                damage_dir, "tower_weighted_damage_summary.csv"),
                                      index=False)

            # Save weighted damage profile
            damage_profile_df = pd.DataFrame({
                "section_id": range(1, self.tower.num_sections + 1),
                "mean z [m]": self.tower.mean_z_sections,
                "mean radius [m]": self.tower.mean_radius_sections,
                "thickness [m]": self.tower.thickness_sections,
                "damage": weighted_damage
            })
            damage_profile_df.to_csv(os.path.join(
                damage_profile_dir, "tower_weighted_damage_profile.csv"),
                                     index=False)

            # Group by wind conditions
            grouped_raw = self.group_section_damage_by_bts_wind(tower_damage_df)
            grouped_weighted = self.group_section_damage_by_bts_wind(
                weighted_damage_df)

            grouped_raw.to_csv(os.path.join(
                damage_by_wind_dir, "tower_raw_damage_by_bts_wind_summary.csv"),
                               index=False)
            grouped_weighted.to_csv(os.path.join(
                damage_by_wind_dir,
                "tower_weighted_damage_by_bts_wind_summary.csv"),
                                    index=False)

        if save_yaml:

            damage_profile_dir = os.path.join(self.analysis_dir,
                                              "damage_profile")
            path = os.path.join(damage_profile_dir, "tower_fatigue.yaml")

            def flow(seq: list) -> CommentedSeq:
                """
                Converts a list to a ruamel.yaml CommentedSeq with flow style.

                Args:
                    seq (list): Input sequence to be converted.

                Returns:
                    CommentedSeq: A CommentedSeq object with flow style enabled.
                """
                cs = CommentedSeq(seq)
                cs.fa.set_flow_style()
                return cs

            def format_sig(x: float, sig: int = 18) -> float:
                """
                Formats a number with a given number of significant digits.

                Args:
                    x (float): Number to be formatted.
                    sig (int, optional): Number of significant digits. Defaults
                      to 18.

                Returns:
                    float: The formatted number with the specified significant
                    digits.
                """
                return float(f"{x:.{sig}g}")

            def format_list(arr: Union[np.ndarray, list],
                            sig: int = 18) -> CommentedSeq:
                """
                Format numbers with given significant digits.

                Args:
                    arr (Union[np.ndarray, list]): Input sequence of numbers.
                    sig (int, optional): Number of significant digits. Defaults
                      to 18.

                Returns:
                    CommentedSeq: Sequence with formatted numbers, stored in
                      flow style.
                """
                arr = np.asarray(arr, dtype=float).tolist()
                return flow([format_sig(v, sig) for v in arr])

            fatigue_dict = {
                "fatigue": {
                    "m": float(np.mean(self.sn_slopes)),
                    "k": float(self.thickness_exponent),
                    "t_ref": float(self.thickness_reference),
                    "tower_ref": {
                        "grid":
                            format_list(self.tower.grid),
                        "outer_diameter":
                            format_list(self.tower.diameter_transitions),
                        "wall_thickness":
                            format_list(self.tower.thickness_sections),
                        "z":
                            format_list(self.tower.z_transitions),
                        "section_damage":
                            format_list(weighted_damage),
                    },
                }
            }

            yaml_ruamel = YAML()
            yaml_ruamel.width = 2000

            with open(path, "w", encoding="utf-8") as f:
                yaml_ruamel.dump(fatigue_dict, f)

        # Save plot
        if save_plot:

            # Plot total weighted damage profile over tower height
            self.plot_tower_weighted_damage_profile(weighted_damage)

            # Plot violin plots for selected sections: raw and weighted
            section_labels = ["section_1", "section_30"]
            selected_raw = grouped_raw[["wind_speed_mid"] + section_labels]
            selected_weighted = grouped_weighted[["wind_speed_mid"] +
                                                 section_labels]
            self.plot_tower_damage_violin_by_wind_speed_bins(
                section_damage_by_bts_wind=selected_raw,
                is_weighted_damage=False)
            self.plot_tower_damage_violin_by_wind_speed_bins(
                section_damage_by_bts_wind=selected_weighted,
                is_weighted_damage=True)

            # Plot violin plots for all sections: raw and weighted
            self.plot_tower_damage_violin_per_section_by_wind_speed_bins(
                section_damage_by_bts_wind=grouped_raw,
                is_weighted_damage=False)
            self.plot_tower_damage_violin_per_section_by_wind_speed_bins(
                section_damage_by_bts_wind=grouped_weighted,
                is_weighted_damage=True)

        return weighted_damage

    def plot_tower_weighted_damage_profile(self,
                                           damages: List[float],
                                           plot_dir: Optional[str] = None
                                          ) -> None:
        """
        Plots the fatigue damage profile along the height of the tower.

        Each damage value corresponds to a specific tower section, and the plot
        shows how damage accumulates vertically along the structure.

        Args:
            damages (List[float]): Fatigue damage values (unitless), one per
              tower section.
            plot_dir (str, optional): File directory to save the plot. If not
              provided, uses default directory.
        """
        # Set global visual style
        plt.style.use('fivethirtyeight')

        # Create the figure and axes with white background
        fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')
        ax.set_facecolor('white')

        # Scatter plot: fatigue damage (x-axis) vs. height (y-axis)
        ax.scatter(damages, self.tower.mean_z_sections, color="tab:blue", s=6)

        # Step plot: mean damage profile across sections
        ax.step(damages,
                self.tower.mean_z_sections,
                color='tab:blue',
                where="mid",
                label="Section Damage",
                linewidth=1.5,
                markersize=4)

        # Add vertical reference line at damage = 1 (fatigue failure limit)
        ax.axvline(x=1,
                   color='gray',
                   linestyle='--',
                   linewidth=1,
                   alpha=0.9,
                   label="Damage = 1 (Fatigue Limit)")

        # Axis labels and formatting
        ax.set_xlabel("Weighted Fatigue Damage", fontsize=10, labelpad=10)
        ax.set_ylabel("Tower Height [m]", fontsize=10, labelpad=10)

        # Automatically set axis limits starting from 0
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        # Configure tick appearance
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=10,
                       length=5,
                       width=1)

        # Remove grid for visual clarity
        ax.grid(False)

        # Customize plot borders (spines)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(0.8)

        # Add legend for plotted data
        ax.legend(fontsize=9, frameon=False)

        # Determine where to save the plot
        # If no dir is provided, create a default directory
        if plot_dir is None:
            plot_dir = os.path.join(self.analysis_dir, "damage_profile")
            os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "tower_weighted_damage_profile.png")

        # Save figure to file
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        # Close the figure to release memory
        plt.close(fig)

    def plot_tower_damage_violin_by_wind_speed_bins(
            self,
            section_damage_by_bts_wind: pd.DataFrame,
            is_weighted_damage: bool = False,
            plot_dir: Optional[str] = None,
            colors: list = None,
            y_label: str = "Tower Damage",
            suffix_plotname: str = None) -> None:
        """
        Plots violin plots of tower damage distributions by wind speed bins.

        Each row in `section_damage_by_bts_wind` corresponds to a wind speed
        bin, with the center of the bin in 'wind_speed_mid' and each
        'section_*' column containing a list of damage values (raw or weighted).

        Args:
            section_damage_by_bts_wind (pd.DataFrame): DataFrame with:
                - 'wind_speed_mid': float values (bin centers)
                - 'section_*': list of damage values per section
            is_weighted_damage (bool): If True, assumes values are already
              weighted by probability.
            plot_dir (Optional[str]): Output directory for the saved plot.
            colors (list): Colors for each section.
            y_label (str): Label for the y-axis.
            suffix_plotname (Optional[str]): Custom suffix for the plot
              filename. If None, it will be set based on `is_weighted_damage`.
        """

        # Set global visual style
        plt.style.use('fivethirtyeight')

        # Default sections and colors if not provided
        if colors is None:
            colors = ["midnightblue", "firebrick"]

        # Compute violin plot positions
        num_bins = len(section_damage_by_bts_wind)
        positions = [
            np.arange(num_bins) * 2.0 + offset for offset in [1.15, 1.85]
        ]

        # Create the figure and axes with white background
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        ax.set_facecolor('white')

        # Get section columns sorted numerically
        section_cols = sorted([
            col for col in section_damage_by_bts_wind.columns
            if col.startswith("section_")
        ],
                              key=lambda x: int(x.split("_")[1]))

        # Check if all values are single-element lists
        is_single_value_only = section_damage_by_bts_wind[section_cols[
            0]].apply(lambda x: isinstance(x, list) and len(x) == 1).all()

        # Iterate over each tower section
        for i, section in enumerate(section_cols):
            medians = []
            valid_pos = []

            # Loop through each wind speed bin
            for j, vals in enumerate(section_damage_by_bts_wind[section]):
                if not vals:
                    continue

                # Get the x-axis position for this section in this bin
                x = positions[i][j]

                # Plot the individual violin for this bin
                part = ax.violinplot([vals],
                                     positions=[x],
                                     showmedians=False,
                                     showextrema=False)

                # Customize the appearance of the violin body
                for pc in part['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.6)
                    pc.set_linewidth(0.8)

                # Compute boxplot statistics
                stats = boxplot_stats(vals)[0]

                # Plot median and box lines on top of the violin
                ax.scatter(x,
                           stats['med'],
                           color="white",
                           edgecolor='black',
                           zorder=3,
                           s=10)
                ax.vlines(x,
                          stats['q1'],
                          stats['q3'],
                          color='k',
                          linestyle='-',
                          linewidth=1.5,
                          zorder=2)
                ax.vlines(x,
                          stats['whislo'],
                          stats['whishi'],
                          color='k',
                          linestyle='-',
                          linewidth=0.8,
                          zorder=1)
                medians.append(stats['med'])
                valid_pos.append(x)

                # Axis labels and formatting
                ax.set_xticks((positions[0] + positions[1]) / 2)
                ax.set_xticklabels(section_damage_by_bts_wind["wind_speed_mid"])
                ax.set_xlabel("Mean Wind Speed (m/s) [Bin centers]",
                              fontsize=10,
                              labelpad=10)
                if is_weighted_damage:
                    ax.set_ylabel(f"Weighted {y_label}",
                                  fontsize=10,
                                  labelpad=10)
                else:
                    ax.set_ylabel(y_label, fontsize=10, labelpad=10)
                ax.yaxis.get_offset_text().set_fontsize(10)

                # Configure tick appearance
                ax.tick_params(axis='both',
                               which='major',
                               labelsize=10,
                               length=5,
                               width=1)

            # Connect medians if all bins contain single damage values
            if is_single_value_only:
                ax.plot(valid_pos,
                        medians,
                        color=colors[i],
                        linewidth=2.0,
                        marker='o')

        # Remove grid for visual clarity
        ax.grid(False)

        # Customize plot borders (spines)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        # Add legend for plotted data
        legend_labels = [
            label.replace("section_", "Section ") for label in section_cols
        ]
        legend_handles = [
            Patch(facecolor=colors[i],
                  edgecolor='black',
                  label=legend_labels[i],
                  alpha=0.6) for i in range(len(section_cols))
        ]
        ax.legend(handles=legend_handles, fontsize=9, frameon=False)

        # Determine where to save the plot
        # If no dir is provided, create a default directory
        if plot_dir is None:
            plot_dir = os.path.join(self.analysis_dir, "damage_by_windspeed")
            os.makedirs(plot_dir, exist_ok=True)

        if suffix_plotname is None:
            suffix = "weighted_damage" if is_weighted_damage else "raw_damage"
        else:
            suffix = (f"weighted_damage_{suffix_plotname}" if is_weighted_damage
                      else f"raw_damage_{suffix_plotname}")
        plot_path = os.path.join(plot_dir,
                                 f"tower_{suffix}_by_wind_speed_violin.png")

        # Save figure to file
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        # Close the figure to release memory
        plt.close(fig)

    def plot_tower_damage_violin_per_section_by_wind_speed_bins(
            self,
            section_damage_by_bts_wind,
            is_weighted_damage: bool = False,
            plot_dir: Optional[str] = None,
            n_cols: int = 6) -> None:
        """
        Creates violin plots showing the tower damage distribution per section.
        
        Each subplot corresponds to one tower section, and within each subplot,
        damage is grouped by wind speed.

        Args:
            section_damage_by_bts_wind (pd.DataFrame): DataFrame with columns:
                - 'wind_speed_mid': float, wind speed bin centers
                - 'section_*': list of damage values per section for each bin
            is_weighted_damage (bool): Whether the damage values are already
                weighted by probability.
            plot_dir (Optional[str]): Directory to save the plot. If None,
                defaults to self.analysis_dir.
            n_cols (int): Number of subplot columns in the figure layout.
        """

        # Set global visual style
        plt.style.use('fivethirtyeight')

        # Identify section columns to be plotted
        section_labels = [
            col for col in section_damage_by_bts_wind.columns
            if col.startswith("section_")
        ]
        n_sections = len(section_labels)

        # Compute violin plot positions
        num_bins = len(section_damage_by_bts_wind)
        positions = np.arange(num_bins)

        # Create the figure and axes with white background
        n_rows = math.ceil(n_sections / n_cols)
        fig, axs = plt.subplots(n_rows,
                                n_cols,
                                figsize=(70, 40),
                                facecolor='white',
                                sharey=False)
        plt.subplots_adjust(hspace=0.25, wspace=0.15)
        axs = axs.flatten()

        # Create one subplot per tower section
        for i, section in enumerate(section_labels):
            ax = axs[i]

            # Gather damage values for this section per wind speed
            section_damages = (section_damage_by_bts_wind[section]).tolist()

            # Check if all values are single-element lists
            is_single_value_only = (all(
                len(vals) == 1 for vals in section_damages))
            medians = []
            valid_pos = []

            for j, vals in enumerate(section_damages):
                if not vals:
                    continue

                # Get the x-axis position for this section in this bin
                x = positions[j]

                # Plot the individual violin for this bin
                parts = ax.violinplot([vals],
                                      positions=[x],
                                      showmedians=False,
                                      showextrema=False)

                # Customize violin appearance: color and transparency
                for pc in parts['bodies']:
                    pc.set_facecolor("tab:blue")
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.6)
                    pc.set_alpha(0.8)

                # Plot median and box lines on top of the violin
                stats = boxplot_stats(vals)[0]
                ax.scatter(x,
                           stats['med'],
                           color='white',
                           edgecolor='black',
                           zorder=3,
                           s=10)
                ax.vlines(x,
                          stats['q1'],
                          stats['q3'],
                          color='k',
                          linestyle='-',
                          linewidth=1.5,
                          zorder=2)
                ax.vlines(x,
                          stats['whislo'],
                          stats['whishi'],
                          color='k',
                          linestyle='-',
                          linewidth=0.8,
                          zorder=1)
                medians.append(stats['med'])
                valid_pos.append(x)

            # Connect medians if all bins contain single damage values
            if is_single_value_only:
                ax.plot(valid_pos,
                        medians,
                        color="tab:blue",
                        linewidth=2.0,
                        marker='o')

            # Axis labels and formatting
            ax.set_xticks(positions)
            ax.set_xticklabels(section_damage_by_bts_wind["wind_speed_mid"])
            ax.set_title(section.replace("_", " ").title(), fontsize=12)
            if i // n_cols == n_rows - 1:
                ax.set_xlabel("Mean Wind Speed (m/s) [Bin centers]",
                              fontsize=10,
                              labelpad=10)
            if i % n_cols == 0:
                if is_weighted_damage:
                    ax.set_ylabel("Weighted Tower Damage",
                                  fontsize=10,
                                  labelpad=10)
                else:
                    ax.set_ylabel("Tower Damage", fontsize=10, labelpad=10)
            ax.yaxis.get_offset_text().set_fontsize(10)

            # Configure tick appearance
            ax.tick_params(axis='both',
                           which='major',
                           labelsize=10,
                           length=5,
                           width=1)

            # Remove grid for visual clarity
            ax.grid(False)

            # Customize plot borders (spines)
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)

        # Determine where to save the plot
        # If no dir is provided, create a default directory
        suffix = "weighted" if is_weighted_damage else "raw"
        if plot_dir is None:
            plot_dir = os.path.join(self.analysis_dir, "damage_by_windspeed")
            os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir, f"tower_{suffix}_damage_by_section_violin.png")

        # Save figure to file
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        # Close the figure to release memory
        plt.close(fig)


class TowerFatigueComparisonAnalysis(TowerFatigueDatasetAnalysis):
    """
    Compare fatigue analyses.

    Attributes:
        comparison_dir (str): Directory to store comparison-related outputs.
        fatigue_analysis_dirs (List[str]): List of directories with fatigue
          results to compare.
        ref_idx (int): Index of the reference analysis in fatigue_analysis_dirs.
        raw_damage_summary_df_ref (pd.DataFrame): DataFrame with raw damage
          summary for the reference analysis.
        weighted_damage_summary_df_ref (pd.DataFrame): DataFrame with weighted
          damage summary for the reference analysis.
        damage_profile_df_ref (pd.DataFrame): DataFrame with damage profile for
          the reference analysis.
        raw_damage_summary_df_news (List[pd.DataFrame]): List of DataFrames with
          raw damage summaries for new analyses.
        weighted_damage_summary_df_news (List[pd.DataFrame]): List of DataFrames
          with weighted damage summaries for new analyses.
        damage_profile_df_news (List[pd.DataFrame]): List of DataFrames with
          damage profiles for new analyses.
        colors (List[str]): List of colors for plotting different analyses.
    """

    def __init__(self,
                 comparison_dir: str,
                 fatigue_analysis_dirs: List[str],
                 ref_idx: int,
                 colors: Optional[List[str]] = None,
                 **kwargs):
        """
        Initializes the TowerFatigueComparisonAnalysis object.

        Args:
            comparison_dir (str): Directory to store comparison-related
              outputs.
            fatigue_analysis_dirs (List[str]): List of directories with fatigue
              results to compare.
            ref_idx (int): Index of the reference analysis in 
              fatigue_analysis_dirs.
            colors (Optional[List[str]]): List of colors for plotting different
              analyses. If None, default colors are used.
        """
        self.ref_idx = ref_idx
        self.fatigue_analysis_dirs = fatigue_analysis_dirs
        self.comparison_dir = comparison_dir
        self.output_dir = os.path.join(self.comparison_dir,
                                       'fatigue_comparison_results')
        os.makedirs(self.output_dir, exist_ok=True)

        super().__init__(dataset_dir=self.comparison_dir,
                         tower=None,
                         dataset_inputs_path=None,
                         **kwargs)

        # Load data from analyses
        self._get_data()

        # Validate matching properties
        self._validate_matching_z_axis()

        # Set colors for plotting
        self.colors = colors if colors is not None else [
            "slategrey",
            'midnightblue',
            'firebrick',
            "darkgreen",
            "orange",
        ]

    def _get_data(self) -> None:
        """
        Loads fatigue analysis data from specified directories.
        """
        self.raw_damage_summary_df_news = []
        self.weighted_damage_summary_df_news = []
        self.damage_profile_df_news = []

        for i, fatigue_analysis_dir in enumerate(self.fatigue_analysis_dirs):
            raw_damage_summary_df = pd.read_csv(
                os.path.join(fatigue_analysis_dir, "damage_summary",
                             "tower_raw_damage_summary.csv"))
            weighted_damage_summary_df = pd.read_csv(
                os.path.join(fatigue_analysis_dir, "damage_summary",
                             "tower_weighted_damage_summary.csv"))

            damage_profile_df = pd.read_csv(
                os.path.join(fatigue_analysis_dir, "damage_profile",
                             "tower_weighted_damage_profile.csv"))
            if i == self.ref_idx:
                self.raw_damage_summary_df_ref = raw_damage_summary_df
                self.weighted_damage_summary_df_ref = weighted_damage_summary_df
                self.damage_profile_df_ref = damage_profile_df

            else:
                self.raw_damage_summary_df_news.append(raw_damage_summary_df)
                self.weighted_damage_summary_df_news.append(
                    weighted_damage_summary_df)
                self.damage_profile_df_news.append(damage_profile_df)

    def _validate_matching_z_axis(self) -> None:
        """
        Validates that the 'mean z [m]' values are consistent across datasets.

        Raises:
            ValueError: If the 'mean z [m]' values are not equal across
              datasets.
        """
        # Extrair os vetores z
        z_news = [df["mean z [m]"].values for df in self.damage_profile_df_news]

        # Usar o índice de referência
        z_ref = self.damage_profile_df_ref["mean z [m]"].values

        for z in z_news:
            if not np.allclose(z_ref, z, rtol=1e-6, atol=1e-9, equal_nan=True):
                raise ValueError(
                    "'mean z [m]' values do not match across datasets.")

    def compute_relative_difference_section_damage(
            self, section_damage_df_ref: pd.DataFrame,
            section_damage_df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Computes relative difference in section-wise fatigue damage.

        Args:
            section_damage_df_ref (pd.DataFrame): Reference DataFrame with
              section-wise damage.
            section_damage_df_new (pd.DataFrame): New DataFrame with
              section-wise damage.

        Returns:
            pd.DataFrame: DataFrame with relative differences (%) in damage per
              section, indexed by task ID and wind speed.
        """
        # Make local copies to avoid modifying original DataFrames
        section_damage_df_ref = section_damage_df_ref.copy()
        section_damage_df_new = section_damage_df_new.copy()

        # Extract task ID base from full task_id string
        section_damage_df_ref['task_id_base'] = section_damage_df_ref[
            'task_id'].str.extract(r'^(\d+)', expand=False)
        section_damage_df_new['task_id_base'] = section_damage_df_new[
            'task_id'].str.extract(r'^(\d+)', expand=False)

        # Define merge keys for matching simulation conditions
        key_cols = [
            'task_id_base', 'wind_speed_mid', 'bin_prob', 'norm_bin_prob',
            'num_cycles', "damage_weight"
        ]

        # Select section damage columns (e.g., section_1, section_2, ...)
        section_cols = [
            col for col in section_damage_df_ref.columns
            if col.startswith('section_')
        ]

        # Merge the reference and new damage data
        merged = pd.merge(section_damage_df_ref,
                          section_damage_df_new,
                          on=key_cols,
                          suffixes=('_ref', '_new'))

        # Compute relative difference per section
        section_diff_frames = []
        for section in section_cols:
            val_ref = merged[f"{section}_ref"].replace(0, np.nan)
            val_new = merged[f"{section}_new"]
            diff_percent = (val_new - val_ref) / val_ref * 100

            # Store result with identifiers
            temp = merged[['task_id_base', 'wind_speed_mid']].copy()
            temp[section] = diff_percent
            section_diff_frames.append(temp)

        # Combine all per-section differences
        rel_diff_df = pd.concat(section_diff_frames, axis=1)

        # Remove duplicated identifier columns (from concatenation)
        rel_diff_df = rel_diff_df.loc[:, ~rel_diff_df.columns.duplicated()]

        return rel_diff_df

    def compute_relative_difference_damage_profile(self,
                                                   damage_bound: float = None
                                                  ) -> list[pd.DataFrame]:
        """
        Computes relative difference in damage profile.

        Returns:
            list[pd.DataFrame]: List of DataFrames with relative differences (%)
              in mean radius, thickness, and damage along tower height.
            bound_as_ref (bool): If True, uses the reference profile's.
        """
        # Extract z values (same for all datasets)
        z_vals = self.damage_profile_df_ref["mean z [m]"]

        if damage_bound is None:
            # Relative difference in mean radius
            radius_ref = self.damage_profile_df_ref["mean radius [m]"]

        rel_diff_df_list = []
        # Compute relative differences for each new dataset
        for damage_profile_df_new in self.damage_profile_df_news:

            # Relative difference in mean radius
            radius_new = damage_profile_df_new["mean radius [m]"]
            thickness_new = damage_profile_df_new["thickness [m]"] * 1000
            dmg_new = damage_profile_df_new["damage"]

            if damage_bound is None:
                # Relative difference in mean radius
                rel_radius = 100 * (radius_new -
                                    radius_ref) / radius_ref.replace(0, np.nan)

                # Relative difference in thickness
                thickness_ref = self.damage_profile_df_ref[
                    "thickness [m]"] * 1000
                rel_thickness = 100 * (thickness_new -
                                       thickness_ref) / thickness_ref.replace(
                                           0, np.nan)

                # Relative difference in fatigue damage
                dmg_ref = self.damage_profile_df_ref["damage"]
                diff_damage = dmg_new - dmg_ref
                rel_damage = 100 * (diff_damage) / dmg_ref.replace(0, np.nan)

                rel_diff_df = pd.DataFrame({
                    "mean z [m]": z_vals,
                    "mean radius [%]": rel_radius,
                    "thickness [%]": rel_thickness,
                    "damage ref": dmg_ref,
                    "damage new": dmg_new,
                    "damage diff": diff_damage,
                    "damage [%]": rel_damage
                })
                rel_diff_df_list.append(rel_diff_df)

            else:
                dmg_ref = pd.Series(damage_bound,
                                    index=dmg_new.index,
                                    name="damage")

                diff_damage = dmg_new - dmg_ref
                rel_damage = 100 * (diff_damage) / dmg_ref

                # Combine all relative values into a single DataFrame
                rel_diff_df = pd.DataFrame({
                    "mean z [m]": z_vals,
                    "mean radius [m]": radius_new,
                    "thickness [m]": thickness_new,
                    "damage ref": dmg_ref,
                    "damage new": dmg_new,
                    "damage diff": diff_damage,
                    "damage [%]": rel_damage
                })
                rel_diff_df_list.append(rel_diff_df)

        return rel_diff_df_list

    def compare_fatigue_analysis(self,
                                 plot_labels: List[str] = None,
                                 plot_damage_bounds: List[float] = None,
                                 save_svg: bool = False) -> None:
        """
        Compare fatigue damages between the reference and new analyses.

        Calculates relative differences in damage profiles and section-wise
        damages, and generates plots to visualize the comparisons.

        Args:
            plot_labels (List[str], optional): Labels for each analysis in
              plots.
            plot_damage_bounds (List[float], optional): Y-axis bounds for damage
              profile plots.
            save_svg (bool): If True, saves plots in SVG format.
        """
        # Create output directories
        damage_dir = os.path.join(self.output_dir, "damage_summary")
        damage_profile_dir = os.path.join(self.output_dir, "damage_profile")
        damage_profile_bound_dir = os.path.join(damage_profile_dir,
                                                "bounds_as_ref")
        damage_profile_ref_dir = os.path.join(damage_profile_dir, "ref_as_ref")
        damage_by_wind_dir = os.path.join(self.output_dir,
                                          "damage_by_windspeed")
        os.makedirs(damage_dir, exist_ok=True)
        os.makedirs(damage_profile_dir, exist_ok=True)
        os.makedirs(damage_profile_bound_dir, exist_ok=True)
        os.makedirs(damage_profile_ref_dir, exist_ok=True)
        os.makedirs(damage_by_wind_dir, exist_ok=True)

        # Set plot labels
        plot_label_ref = plot_labels[
            self.ref_idx] if plot_labels else "Ref Tower"
        plot_labels_news = [
            plot_labels[i] if plot_labels else f"New Tower {i+1}"
            for i in range(len(self.fatigue_analysis_dirs))
            if i != self.ref_idx
        ]

        summary_ref_as_baseline = {}

        # Compute and save relative difference in damage profile
        damage_profile_rel_diff_pd_list = (
            self.compute_relative_difference_damage_profile())
        for damage_profile_rel_diff_pd, plot_label_new in zip(
                damage_profile_rel_diff_pd_list, plot_labels_news):
            case = (f"{plot_label_new.replace(' ', '').lower()}_"
                    f"vs_{plot_label_ref.replace(' ', '').lower()}")
            damage_profile_rel_diff_pd.to_csv(os.path.join(
                damage_profile_ref_dir,
                f"tower_weighted_damage_rel_diff_profile_{case}.csv"),
                                              index=False)
            case_with_ref_as_ref_summary = self.extract_summary_from_pf(
                damage_profile_rel_diff_pd)
            summary_ref_as_baseline[case] = case_with_ref_as_ref_summary

        summary_bound_as_baseline = {}

        for damage_bound in plot_damage_bounds:
            damage_profile_rel_diff_pd_list_bound = (
                self.compute_relative_difference_damage_profile(
                    damage_bound=damage_bound))

            for damage_profile_rel_diff_pd, plot_label_new in zip(
                    damage_profile_rel_diff_pd_list_bound, plot_labels_news):
                case = (f"{plot_label_new.replace(' ', '').lower()}_"
                        f"vs_{str(damage_bound).replace('.', '').lower()}")
                damage_profile_rel_diff_pd.to_csv(os.path.join(
                    damage_profile_bound_dir,
                    f"tower_weighted_damage_rel_diff_profile_{case}.csv"),
                                                  index=False)

                case_with_bound_as_ref_summary = self.extract_summary_from_pf(
                    damage_profile_rel_diff_pd)
                summary_bound_as_baseline[case] = case_with_bound_as_ref_summary

        # join both summaries
        full_summary = {
            "ref_as_baseline": summary_ref_as_baseline,
            "bound_as_baseline": summary_bound_as_baseline
        }
        # save to json
        with open(os.path.join(damage_profile_dir,
                               "fatigue_profile_comparison_summary.json"),
                  'w',
                  encoding='utf-8') as f:
            json.dump(full_summary, f, indent=4)

        # Plot tower weighted damage profiles
        self.plot_tower_weighted_damage_profile_comparison(
            plot_dir=damage_profile_dir,
            labels=plot_labels,
            plot_geometry=False,
            bounds=plot_damage_bounds,
            save_svg=save_svg)
        self.plot_tower_weighted_damage_profile_comparison(
            plot_dir=damage_profile_dir,
            labels=plot_labels,
            plot_geometry=False,
            plot_ref=False,
            bounds=plot_damage_bounds,
            save_svg=save_svg)
        self.plot_tower_weighted_damage_profile_comparison(
            plot_dir=damage_profile_dir, labels=plot_labels, plot_geometry=True)
        self.plot_tower_weighted_damage_profile_comparison(
            plot_dir=damage_profile_dir,
            rel_diff_df_list=damage_profile_rel_diff_pd_list,
            labels=plot_labels,
            plot_geometry=True)

        # Compute relative differences by wind condition (per section)
        for (raw_damage_summary_df_new, weighted_damage_summary_df_new,
             plot_label_new) in zip(self.raw_damage_summary_df_news,
                                    self.weighted_damage_summary_df_news,
                                    plot_labels_news):
            case = (f"{plot_label_new.replace(' ', '').lower()}_"
                    f"vs_{plot_label_ref.replace(' ', '').lower()}")

            damage_by_wind_dir_new = os.path.join(
                damage_by_wind_dir,
                plot_label_new.replace(' ', '').lower())
            os.makedirs(damage_by_wind_dir_new, exist_ok=True)

            damage_dir_new = os.path.join(
                damage_dir,
                plot_label_new.replace(' ', '').lower())
            os.makedirs(damage_dir_new, exist_ok=True)

            raw_rel_err_damage_summary = (
                self.compute_relative_difference_section_damage(
                    self.raw_damage_summary_df_ref, raw_damage_summary_df_new))
            weighted_rel_err_damage_summary = (
                self.compute_relative_difference_section_damage(
                    self.weighted_damage_summary_df_ref,
                    weighted_damage_summary_df_new))
            raw_rel_err_damage_summary.to_csv(os.path.join(
                damage_dir_new,
                f"tower_raw_damage_rel_diff_by_bts_wind_summary_{case}.csv"),
                                              index=False)
            weighted_rel_err_damage_summary.to_csv(os.path.join(
                damage_dir_new,
                f"tower_weighted_damage_rel_diff_by_bts_wind_summary_{case}.csv"
            ),
                                                   index=False)

            # Group by wind conditions
            grouped_raw = self.group_section_damage_by_bts_wind(
                raw_rel_err_damage_summary)
            grouped_weighted = self.group_section_damage_by_bts_wind(
                weighted_rel_err_damage_summary)
            grouped_raw.to_csv(os.path.join(
                damage_by_wind_dir_new,
                f"tower_raw_damage_rel_diff_by_bts_wind_summary_{case}.csv"),
                               index=False)
            grouped_weighted.to_csv(os.path.join(
                damage_by_wind_dir_new,
                f"tower_weighted_damage_rel_diff_by_bts_wind_summary_{case}.csv"
            ),
                                    index=False)

            # Plot violin plots for selected sections: raw and weighted
            section_labels = ["section_1", "section_30"]
            selected_raw = grouped_raw[["wind_speed_mid"] + section_labels]
            selected_weighted = grouped_weighted[["wind_speed_mid"] +
                                                 section_labels]
            self.plot_tower_damage_violin_by_wind_speed_bins(
                plot_dir=damage_by_wind_dir_new,
                section_damage_by_bts_wind=selected_raw,
                is_weighted_damage=False,
                y_label="Tower Damage Rel. Diff.",
                suffix_plotname="rel_diff")
            self.plot_tower_damage_violin_by_wind_speed_bins(
                plot_dir=damage_by_wind_dir_new,
                section_damage_by_bts_wind=selected_weighted,
                is_weighted_damage=True,
                y_label="Tower Damage Rel. Diff.",
                suffix_plotname="rel_diff")

    def extract_summary_from_pf(self, data_pd: pd) -> dict:
        """
        Extracts a summary DataFrame from the provided DataFrame.

        Args:
            data_pd (pd.DataFrame): Input DataFrame containing fatigue analysis
              results.

        Returns:
            pd.DataFrame: Summary DataFrame with selected columns.
        """

        summary_cols = ["damage ref", "damage new", "damage diff", "damage [%]"]

        summary_df = data_pd[summary_cols].copy()

        dict_summary = {}
        #sort by damage diff

        dict_summary_damagediff = {}

        min_damage_diff = summary_df['damage diff'].min()
        min_damade_diff_idx = summary_df['damage diff'].idxmin()
        dict_summary_damage_diff_min = {
            "value": min_damage_diff,
            "index": int(min_damade_diff_idx)
        }

        max_damage_diff = summary_df['damage diff'].max()
        max_damade_diff_idx = summary_df['damage diff'].idxmax()
        mean_damage_diff = summary_df['damage diff'].mean()

        top_damage_diff = summary_df['damage diff'].iloc[-1]
        bottom_damage_diff = summary_df['damage diff'].iloc[0]
        dict_summary_damage_diff_max = {
            "value": max_damage_diff,
            "index": int(max_damade_diff_idx)
        }

        dict_summary_damagediff['min'] = dict_summary_damage_diff_min
        dict_summary_damagediff['max'] = dict_summary_damage_diff_max
        dict_summary_damagediff['mean'] = mean_damage_diff
        dict_summary_damagediff['bottom'] = bottom_damage_diff
        dict_summary_damagediff['top'] = top_damage_diff
        dict_summary_damagediff['array'] = summary_df['damage diff'].tolist()

        dict_summary['damage_diff_summary'] = dict_summary_damagediff

        dict_summary_damage_rel = {}
        min_damage_rel = summary_df['damage [%]'].min()
        min_damade_rel_idx = summary_df['damage [%]'].idxmin()
        dict_summary_damage_rel_min = {
            "value": min_damage_rel,
            "index": int(min_damade_rel_idx)
        }

        max_damage_rel = summary_df['damage [%]'].max()
        max_damade_rel_idx = summary_df['damage [%]'].idxmax()
        mean_damage_rel = summary_df['damage [%]'].mean()
        top_damage_rel = summary_df['damage [%]'].iloc[-1]
        bottom_damage_rel = summary_df['damage [%]'].iloc[0]
        dict_summary_damage_rel_max = {
            "value": max_damage_rel,
            "index": int(max_damade_rel_idx)
        }

        dict_summary_damage_rel['min'] = dict_summary_damage_rel_min
        dict_summary_damage_rel['max'] = dict_summary_damage_rel_max
        dict_summary_damage_rel['mean'] = mean_damage_rel
        dict_summary_damage_rel['bottom'] = bottom_damage_rel
        dict_summary_damage_rel['top'] = top_damage_rel
        dict_summary_damage_rel['array'] = summary_df['damage [%]'].tolist()

        dict_summary['damage_rel_summary'] = dict_summary_damage_rel

        top_damage_ref = summary_df['damage ref'].iloc[-1]
        bottom_damage_ref = summary_df['damage ref'].iloc[0]

        dict_summary_damage_ref = {}
        dict_summary_damage_ref['bottom'] = bottom_damage_ref
        dict_summary_damage_ref['top'] = top_damage_ref
        dict_summary['damage_ref_summary'] = dict_summary_damage_ref

        top_damage_new = summary_df['damage new'].iloc[-1]
        bottom_damage_new = summary_df['damage new'].iloc[0]
        dict_summary_damage_new = {}
        dict_summary_damage_new['bottom'] = bottom_damage_new
        dict_summary_damage_new['top'] = top_damage_new
        dict_summary['damage_new_summary'] = dict_summary_damage_new

        # round 2 decimals for all float values in the dict summary
        def round_floats(obj):
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [round_floats(i) for i in obj]
            if isinstance(obj, float):
                return round(obj, 3)
            return obj

        dict_summary = round_floats(dict_summary)

        return dict_summary

    def plot_tower_weighted_damage_profile_comparison(
            self,
            plot_dir: str,
            rel_diff_df_list: Optional[List[pd.DataFrame]] = None,
            labels: Optional[List[str]] = None,
            plot_geometry=False,
            plot_ref=True,
            bounds=None,
            save_svg=False) -> None:
        """
        Plot side-by-side comparison of tower geometry and damage profiles.

        Args:
            plot_dir (str): Directory to save the plot.
            rel_diff_df_list (List[pd.DataFrame], optional): List of DataFrames
              with relative differences (%) in mean radius, thickness, and
              damage along tower height. If provided, plots relative differences
              instead of absolute values.
            labels (List[str], optional): Labels for each analysis in plots.
            plot_geometry (bool): If True, includes geometry profiles (diameter
                and thickness) in the plot.
            plot_ref (bool): If True, includes the reference tower in the plots.
            bounds (List[float], optional): Vertical lines to indicate bounds
                on the damage profile plot.
            save_svg (bool): If True, saves plots in SVG format.    
        """

        # Set plot labels
        plot_label_ref = labels[self.ref_idx] if labels else "Ref Tower"
        plot_labels_news = [
            labels[i] if labels else f"New Tower {i+1}"
            for i in range(len(self.fatigue_analysis_dirs))
            if i != self.ref_idx
        ]

        # Extract z values (same for all datasets)
        z_vals = self.damage_profile_df_ref["mean z [m]"]

        # Determine number of subplots
        subplots = 3 if plot_geometry else 1

        # Set global visual style
        plt.style.use("fivethirtyeight")

        # Create the figure and axes with white background
        fig, axs = plt.subplots(1,
                                subplots,
                                figsize=(4 * subplots, 4),
                                facecolor='white')
        if subplots == 1:
            ax3 = axs
        else:
            ax1, ax2, ax3 = axs

        # LEFT: Radius / Diameter profile
        if plot_geometry:
            if rel_diff_df_list is not None:
                for i, rel_diff_df in enumerate(rel_diff_df_list):
                    ax1.step(rel_diff_df["mean radius [%]"],
                             z_vals,
                             where="mid",
                             label=plot_labels_news[i],
                             color=self.colors[i + 1],
                             linewidth=1.5)
                ax1.axvline(x=0.0,
                            linestyle='--',
                            color='gray',
                            linewidth=1,
                            alpha=0.8)
                ax1.set_xlabel("Rel. Diff. in Diameter (%)",
                               fontsize=10,
                               labelpad=10)
            else:
                if plot_ref:
                    ax1.step(self.damage_profile_df_ref["mean radius [m]"],
                             z_vals,
                             label=plot_label_ref,
                             color=self.colors[0],
                             where="mid",
                             linewidth=1.5)
                for i, damage_profile_df_new in enumerate(
                        self.damage_profile_df_news):
                    ax1.step(damage_profile_df_new["mean radius [m]"],
                             z_vals,
                             label=plot_labels_news[i],
                             color=self.colors[i + 1],
                             where="mid",
                             linewidth=1.5)
                ax1.set_xlabel("Mean Diameter (m)", fontsize=10, labelpad=10)
            ax1.set_title("Tower Mean Diameter Profile", fontsize=12)

            # MIDDLE: Thickness profile
            if rel_diff_df_list is not None:
                for i, rel_diff_df in enumerate(rel_diff_df_list):
                    ax2.step(rel_diff_df["thickness [%]"],
                             z_vals,
                             where="mid",
                             label=plot_labels_news[i],
                             color=self.colors[i + 1],
                             linewidth=1.5)
                ax2.axvline(x=0.0,
                            linestyle='--',
                            color='gray',
                            linewidth=1,
                            alpha=0.9)
                ax2.set_xlabel("Rel. Diff. in Thickness (%)",
                               fontsize=10,
                               labelpad=10)
            else:
                if plot_ref:
                    ax2.step(self.damage_profile_df_ref["thickness [m]"] * 1000,
                             z_vals,
                             where="mid",
                             label=plot_label_ref,
                             color=self.colors[0],
                             linewidth=1.5)
                for i, damage_profile_df_new in enumerate(
                        self.damage_profile_df_news):
                    ax2.step(damage_profile_df_new["thickness [m]"] * 1000,
                             z_vals,
                             where="mid",
                             label=plot_labels_news[i],
                             color=self.colors[i + 1],
                             linewidth=1.5)
                ax2.set_xlabel("Wall Thickness (mm)", fontsize=10, labelpad=10)
            ax2.set_title("Wall Thickness Profile", fontsize=12)

        # RIGHT: Damage profile
        if rel_diff_df_list is not None:
            for i, rel_diff_df in enumerate(rel_diff_df_list):
                rel_dmg = rel_diff_df["damage [%]"]
                ax3.step(rel_dmg,
                         z_vals,
                         where="mid",
                         label=plot_labels_news[i],
                         color=self.colors[i + 1],
                         linewidth=1.5)
            ax3.axvline(x=0.0,
                        linestyle='--',
                        color='gray',
                        linewidth=1,
                        alpha=0.9)
            ax3.set_xlabel("Rel. Diff. in Damage (%)", fontsize=10, labelpad=10)
            ax3.set_title("Relative Diff.: Fatigue Damage", fontsize=12)

            if bounds is not None:
                for bound in bounds:
                    ax3.axvline(x=bound,
                                color="gray",
                                linestyle="--",
                                linewidth=0.9)

                    ax3.text(bound,
                             min(z_vals),
                             f"Bound ({(bound):.2f})",
                             fontsize=7,
                             color="gray",
                             va="bottom",
                             ha="right",
                             rotation=90)

        else:
            if plot_ref:
                ax3.step(self.damage_profile_df_ref["damage"],
                         z_vals,
                         where="mid",
                         label=plot_label_ref,
                         color=self.colors[0],
                         linewidth=1.5)
            for i, damage_profile_df_new in enumerate(
                    self.damage_profile_df_news):
                ax3.step(damage_profile_df_new["damage"],
                         z_vals,
                         where="mid",
                         label=plot_labels_news[i],
                         color=self.colors[i + 1],
                         linewidth=1.5)
            if bounds is not None:
                for bound in bounds:
                    ax3.axvline(x=bound,
                                color="gray",
                                linestyle="--",
                                linewidth=0.9)

                    ax3.text(bound,
                             min(z_vals),
                             f"Bound ({(bound):.2f})",
                             fontsize=7,
                             color="gray",
                             va="bottom",
                             ha="right",
                             rotation=90)

            ax3.set_xlabel("Damage", fontsize=10, labelpad=10)
        ax3.set_title("Damage Profile", fontsize=12)

        # Legend
        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   loc="lower center",
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.125),
                   fontsize=9,
                   frameon=False)

        axs = np.atleast_1d(axs)
        for ax in axs:
            # Hide spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(1)
                ax.spines[spine].set_color("black")

            # Configure tick appearance
            ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10)
            ax.grid(True, linestyle="-", alpha=0.5)
        axs[0].set_ylabel("Tower Height (m)", fontsize=10, labelpad=10)

        # Determine where to save the plot
        # If no dir is provided, create a default directory
        if plot_dir is None:
            plot_dir = os.path.join(self.comparison_dir, "damage_profile")
            os.makedirs(plot_dir, exist_ok=True)
        suffix = "_rel_diff_" if rel_diff_df_list is not None else "_"
        if plot_geometry:
            suffix2 = "_with_geometry"
        else:
            suffix2 = ""

        if not plot_ref:
            suffix2 += "_without_ref"

        if bounds is not None:
            suffix2 += "_with_bounds"
            for b in bounds:
                suffix2 += f"_{b:.2f}"

        plot_path = os.path.join(
            plot_dir, f"tower_weighted_damage{suffix}profile{suffix2}.png")

        # Save figure to file
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)

        if save_svg:
            plot_path_svg = plot_path.replace(".png", ".svg")
            plt.savefig(plot_path_svg,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)

        # Close the figure to release memory
        plt.close(fig)
