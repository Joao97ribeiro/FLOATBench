# pylint: skip-file
# Released post-processing pipeline of the FLOATBench labels, kept verbatim
# (only the colour import points to floatbench.colors).
# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=import-error
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=useless-import-alias
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-lines
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-nested-blocks
"""Analysis of OpenFAST results."""

import os
import glob
import shutil
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
import scipy
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from floatbench.colors import COLORS_DICT

from .openfast_platform import PlatformWaterBalancer

RESULTS_FILENAME = 'IEA-22-280-RWT-Semi.out'
METRICS_COLUMNS = {
    'wind_speed': 'Wind1VelX',
    'rotor_speed': 'RotSpeed',
    'rotor_torque': 'RotTorq',
    'rotor_thrust': 'RotThrust',
    'rotor_fxh': 'RtAeroFxh',
    'rotor_myh': 'RtAeroMyh',
    'electrical_power': 'GenPwr',
    'blade_pitch': 'BldPitch1',
    'wind_speed_hub': 'WindHubVelX',
    'tower_bottom_mss': 'TwrBsMxt',
    'tower_bottom_mfa': 'TwrBsMyt',
    'tower_1_mfa': 'TwHt1MLyt',
    'tower_2_mfa': 'TwHt2MLyt',
    'tower_3_mfa': 'TwHt3MLyt',
    'tower_4_mfa': 'TwHt4MLyt',
    'tower_5_mfa': 'TwHt5MLyt',
    'tower_6_mfa': 'TwHt6MLyt',
    'tower_7_mfa': 'TwHt7MLyt',
    'tower_8_mfa': 'TwHt8MLyt',
    'tower_9_mfa': 'TwHt9MLyt',
    'tower_7_afa': 'TwHt7ALxt',
    'tower_7_ass': 'TwHt7ALyt',
    'tower_top_afa': 'YawBrTAxp',
    'tower_top_afa_mod': 'YawBrTAxpMod',
    'tower_top_ass': 'YawBrTAyp',
    'tower_top_ffa': 'YawBrFxp',
    'tower_top_mfa': 'YawBrMyp',
    'tower_top_mss': 'YawBrMxp',
    'tower_top_rfa': 'YawBrRDyt',
    'plat_surge': 'PtfmSurge',
    'plat_sway': 'PtfmSway',
    'plat_heave': 'PtfmHeave',
    'plat_roll': 'PtfmRoll',
    'plat_pitch': 'PtfmPitch',
    'plat_yaw': 'PtfmYaw',
    'wave_elev': 'Wave1Elev',
}
SCALE_FACTORS = {
    "RotTorq": 1e-3,
    "RotThrust": 1e-3,
    "GenPwr": 1e-3,
    "TwrBsMxt": 1e-3,
    "TwrBsMyt": 1e-3,
    "YawBrMyp": 1e-3,
    "YawBrMxp": 1e-3,
    "RtAeroFxh": 1e-6
}
DERIVED_COLUMNS = {
    "YawBrTAxpMod":
        lambda df: (df["YawBrTAxp"] - 9.81 * np.sin(
            np.deg2rad(df["YawBrRDyt"] + df["PtfmPitch"])))
}

# Metric groups for spectral computation
SPECTRAL_ACCEL_METRICS = ['tower_top_afa', 'tower_top_afa_mod', 'tower_top_ass']
SPECTRAL_MOMENT_METRICS = [
    'tower_bottom_mss', 'tower_bottom_mfa', 'tower_top_mfa', 'tower_top_mss'
]
SPECTRAL_PLAT_METRICS = [
    'plat_surge', 'plat_sway', 'plat_heave', 'plat_roll', 'plat_pitch',
    'plat_yaw'
]


class OpenFASTOutput:
    """Loads and represents data from an OpenFAST output file."""

    def __init__(self,
                 output_dir: str,
                 results_filename: str = RESULTS_FILENAME):
        """
        Initializes an OpenFASTOutput instance.

        Args:
            output_dir (str): Directory containing the OpenFAST output.
            results_filename (str, optional): Name of the results file. Defaults
              to "OpenFAST.out".
        """
        self.output_dir = output_dir
        self.results_filename = results_filename
        self.output_id = os.path.basename(os.path.dirname(self.output_dir))
        self.results = self.load_results()

    def load_results(self) -> pd.DataFrame:
        """
        Loads data from the specified OpenFAST output file.

        Returns:
            pd.DataFrame: DataFrame containing the parsed OpenFAST output data.
        """
        results_path = glob.glob(os.path.join(self.output_dir, "**",
                                              self.results_filename),
                                 recursive=True)[0]

        return pd.read_csv(results_path,
                           skiprows=[0, 1, 2, 3, 4, 5, 7],
                           delimiter=r"\s+",
                           header=0)


class OpenFASTOutputAnalysis(OpenFASTOutput):
    """Analyzes results from a single OpenFAST output."""

    def __init__(self,
                 output_dir: str,
                 metrics_columns: dict = METRICS_COLUMNS):
        """
        Initializes an OpenFASTOutputAnalysis instance.

        Args:
            output_dir (str): Directory containing the OpenFAST output.
            metrics_columns (dict, optional): Mapping of metric names to dataset
              column names.
        """
        super().__init__(output_dir)
        self.results = self._ensure_derived_columns(self.results)

        self.output_dir = output_dir
        self.metrics_columns = metrics_columns

    def _ensure_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures that all derived columns are present in the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to check and update.
        
        Returns:
            pd.DataFrame: The updated DataFrame with derived columns added if
              they were missing.
        """

        for col_name, fn in DERIVED_COLUMNS.items():
            if col_name not in df.columns:
                df[col_name] = fn(df)
        return df

    def get_simulation_duration(self, min_time: float = None) -> float:
        """
        Retrieves the total duration of the simulation (in seconds).

        Returns:
            float: Total simulation time in seconds.
        """
        df = self.results
        if min_time is not None:
            df = df[df["Time"] >= min_time]

        return df["Time"].max() - df["Time"].min()

    def get_column_from_metric(self, metric_name: str) -> str:
        """
        Retrieves the dataset column name for a given metric.

        Args:
            metric_name (str): The name of the metric (e.g., 'wind_speed').

        Returns:
            str: The corresponding dataset column name.
            
        Raises:
            ValueError: If the metric is not found.
        """
        if metric_name not in self.metrics_columns:
            raise ValueError(
                f"Metric '{metric_name}' not found in metrics_columns.")

        return self.metrics_columns.get(metric_name)

    def get_metric_from_column(self, column_name: str) -> str:
        """
        Retrieves the dataset column name for a given metric.

        Args:
            column_name (str): The dataset column name (e.g., 'Wind1VelX').

        Returns:
            str: str: The corresponding metric name.
            
        Raises:
            ValueError: If the column is not found.
        """
        for metric, column in self.metrics_columns.items():
            if column == column_name:
                return metric

        raise ValueError(
            f"Column '{column_name}' not found in metrics_columns.")

    def filter_by_metrics_and_time(self,
                                   metrics: list = None,
                                   min_time: float = 0.0,
                                   include_time: bool = True) -> pd.DataFrame:
        """
        Filters data by selected metrics and a time threshold.

        Args:
            metrics (list, optional): List of metric names. Defaults to all
              metrics.
            min_time (float, optional): Minimum time threshold. Defaults to 0.0.
            include_time (bool, optional): Whether to include the 'Time' column.
              Defaults to True.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        metrics = metrics or list(self.metrics_columns.keys())
        column_names = [self.get_column_from_metric(m) for m in metrics]

        return self.results.loc[self.results['Time'] >= min_time, ['Time'] +
                                column_names if include_time else column_names]

    def compute_mean_filtered_by_metrics_and_time(self,
                                                  metrics=None,
                                                  min_time=0.0) -> dict:
        """
        Computes mean values for specified metrics where time >= min_time.

        Args:
            metrics (list, optional): List of metric names. Defaults to all 
              available metrics.
            min_time (float, optional): Minimum time threshold. Defaults to 0.0.

        Returns:
            pd.DataFrame: DataFrame containing mean values for each metric.
        """
        metrics = metrics or list(self.metrics_columns.keys())

        # Filter dataset and exclude 'Time' column
        filtered_df = self.filter_by_metrics_and_time(metrics=metrics,
                                                      min_time=min_time,
                                                      include_time=False)

        return filtered_df.mean()

    def compute_std_filtered_by_metrics_and_time(self,
                                                 metrics=None,
                                                 min_time=0.0) -> dict:
        """
        Computes standard deviation for specified metrics (time >= min_time).

        Args:
            metrics (list, optional): List of metric names. Defaults to all 
            available metrics.
            min_time (float, optional): Minimum time threshold. Defaults to 0.0.

        Returns:
            pd.DataFrame: DataFrame containing standard deviation for each
              metric.
        """
        metrics = metrics or list(self.metrics_columns.keys())

        # Filter dataset and exclude 'Time' column
        filtered_df = self.filter_by_metrics_and_time(metrics=metrics,
                                                      min_time=min_time,
                                                      include_time=False)

        return filtered_df.std()

    def compute_spectral_filtered_by_metrics_and_time(self,
                                                      metrics=None,
                                                      min_time=0.0,
                                                      sampling_frequency=10.0,
                                                      m=4,
                                                      freq_exponent=1.0,
                                                      f_min=0.0) -> dict:
        """Computes spectral fatigue metric sum(|A|^m * f^e).

        For each metric, computes the FFT of the time series filtered by
        min_time, then sums |A|^m * f^freq_exponent for frequency bins
        where f >= f_min.

        Two typical uses:
          - Moments:       freq_exponent=1,     f_min=0
          - Accelerations: freq_exponent=1-2*m, f_min=0.20

        Args:
            metrics (list, optional): List of metric names. Defaults to
              all available metrics.
            min_time (float, optional): Minimum time threshold. Defaults
              to 0.0.
            sampling_frequency (float, optional): Sampling frequency in
              Hz. Defaults to 10.0.
            m (int, optional): SN curve exponent. Defaults to 4.
            freq_exponent (float, optional): Exponent applied to
              frequency bins. Defaults to 1.0.
            f_min (float, optional): Minimum frequency cutoff in Hz.
              Defaults to 0.0.

        Returns:
            dict: Spectral fatigue metric for each metric column.
        """
        metrics = metrics or list(self.metrics_columns.keys())

        filtered_df = self.filter_by_metrics_and_time(metrics=metrics,
                                                      min_time=min_time,
                                                      include_time=False)

        result = {}
        n = len(filtered_df)
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)
        mask = freqs >= f_min
        freq_weights = freqs[mask]**freq_exponent

        for col in filtered_df.columns:
            signal = filtered_df[col].values
            fft_vals = np.fft.rfft(signal)
            amplitudes = np.abs(fft_vals) * (2.0 / n)
            spectral_sum = np.sum((amplitudes[mask]**m) * freq_weights)
            result[col] = spectral_sum

        return result

    def compute_psd_metric_filtered_by_time(
            self, metric: str, min_time: float, sampling_frequency: float,
            segment_length: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the PSD for a given metric after filtering by time.

        Args:
            metric (str): Name of the metric for which the PSD is computed.
            min_time (float): Minimum time threshold for filtering the data.
            sampling_frequency (float): Sampling frequency of the data.
            segment_length (int): Length of each segment for PSD computation.

        Returns:
            tuple[np.ndarray, np.ndarray]: Frequencies and the corresponding PSD
              values.
        """
        # Filter data for the metric after applying the time threshold
        filtered_data = self.filter_by_metrics_and_time(metrics=[metric],
                                                        min_time=min_time,
                                                        include_time=False)

        # Compute PSD using Welch’s method
        column = self.get_column_from_metric(metric)
        frequencies, psd_values = scipy.signal.welch(
            filtered_data[column].values,
            fs=sampling_frequency,
            nperseg=segment_length)

        return frequencies, psd_values


class OpenFASTDatasetAnalysis:
    """Analyzes multiple OpenFAST outputs within a dataset.
    
    Attributes:
        dataset_dir (str): Directory containing OpenFAST output files.
        analysis_dir (str): Directory to save analysis results.
        dataset_inputs_path (str): Path to the dataset inputs CSV file.
        results_filename (str): Name of the OpenFAST results file.
        min_time (float): Minimum time threshold for filtering data.
        metrics_columns (dict): Mapping of metric names to dataset column names.
        sampling_frequency (int): Sampling frequency for PSD computations.
        segment_length (int): Segment length for PSD computations.
        input_dirs (list[str]): List of input directories for simulations.
        output_dirs (list[str]): List of output directories containing results.
        outputs_analysis (list[OpenFASTOutputAnalysis]): List of OpenFAST output
          analyses for each output directory.
        output_ids (list[str]): List of output IDs corresponding to each output.
    """

    def __init__(self,
                 dataset_dir: str,
                 min_time: float = 400,
                 metrics_columns: dict = METRICS_COLUMNS,
                 results_filename: str = RESULTS_FILENAME,
                 sampling_frequency: int = 10,
                 segment_length: int = 4096,
                 analysis_dir: str = None,
                 dataset_inputs_path: str = None):
        """
        Initializes the OpenFASTDatasetAnalysis instance.

        Args:
            dataset_dir (str): Directory containing OpenFAST output files.
            min_time (float, optional): Minimum time threshold for filtering 
              data. Defaults to 400.
            metrics_columns (dict, optional): Mapping of metric names to dataset
              column names.
            results_filename (str, optional): Name of the OpenFAST results file.
            sampling_frequency (int, optional): Sampling frequency for PSD
              computations. Defaults to 10.
            segment_length (int, optional): Segment length for PSD computations. 
              Defaults to 4096.
            analysis_dir (str, optional): Directory to save analysis results.
                Defaults to a subdirectory "analysis_results" in `dataset_dir`.
            dataset_inputs_path (str, optional): Path to the dataset inputs.
        """
        self.dataset_dir = dataset_dir
        self.analysis_dir = analysis_dir or os.path.join(
            self.dataset_dir, "analysis_results")
        self.dataset_inputs_path = dataset_inputs_path
        self.results_filename = results_filename
        self.min_time = min_time
        self.metrics_columns = metrics_columns
        self.sampling_frequency = sampling_frequency
        self.segment_length = segment_length
        self.input_dirs, self.output_dirs = self.get_simulation_directories()
        self.outputs_analysis = self.load_outputs_analysis()
        self.output_ids = [out.output_id for out in self.outputs_analysis]

    def get_sim_ids(self) -> list[str]:
        """
        Retrieves the simulation IDs from the outputs ids.
        
        Returns:
            list[str]: List of simulation IDs.
        """
        return [int(output_id.split("_")[0]) for output_id in self.output_ids]

    def get_reference_simulation_duration(self) -> float:
        """
        Returns the simulation duration (after min_time) of the first output.
        
        Assuming all simulations have the same duration.

        Returns:
            float: Duration in seconds.
        """
        return self.outputs_analysis[0].get_simulation_duration(self.min_time)

    def extract_wind_speed_and_seed_from_bts_filenames(self) -> pd.DataFrame:
        """
        Extracts wind speed and seed information from BTS file names.

        Returns:
            pd.DataFrame: DataFrame containing the following columns:
                - task_id: Task ID.
                - wind_speed: Wind speed extracted from the file name.
                - seed: Seed extracted from the file name.
        """
        # Build mapping from sim_id to task_id
        task_map = pd.DataFrame({
            "sim_id": self.get_sim_ids(),
            "task_id": self.output_ids
        })

        # Load dataset inputs containing BTS file names
        input_df = pd.read_csv(self.dataset_inputs_path)

        # Extract wind speed and seed from file name
        input_df["wind_speed"] = input_df["file_name_bts"].str.extract(
            r"W(\d{4})")[0].astype(float) / 100
        input_df["seed"] = input_df["file_name_bts"].str.extract(
            r"_(S\d+)\.bts$")

        # Merge task_id
        merged_df = input_df.merge(task_map, on="sim_id", how="left")

        # Final cleanup and column ordering
        result_df = merged_df[["task_id", "wind_speed", "seed"]]
        result_df = result_df.dropna(subset=["task_id"]).reset_index(drop=True)

        return result_df

    def group_ids_by_bts_wind_speed(self,
                                    bin_width: float = None,
                                    v_in: float = 3,
                                    v_out: float = 25) -> tuple[pd.DataFrame]:
        """
        Groups task IDs by nominal wind speeds extracted from BTS file names.
        
        Args:
            bin_width (float, optional): Width of the wind speed bins. If None,
              the bin width is inferred from the spacing between the first two
              unique wind speed values.
            v_in (float, optional): Cut-in wind speed (m/s). Defines the lower
              boundary of the wind speed range considered for binning. Defaults
              to 3 m/s.
            v_out (float, optional): Cut-out wind speed (m/s). Defines the upper
              boundary of the wind speed range considered for binning. Defaults
              to 25 m/s.
              
        Returns:
            pd.DataFrame: DataFrame with:
                - wind_speed_bin: bin interval (e.g. (4.0, 5.0])
                - wind_speed_mid: midpoint of each bin (e.g. 4.5)
                - task_id: list of task IDs in that bin
        """
        df = self.extract_wind_speed_and_seed_from_bts_filenames()

        wind_speed = np.sort(df["wind_speed"].unique())

        # If bin width is not given, infer it from sorted unique wind speeds
        if bin_width is None:
            bin_width = (v_out - v_in) / len(wind_speed)

        if bin_width == 1:
            bin_start, bin_end = v_in, v_out
        elif bin_width == 2:
            bin_start, bin_end = v_in + 0.5, v_out - 0.5
        else:
            bin_start, bin_end = v_in, v_out

        bin_edges = np.arange(bin_start, bin_end + bin_width, bin_width)

        # Assign each wind speed to a bin
        df["wind_speed_bin"] = pd.cut(df["wind_speed"],
                                      bins=bin_edges,
                                      include_lowest=False)

        # Aggregate task_id by wind bin
        grouped_df = df.groupby("wind_speed_bin")["task_id"].agg(
            list).reset_index(name="task_id")

        grouped_df["wind_speed_mid"] = grouped_df["wind_speed_bin"].apply(
            lambda b: round(b.mid, 1))

        grouped_df = grouped_df[grouped_df["task_id"].map(len) > 0]

        return grouped_df[['wind_speed_bin', 'wind_speed_mid',
                           'task_id']].reset_index(drop=True)

    def get_simulation_directories(self) -> tuple[list[str], list[str]]:
        """
        Retrieves valid input and output directories for OpenFAST simulations.

        Returns:
            tuple[list[str], list[str]]: 
                - List of input directories (inputs/sim_dir). If missing, 
                  None is assigned.
                - List of output directories (outputs), only if they contain
                  .out files.
        """

        def extract_leading_number(path):
            name = os.path.basename(os.path.normpath(path))
            try:
                return int(name.split("_")[0])
            except (IndexError, ValueError):
                return float('inf')

        results_dirs = sorted(glob.glob(os.path.join(self.dataset_dir, "*/"),
                                        recursive=True),
                              key=extract_leading_number)

        input_dirs = []
        output_dirs = []

        for result_dir in results_dirs:
            input_dir = os.path.join(result_dir, "inputs", "sim_dir")
            output_dir = os.path.join(result_dir, "outputs")

            # Check if the output directory contains .out files
            if glob.glob(os.path.join(output_dir, "*.out")):
                output_dirs.append(output_dir)

                # Check if the input directory contains .fst files
                input_dirs.append(input_dir if glob.glob(
                    os.path.join(input_dir, "*.fst")) else None)

        return input_dirs, output_dirs

    def load_outputs_analysis(self) -> list[OpenFASTOutputAnalysis]:
        """
        Loads all OpenFAST outputs available in the directory.

        Returns:
            list: A list of OpenFASTOutputAnalysis objects, each representing
               a single output.
        """
        openfast_outputs = []
        for output_dir in self.output_dirs:
            openfast_outputs.append(
                OpenFASTOutputAnalysis(output_dir=output_dir,
                                       metrics_columns=self.metrics_columns))

        return openfast_outputs

    def group_results_labels_ids_by_time(
            self,
            id_label=True) -> tuple[list[pd.DataFrame], list[str], list[str]]:
        """Groups and sorts filtered metrics, labels and IDs by time.
        
        Args:
            id_label (bool, optional): If True, uses output ID as label.
              If False, uses Wind1VelX (wind speed). Defaults to True.

        Returns:
            tuple[list[pd.DataFrame], list[str], list[str]]: A tuple containing:
                - A list of filtered results sorted in ascending order by label
                  values.
                - A list of sorted labels (wind speeds as strings).
                - A list of corresponding output IDs.
        """
        results, labels, ids = [], [], []

        for output in self.outputs_analysis:
            filtered_results = output.filter_by_metrics_and_time(
                min_time=self.min_time, include_time=True)
            results.append(filtered_results)

            if id_label:
                label = output.output_id.split(
                    '_')[0] if '_' in output.output_id else output.output_id
            else:
                mean_metrics = output.compute_mean_filtered_by_metrics_and_time(
                    min_time=self.min_time)
                label = f"{float(mean_metrics['Wind1VelX']):.2f}"

            labels.append(label)
            ids.append(output.output_id)

        # Sort the results and labels based on label values
        sorted_pairs = sorted(
            zip(labels, ids, results),
            key=lambda x: (not x[0].replace('.', '', 1).isdigit(), float(x[0])
                           if x[0].replace('.', '', 1).isdigit() else x[0]))

        sorted_labels, sorted_ids, sorted_results = zip(*sorted_pairs)

        return list(sorted_results), list(sorted_ids), list(sorted_labels)

    def group_ids_by_mean_wind_speed(self,
                                     bin_width: float = 1.0,
                                     save_csv: bool = False) -> pd.DataFrame:
        """
        Groups task IDs by simulated mean wind speed using Wind1VelX values.
        
        Args:
            bin_width (float, optional): Width of the wind speed bins. Defaults
              to 1.0.
            save_csv (bool, optional): If True, saves the grouped task IDs to a
              CSV file. Defaults to False.
        
        Returns:
            pd.DataFrame: DataFrame with columns:
                - wind_speed_bin: bin interval (e.g. (4.0, 5.0])
                - wind_speed_mid: midpoint of each bin (e.g. 4.5)
                - task_id: list of task IDs in that bin
        """
        # Compute mean metrics for all tasks
        results = self.compute_mean_metrics()

        # Group task IDs by exact Wind1VelX values
        grouped = (
            results.groupby("Wind1VelX")["task_id"].apply(list).reset_index(
                name="task_id"))

        # Define bin edges based on min/max wind speed
        vmin, vmax = grouped["Wind1VelX"].min(), grouped["Wind1VelX"].max()
        bin_start, bin_end = np.floor(vmin), np.ceil(vmax)
        bin_edges = np.arange(bin_start, bin_end + bin_width, bin_width)

        # Assign each Wind1VelX to a bin
        grouped["wind_speed_bin"] = pd.cut(grouped["Wind1VelX"],
                                           bins=bin_edges,
                                           include_lowest=True)
        grouped = grouped.sort_values(by="Wind1VelX", ascending=True)

        # Flatten nested task_id lists by bin
        grouped_by_bin = grouped.groupby("wind_speed_bin")["task_id"].agg(
            lambda x: list(chain.from_iterable(x))).reset_index(name="task_id")

        # Add bin midpoints
        grouped_by_bin["wind_speed_mid"] = grouped_by_bin[
            "wind_speed_bin"].apply(lambda b: round(b.mid, 1))

        if save_csv:
            filename = f"grouped_task_ids_mintime{self.min_time:.1f}".replace(
                ".", "_") + ".csv"
            os.makedirs(self.analysis_dir, exist_ok=True)
            csv_path = os.path.join(self.analysis_dir, filename)
            grouped_by_bin.to_csv(csv_path, index=False, encoding="utf-8")

        return grouped_by_bin.reset_index(drop=True)

    def compute_mean_metrics(self, save_csv: bool = False) -> pd.DataFrame:
        """Computes mean metrics from outputs.

        Args:
            save_csv (bool, optional): If True, saves the computed metrics to a
              CSV file. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing mean metrics for each output
              file.
        """
        results = []

        for output in self.outputs_analysis:
            mean_metrics_df = output.compute_mean_filtered_by_metrics_and_time(
                min_time=self.min_time)
            metrics_summary = {"task_id": output.output_id}
            metrics_summary.update(mean_metrics_df)
            results.append(metrics_summary)

        results_df = pd.DataFrame(results)

        if save_csv:
            filename = f"mean_metrics_mintime{self.min_time:.1f}".replace(
                ".", "_") + ".csv"
            os.makedirs(self.analysis_dir, exist_ok=True)
            csv_path = os.path.join(self.analysis_dir, filename)
            results_df.to_csv(csv_path, index=False, encoding="utf-8")

        return results_df.reset_index(drop=True)

    def compute_std_metrics(self, save_csv: bool = False) -> pd.DataFrame:
        """Computes standard deviation of metrics from outputs.

        Args:
            save_csv (bool, optional): If True, saves the computed metrics to a
              CSV file. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing std metrics for each output
              file.
        """
        results = []

        for output in self.outputs_analysis:
            std_metrics_df = output.compute_std_filtered_by_metrics_and_time(
                min_time=self.min_time)
            metrics_summary = {"task_id": output.output_id}
            metrics_summary.update(std_metrics_df)
            results.append(metrics_summary)

        results_df = pd.DataFrame(results)

        if save_csv:
            filename = f"std_metrics_mintime{self.min_time:.1f}".replace(
                ".", "_") + ".csv"
            os.makedirs(self.analysis_dir, exist_ok=True)
            csv_path = os.path.join(self.analysis_dir, filename)
            results_df.to_csv(csv_path, index=False, encoding="utf-8")

        return results_df.reset_index(drop=True)

    def compute_spectral_metrics(self,
                                 m=4,
                                 f_min_accel=0.20,
                                 save_csv: bool = False) -> pd.DataFrame:
        """Computes spectral fatigue metrics for all outputs.

        Uses two formulas depending on metric type:
          - Accelerations: sum(|A|^m * f^(1-2m)), f >= f_min_accel
          - Moments:       sum(|A|^m * f),        f >= 0

        Args:
            m (int, optional): SN curve exponent. Defaults to 4.
            f_min_accel (float, optional): Frequency cutoff for
              acceleration metrics in Hz. Defaults to 0.20.
            save_csv (bool, optional): If True, saves to CSV.
              Defaults to False.

        Returns:
            pd.DataFrame: Spectral metrics for each output file.
        """
        accel_exp = 1.0 - 2.0 * m
        results = []

        for output in self.outputs_analysis:
            metrics_summary = {"task_id": output.output_id}

            # Accelerations: sum(|A|^m * f^(1-2m)), f >= f_min
            accel = (output.compute_spectral_filtered_by_metrics_and_time(
                metrics=SPECTRAL_ACCEL_METRICS,
                min_time=self.min_time,
                sampling_frequency=self.sampling_frequency,
                m=m,
                freq_exponent=accel_exp,
                f_min=f_min_accel))
            metrics_summary.update(accel)

            # Moments: sum(|A|^m * f), f >= 0
            moments = (output.compute_spectral_filtered_by_metrics_and_time(
                metrics=SPECTRAL_MOMENT_METRICS,
                min_time=self.min_time,
                sampling_frequency=self.sampling_frequency,
                m=m,
                freq_exponent=1.0,
                f_min=0.0))
            metrics_summary.update(moments)

            # Platform: sum(|A|^m * f), f >= 0
            plat = (output.compute_spectral_filtered_by_metrics_and_time(
                metrics=SPECTRAL_PLAT_METRICS,
                min_time=self.min_time,
                sampling_frequency=self.sampling_frequency,
                m=m,
                freq_exponent=1.0,
                f_min=0.0))
            metrics_summary.update(plat)

            results.append(metrics_summary)

        results_df = pd.DataFrame(results)

        if save_csv:
            filename = (f"spectral_metrics_mintime{self.min_time:.1f}").replace(
                ".", "_") + ".csv"
            os.makedirs(self.analysis_dir, exist_ok=True)
            csv_path = os.path.join(self.analysis_dir, filename)
            results_df.to_csv(csv_path, index=False, encoding="utf-8")

        return results_df.reset_index(drop=True)

    def count_wind_speeds(self,
                          results_df: pd.DataFrame,
                          save_csv: bool = False) -> pd.DataFrame:
        """Counts wind speed occurrences and saves to CSV.

        Args:
            results_df (pd.DataFrame): DataFrame containing mean metrics.
            save_csv (bool, optional): If True, saves the wind speed counts to a
              CSV file. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame with wind speeds and their corresponding
              counts.
        """
        # Count occurrences of each wind speed
        wind_counts = results_df['Wind1VelX'].value_counts().reset_index()
        wind_counts.columns = ['Wind1VelX', 'count']
        wind_counts = wind_counts.sort_values(by='Wind1VelX')

        # Save to CSV if requested
        if save_csv:
            filename = f"wind_speed_counts_mintime{self.min_time:.1f}".replace(
                ".", "_") + ".csv"
            os.makedirs(self.analysis_dir, exist_ok=True)
            csv_path = os.path.join(self.analysis_dir, filename)
            wind_counts.to_csv(csv_path, index=False, encoding="utf-8")

        return wind_counts

    def compute_mean_colwater_plat_metrics(self,
                                           save_csv: bool = False
                                          ) -> pd.DataFrame:
        """"Computes mean column water balance metrics for horizontal stability.

        This method calculates the required water mass adjustments in platform 
        columns to counteract horizontal moment imbalances, ensuring stability.

        Args:
            save_csv (bool, optional): If True, saves the computed metrics to a
              CSV file. Defaults to False.
              
        Returns:
            pd.DataFrame: A DataFrame containing computed column water
              parameters for horizontal platform balance, including initial
              mass, moment contributions, water height adjustments, and final
              platform mass.
              
        Raises:
            ValueError: If any required input directory is missing.
        """
        # Check if any input directory is missing
        if any(input_dir is None for input_dir in self.input_dirs):
            raise ValueError("Error: Missing required input directories. All "
                             "input directories must be available to compute "
                             "mean column water balance metrics.")

        # Compute mean metrics
        metrics = self.compute_mean_metrics()

        # Process each output directory
        results = []
        for i, input_dir in enumerate(self.input_dirs):
            balancer = PlatformWaterBalancer(metrics=metrics.iloc[[i]],
                                             input_dir=input_dir)
            balance_metrics = balancer.compute_colwater_parameters()
            results.append(balance_metrics)

        # Convert list of DataFrames into a single DataFrame
        results_df = pd.concat(results, ignore_index=True)

        # Sort by wind speed if the column exists
        results_df = results_df.sort_values(by="Wind1VelX", ascending=True)

        if save_csv:
            # Construct and save CSV filename
            filename = f"plathor_metrics_mintime{self.min_time:.1f}".replace(
                ".", "_") + ".csv"
            columns = [
                "task_id", "plat_mass_init", "colwater_moment", "colwater_type",
                "colwater_height_init", "colwater_height", "colwater_mass",
                "plat_mass", "colwater_z"
            ]
            os.makedirs(self.analysis_dir, exist_ok=True)
            csv_path = os.path.join(self.analysis_dir, filename)
            results_df.to_csv(csv_path,
                              index=False,
                              encoding="utf-8",
                              columns=columns)

        return results_df.reset_index(drop=True)

    def compute_psd_metric(
            self, metric: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
        """
        Computes the PSD for a specific metric across multiple outputs.

        The PSD is calculated for each output after filtering by the minimum
        time threshold. Results are organized and sorted by the output labels.

        Args:
            column_name (str, optional): Name of the column for which PSD is
              computed. Defaults to 'TwrBsMyt'.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[str]]:
                - A list of frequency arrays for each output.
                - A list of PSD value arrays for each output.
                - A list of sorted labels corresponding to each output.
        """
        results = []

        for output in self.outputs_analysis:
            # Compute PSD for the metric after filtering by time
            (frequencies,
             psd_values) = output.compute_psd_metric_filtered_by_time(
                 metric, self.min_time, self.sampling_frequency,
                 self.segment_length)

            results.append((output.output_id, frequencies, psd_values))

        # Sort results by labels (numeric-first sorting)
        sorted_results = sorted(results,
                                key=lambda x: int(x[0])
                                if x[0].isdigit() else x[0])
        sorted_labels, sorted_frequencies, sorted_psd_values = zip(
            *sorted_results)

        return list(sorted_frequencies), list(sorted_psd_values), list(
            sorted_labels)

    def compute_psd_metrics(
        self, metrics: list[str]
    ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[str]]:
        """
        Computes the PSD for multiple metrics across multiple outputs.

        Args:
            metrics (list[str]): List of metric names for which PSD is computed.

        Returns:
            tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[str]]:
                - A list of frequency arrays for each metric.
                - A list of PSD value arrays for each metric.
                - A list of sorted labels corresponding to each output.
        """
        frequencies_all_metrics = []
        psd_values_all_metrics = []

        for metric in metrics:
            # Compute PSD data for the specified column
            (frequencies_metric, psd_values_metric,
             label_metric) = self.compute_psd_metric(metric)

            frequencies_all_metrics.append(frequencies_metric)
            psd_values_all_metrics.append(psd_values_metric)

        return frequencies_all_metrics, psd_values_all_metrics, label_metric

    def compute_psd_heatmap_by_windspeed(
            self, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a PSD heatmap for a specified column, organized by wind speed.

        Args:
            metric (str): The name of the metric for which the PSD heatmap is
              computed.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Frequencies (np.ndarray): Frequency bins of the PSD computation.
            - Wind speeds (np.ndarray): Unique sorted wind speed values.
            - PSD heatmap (np.ndarray): Averaged PSD values per wind speed.
        """
        num_files = len(self.outputs_analysis)

        # Initialize wind speed array and PSD heatmap
        wind_speeds = np.zeros(num_files)
        psd_heatmap = np.zeros((int(self.segment_length / 2) + 1, num_files))

        for i, output in enumerate(self.outputs_analysis):
            # Compute mean metrics and extract wind speed
            results_df = output.compute_mean_filtered_by_metrics_and_time(
                min_time=self.min_time)
            wind_speeds[i] = float(results_df["Wind1VelX"])

            # Compute PSD for the specified column
            (frequencies,
             psd_values) = output.compute_psd_metric_filtered_by_time(
                 metric, self.min_time, self.sampling_frequency,
                 self.segment_length)

            psd_heatmap[:, i] = psd_values

        # Sort the heatmap and wind speeds by wind speed
        idx = np.argsort(wind_speeds)
        psd_heatmap = psd_heatmap[:, idx]
        wind_speeds = wind_speeds[idx]

        # Group by repeated wind speeds and compute averages
        unique_ws = np.unique(wind_speeds)
        psd_mean = np.zeros((psd_heatmap.shape[0], len(unique_ws)))

        for j, ws in enumerate(unique_ws):
            mask = wind_speeds == ws
            psd_mean[:, j] = psd_heatmap[:, mask].mean(axis=1)

        return frequencies, unique_ws, psd_mean

    def plot_rotor_metrics_by_windspeed_benchmark(self,
                                                  rotor_ref_path: str,
                                                  save_svg: bool = False
                                                 ) -> None:
        """
        Compares rotor performance metrics from OpenFAST with the IEA 22MW RWT.

        Creates five subplots:
        - Blade pitch
        - Power output
        - Rotor thrust
        - Rotor angular velocity
        - Rotor torque
            
        Args:
            rotor_ref_path (str): Path to a reference file for rotor performance
              benchmarking.
            save_svg (bool): If True, saves the plot as an SVG file.
        """
        # Compute mean metrics for our simulation
        output_df = self.compute_mean_metrics()

        # Load reference data (IEA 22MW RWT)
        rename_dict = {
            "wind_speed": "Wind1VelX",
            "pitch": "BldPitch1",
            "electrical_power": "GenPwr",
            "rotor_thrust": "RotThrust",
            "rotor_speed": "RotSpeed",
            "rotor_torque": "RotTorq",
        }
        reference_df = pd.read_csv(rotor_ref_path).rename(columns=rename_dict)

        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Configure the figure and grid layout
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        gs = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.1)

        # Define subplots
        axes = [
            plt.subplot(gs[0, 0], facecolor='white'),
            plt.subplot(gs[0, 1], facecolor='white'),
            plt.subplot(gs[0, 2], facecolor='white'),
            plt.subplot(gs[1, 0], facecolor='white'),
            plt.subplot(gs[1, 1], facecolor='white'),
        ]

        # Define plot parameters
        max_wind_speed = output_df.Wind1VelX.max() + 0.5

        ysim_values = [
            output_df.BldPitch1, output_df.GenPwr * 1e-3,
            output_df.RotThrust * 1e-3 -
            3.61 * np.sin(np.deg2rad(6 + output_df.PtfmPitch)),
            output_df.RotSpeed, output_df.RotTorq * 1e-3
        ]
        yref_values = [
            reference_df.BldPitch1, reference_df.GenPwr * 1e-6,
            reference_df.RotThrust * 1e-6, reference_df.RotSpeed,
            reference_df.RotTorq * 1e-6
        ]
        ylabels = [
            'Blade Pitch (deg)', 'Power Output (MW)', 'Thrust (MN)',
            'Angular Velocity (rpm)', 'Torque (MN.m)'
        ]

        # Plot each metric
        for ax, ysim, yref, ylabel in zip(axes, ysim_values, yref_values,
                                          ylabels):
            ax.plot(reference_df.Wind1VelX,
                    yref,
                    '.-',
                    linewidth=1.5,
                    color=COLORS_DICT["grey_paper"],
                    alpha=0.9,
                    markersize=4,
                    label="IEA 22 MW Reference")
            ax.plot(output_df.Wind1VelX,
                    ysim,
                    '.-',
                    linewidth=1.5,
                    color=COLORS_DICT["red_paper"],
                    alpha=0.9,
                    markersize=4,
                    label="FADO-FWOT")

            # Customize axes and appearance
            ax.set_ylabel(ylabel, fontsize=10, labelpad=10)
            ax.set_xlim(0, max_wind_speed)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10,
                           color=COLORS_DICT["dark_gray_paper"])
            ax.grid(True, linestyle="-", color=COLORS_DICT["light_gray_paper"])

            if ax in axes[-3:]:
                ax.set_xlabel('Mean Wind Speed (m/s)', fontsize=10, labelpad=10)
            else:
                ax.tick_params(axis='x', labelbottom=False)

            # Hide unnecessary spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(1)
                ax.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Add a shared legend for all subplots
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,
                   legend_labels,
                   loc='lower right',
                   fontsize=10,
                   frameon=False)

        # Save the plot
        plot_dir = os.path.join(self.analysis_dir, "plot_benchmark")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "rotor_benchmark.png")
        fig.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        if save_svg:
            svg_path = os.path.join(plot_dir, "rotor_benchmark.svg")
            plt.savefig(svg_path,
                        format='svg',
                        bbox_inches="tight",
                        pad_inches=0,
                        transparent=True)
        plt.close(fig)

    def plot_metrics(self,
                     plot_configs: dict,
                     x_axis: str,
                     stat: str = "mean",
                     plot_name: str = "results.png",
                     id_label: bool = False,
                     outputs_id: list = None,
                     overlay_plots: bool = False,
                     save_svg: bool = False) -> None:
        """
        Plots selected metrics from OpenFAST outputs.

        Dynamically generates subplots for time or wind speed plots.

        Args:
            plot_configs (dict): Configuration dictionary with:
                - "metrics": List of metric names to plot.
                - "ylabels": Corresponding y-axis labels.
                - "plot_labels": Labels for the legend.
                - "grid_shape": Tuple (rows, cols) for subplot layout.
                - "xmin", "xmax", "ymin", "ymax": Axis limits.
            x_axis (str): X-axis variable ("time" or "wind_speed").
            stat (str, optional): Statistic to plot ("mean" or "std"). Defaults
              to "mean". 
            plot_name (str, optional): Output file name. Defaults to 
              "results.png".
            id_label (bool, optional): If True, uses output ID as label.
              If False, uses Wind1VelX (wind speed). Defaults to True.
            outputs_id (list, optional): IDs of specific outputs to include.
            overlay_plots (bool, optional): If True, overlays all metrics on a
              single plot.
            save_svg (bool): If True, saves the plot as an SVG file.
            
        Raises:
            ValueError: If an invalid stat is provided for wind_speed plots.
            ValueError: If an invalid x_axis is provided.
        """

        # Handle input data based on x_axis
        if x_axis in ("time", "frequency"):
            results, ids, labels = self.group_results_labels_ids_by_time(
                id_label=id_label)
            type_plot = "line_plot"
            positions = None

        elif x_axis == "wind_speed":
            if stat not in ["mean", "std"]:
                raise ValueError(
                    "stat must be 'mean' or 'std' for wind_speed plots")

            # Select the correct source DataFrame based on the statistic
            if stat != "mean":
                results_std = self.compute_std_metrics()
                results = [results_std]
            else:
                results_mean = self.compute_mean_metrics()
                results = [results_mean]

            # Labels and IDs are not used for wind_speed plots
            ids, labels = [None], [None]

            # Group task IDs by wind speed bin
            grouped_df = self.group_ids_by_mean_wind_speed()
            wind_speeds_sorted = grouped_df["wind_speed_mid"].tolist()
            positions = np.arange(len(wind_speeds_sorted))

            # Determine if violin plot is needed (more than one task per bin)
            multi_task_groups = grouped_df[grouped_df["task_id"].apply(len) > 1]
            if len(multi_task_groups) != 0:
                type_plot = "violin_plot"
            else:
                type_plot = "line_plot"

        else:
            raise ValueError(
                "x_axis must be 'time', 'frequency', or 'wind_speed'")

        # Generate colormap
        cmap = plt.cm.get_cmap("jet", len(results))

        # Extract plot parameters
        metrics_name = plot_configs.get("metrics", [])
        ylabels = plot_configs.get("ylabels", [])
        plot_labels = plot_configs.get("plot_labels", None)
        xmin = plot_configs.get("xmin") or [None]
        xmax = plot_configs.get("xmax") or [None]
        ymin = plot_configs.get("ymin") or [None]
        ymax = plot_configs.get("ymax") or [None]
        numplots = len(metrics_name)
        xmin = xmin if xmin and len(xmin) == numplots else [xmin[0]] * numplots
        xmax = xmax if xmax and len(xmax) == numplots else [xmax[0]] * numplots
        ymin = ymin if ymin and len(ymin) == numplots else [ymin[0]] * numplots
        ymax = ymax if ymax and len(ymax) == numplots else [ymax[0]] * numplots

        # Filter specific outputs if needed
        if outputs_id is not None and x_axis == "time":
            filtered_results, filtered_ids, filtered_labels = [], [], []
            if plot_labels is not None:
                id_to_label = dict(zip(outputs_id, plot_labels))
            for id_value, label, result in zip(ids, labels, results):
                if id_value in outputs_id:
                    if plot_labels is not None:
                        label = id_to_label.get(id_value, label)
                    filtered_results.append(result)
                    filtered_labels.append(label)
                    filtered_ids.append(id_value)
            results, labels, ids = (filtered_results, filtered_labels,
                                    filtered_ids)

        # Extract grid shape and convert to integers
        grid_shape = plot_configs.get("grid_shape")
        rows, cols = map(int, grid_shape)

        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Configure the figure and grid layout
        if overlay_plots:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
            ax.set_facecolor("white")
            axes = [ax]
        else:

            fig = plt.figure(figsize=(10 / 3 * cols, 3 * rows),
                             facecolor='white')
            gs = gridspec.GridSpec(rows,
                                   cols,
                                   wspace=0.4 * cols / 2,
                                   hspace=0.2)

            # Define subplots
            axes = []
            for i in range(numplots):
                row, col = divmod(i, cols)
                ax = plt.subplot(gs[row, col], facecolor='white')
                axes.append(ax)

        # Retrieve metric columns
        metrics = [
            self.metrics_columns.get(metric, None) for metric in metrics_name
        ]

        # Plot each metric
        output_df = None

        if overlay_plots:
            ax = axes[0]
            cmap = plt.cm.get_cmap("jet", len(results) * len(metrics))
            metric_id = 0
            for metric, ylabel, xlim, ylim in zip(metrics, ylabels,
                                                  zip(xmin, xmax),
                                                  zip(ymin, ymax)):
                # scaling
                scale = SCALE_FACTORS.get(metric, 1)

                if x_axis == "frequency":
                    for j, output in enumerate(self.outputs_analysis):
                        metric_name = output.get_metric_from_column(metric)
                        frequencies, psd_values = (
                            output.compute_psd_metric_filtered_by_time(
                                metric_name, self.min_time,
                                self.sampling_frequency, self.segment_length))

                        color = cmap(
                            (j + metric_id) /
                            max(1,
                                len(self.outputs_analysis) * len(metrics) - 1))

                        ax.plot(
                            frequencies,
                            np.log10(psd_values),
                            "-",
                            linewidth=1.2,
                            alpha=0.9,
                            color=color,
                            label=f"{ylabel}",
                        )

                    metric_id += 1

                else:
                    for j, (output_df, label) in enumerate(zip(results,
                                                               labels)):
                        if x_axis == "time":
                            xdata = output_df["Time"]
                        else:  # wind_speed
                            xdata = output_df["Wind1VelX"]

                        ydata = output_df[metric] * scale
                        color = cmap(
                            (j + metric_id) /
                            max(1,
                                len(self.outputs_analysis) * len(metrics) - 1))

                        ax.plot(
                            xdata,
                            ydata,
                            "-" if x_axis == "time" else ".-",
                            linewidth=1.2,
                            markersize=3,
                            alpha=0.9,
                            color=color,
                            label=f"{ylabel}",
                        )
                    metric_id += 1

                # xlim handling (only once is fine; repeated doesn't hurt)
                if xlim[0] is not None or xlim[1] is not None:
                    ax.set_xlim(xlim)

            # overlay_plots axis labels
            xlabel_map = {
                "time": "Time (s)",
                "wind_speed": "Mean Wind Speed (m/s)",
                "frequency": "Frequency (Hz)"
            }
            ax.set_xlabel(xlabel_map.get(x_axis, ""), fontsize=10, labelpad=10)
            ax.set_ylabel("Value (scaled)", fontsize=10, labelpad=10)

            # Hide unnecessary spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(1)
                ax.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        else:
            for ax, metric, ylabel, xlim, ylim in zip(axes, metrics, ylabels,
                                                      zip(xmin, xmax),
                                                      zip(ymin, ymax)):

                if (x_axis == "time" or x_axis == "frequency" or
                        type_plot == "line_plot"):

                    for i, (output_df, label) in enumerate(zip(results,
                                                               labels)):

                        column_name = ""

                        # Define x-axis and line styl
                        if x_axis == "time":
                            column_name = "Time"
                            line_style = '-'

                        elif x_axis == "frequency":
                            line_style = '-'
                        else:
                            label = (plot_labels
                                     if plot_labels is not None else None)
                            column_name = "Wind1VelX"
                            line_style = '.-'

                        # Get color and scaling
                        color = cmap(i / max(1, len(labels) - 1))
                        scale = SCALE_FACTORS.get(metric, 1)

                        if x_axis != "frequency":

                            xdata = output_df[column_name]
                            ydata = output_df[metric] * scale

                            # Plot line
                            ax.plot(xdata,
                                    ydata,
                                    line_style,
                                    linewidth=1.5,
                                    markersize=4,
                                    label=label,
                                    alpha=0.9,
                                    color=color)

                        if x_axis == "frequency":
                            for i, output in enumerate(self.outputs_analysis):
                                metric_name = output.get_metric_from_column(
                                    metric)

                                (frequencies, psd_values
                                ) = output.compute_psd_metric_filtered_by_time(
                                    metric_name, self.min_time,
                                    self.sampling_frequency,
                                    self.segment_length)
                                xdata = frequencies
                                ydata = np.log10(psd_values)

                                # Plot line
                                ax.plot(xdata,
                                        ydata,
                                        line_style,
                                        linewidth=1.5,
                                        markersize=4,
                                        label=label,
                                        alpha=0.9,
                                        color=color)

                else:
                    value = []

                    # Collect the values for each wind bin
                    for _, row in grouped_df.iterrows():
                        task_ids = set(row["task_id"])
                        filtered = results[0][results[0]["task_id"].astype(
                            str).isin(task_ids)]
                        value.append(filtered[metric].values)

                    # Create violin plot
                    part = ax.violinplot(value,
                                         positions=positions,
                                         showmedians=False,
                                         showextrema=False)
                    color = cmap(i / max(1, len(labels) - 1))

                    # Set a uniform color for all violin bodies
                    for pc in part['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.6)
                        pc.set_linewidth(0.8)

                    # Overlay boxplot-style statistics
                    for x, vals in zip(positions, value):
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

                # Customize axes and appearance
                ylabel_prefix = {
                    "mean": "Mean ",
                    "std": "Std "
                }.get(stat, "") if x_axis == "wind_speed" else ""
                ylabel_final = f"{ylabel_prefix}{ylabel}"
                ax.set_ylabel(ylabel_final, fontsize=10, labelpad=10)

                if x_axis != "frequency" and type_plot == "line_plot":
                    if xlim[0] is None and xlim[1] is None:
                        xlim = (min(output_df[column_name]),
                                max(output_df[column_name]))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.tick_params(axis='both',
                               which='major',
                               length=4,
                               width=1,
                               labelsize=10,
                               color=COLORS_DICT["dark_gray_paper"])
                ax.grid(True,
                        linestyle="-",
                        color=COLORS_DICT["light_gray_paper"])
                if ax in axes[-cols:]:
                    xlabel_map = {
                        "time": "Time (s)",
                        "wind_speed": "Mean Wind Speed (m/s)",
                        "frequency": "Frequency (Hz)",
                    }

                    ax.set_xlabel(xlabel_map.get(x_axis, ""),
                                  fontsize=10,
                                  labelpad=10)

                else:
                    ax.tick_params(axis='x', labelbottom=False)

                # Hide unnecessary spines
                for spine in ['top', 'right', 'left', 'bottom']:
                    ax.spines[spine].set_visible(True)
                    ax.spines[spine].set_linewidth(1)
                    ax.spines[spine].set_color(COLORS_DICT["light_gray_paper"])

        # Add a shared legend for all subplots
        handles, legend_labels = axes[0].get_legend_handles_labels()
        if legend_labels:
            max_columns = min(len(labels), 8)
        else:
            max_columns = 1
        fig.legend(handles,
                   legend_labels,
                   loc='upper center',
                   fontsize=10,
                   frameon=False,
                   ncol=max_columns,
                   bbox_to_anchor=(0.5, -0.05))

        # Save plot
        os.makedirs(self.analysis_dir, exist_ok=True)
        plot_dir = os.path.join(self.analysis_dir,
                                f"plot_by_{x_axis.replace('_', '')}")
        os.makedirs(plot_dir, exist_ok=True)
        suffix = f"_stat_{stat}" if x_axis == "wind_speed" else ""
        plot_filename = f"{plot_name}{suffix}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        fig.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)

        if save_svg:
            plot_filename = f"{plot_name}{suffix}.svg"
            svg_path = os.path.join(plot_dir, plot_filename)
            plt.savefig(svg_path,
                        format='svg',
                        bbox_inches="tight",
                        pad_inches=0,
                        transparent=True)
        plt.close(fig)

    def plot_rotor_tower_plat_metrics(self,
                                      x_axis: str,
                                      stat: str = "mean",
                                      plot_labels: list = None,
                                      outputs_id: list = None,
                                      id_label: bool = False,
                                      save_svg: bool = False) -> None:
        """
        Plots rotor, tower, and platform metrics as a function of wind speed.

        Generates three separate figures:
        - Rotor performance metrics
        - Tower bending moments and accelerations
        - Platform translations and rotations

        Each figure contains multiple subplots arranged in a predefined grid.

        Args:
            x_axis (str): X-axis variable ("time" or "wind_speed").
            stat (str, optional): Statistical measure to compute ('mean' or
              'std'). Defaults to 'mean'.
            plot_labels (list, optional): Custom labels for each output.
            outputs_id (list, optional): IDs of specific outputs to include.
            id_label (bool, optional): If True, uses output ID as label.
              If False, uses Wind1VelX (wind speed). Defaults to True.
            save_svg (bool): If True, saves the plot as an SVG file.
        """

        # Configuration for each component (rotor, tower, platform)
        component_metrics = {
            "rotor": {
                "metrics": [
                    'blade_pitch', 'electrical_power', 'rotor_thrust',
                    'rotor_speed', 'rotor_torque'
                ],
                "ylabels": [
                    'Blade Pitch (deg)', 'Power Output (MW)', 'Thrust (MN)',
                    'Angular Velocity (rpm)', 'Torque (MN.m)'
                ],
                "grid_shape": (2, 3),
                "file_name": f"rotor_by_{x_axis.replace('_', '')}"
            },
            "tower": {
                "metrics": [
                    'tower_top_mfa', 'tower_top_mss', 'tower_bottom_mfa',
                    'tower_bottom_mss', 'tower_top_afa', 'tower_top_ass',
                    'tower_7_afa', 'tower_7_ass'
                ],
                "ylabels": [
                    'Tower Top\nBending Moment FA (kN.m)',
                    'Tower Top\nBending Moment SS (kN.m)',
                    'Tower Top\nAcceleration FA (m/s²)',
                    'Tower Top\nAcceleration SS (m/s²)',
                    'Tower Bottom\nBending Moment FA (kN.m)',
                    'Tower Bottom\nBending Moment SS (kN.m)',
                    'Tower Gage 7\nAcceleration FA (m/s²)',
                    'Tower Gage 7\nAcceleration SS (m/s²)'
                ],
                "grid_shape": (2, 4),
                "file_name": f"tower_by_{x_axis.replace('_', '')}"
            },
            "platform": {
                "metrics": [
                    'plat_surge', 'plat_sway', 'plat_heave', 'plat_roll',
                    'plat_pitch', 'plat_yaw'
                ],
                "ylabels": [
                    'Platform\nTranslation Surge (m)',
                    'Platform\nTranslation Sway (m)',
                    'Platform\nTranslation Heave (m)',
                    'Platform\nRotation Roll (deg)',
                    'Platform\nRotation Pitch (deg)',
                    'Platform\nRotation Yaw (deg)'
                ],
                "grid_shape": (2, 3),
                "file_name": f"plat_by_{x_axis.replace('_', '')}"
            }
        }

        # Generate plots for each component
        for _, config in component_metrics.items():
            plot_configs = {
                "metrics": config["metrics"],
                "ylabels": config["ylabels"],
                "xmin": None,
                "xmax": None,
                "ymin": None,
                "ymax": None,
                "grid_shape": config["grid_shape"],
                "plot_labels": plot_labels
            }
            self.plot_metrics(plot_configs=plot_configs,
                              outputs_id=outputs_id,
                              x_axis=x_axis,
                              stat=stat,
                              plot_name=config["file_name"],
                              id_label=id_label,
                              save_svg=save_svg)

    def plot_psd_heatmap_metric_by_windspeed(
            self,
            plot_configs: dict,
            rpm_by_wind_speed_filepath: str = None,
            save_csv: bool = False) -> None:
        """Plots a PSD heatmap of a metric, organized by wind speed.

        The function computes and visualizes the Power Spectral Density (PSD) 
        of a selected metric across different wind speeds, overlaying
        characteristic frequency multiples (e.g., 1P, 3P, 6P, 9P) derived from
        rotor speed.

        Args:
            plot_configs (dict): Configuration dictionary containing:
                - "metrics" (str): Name of the metric column for PSD 
                  computation.
                - "xmin", "xmax" (float, optional): Wind speed range.
                - "ymin", "ymax" (float, optional): Frequency range.
            rpm_by_wind_speed_filepath (str, optional): Path to CSV file
              containing  wind speed and rotor speed data. If not provided, mean
              metrics will be computed internally.
            save_csv (bool): If True, saves PSD matrix and rotor speed as CSVs.
        """
        if rpm_by_wind_speed_filepath == "":
            rpm_by_wind_speed_filepath = None

        # Extract plot parameters
        metric = plot_configs.get("metrics")
        xmin, xmax = plot_configs.get("xmin"), plot_configs.get("xmax")
        ymin, ymax = plot_configs.get("ymin"), plot_configs.get("ymax")

        # Compute PSD heatmap data
        frequencies, wind_speeds, psd_heatmap = (
            self.compute_psd_heatmap_by_windspeed(metric))

        # Load or compute rotor speed data
        if rpm_by_wind_speed_filepath is None:
            output_df = self.compute_mean_metrics()
        else:
            output_df = pd.read_csv(rpm_by_wind_speed_filepath)

        # Group rotor speed data by wind speed and compute mean values
        grouped = (output_df.groupby("Wind1VelX",
                                     as_index=False).mean(numeric_only=True))

        # Extract wind speed and corresponding average rotor speed
        wind_speed = grouped["Wind1VelX"]
        rpm = grouped["RotSpeed"]

        # Convert rotor speed to multiples of rotational frequency
        p1, p3, p6, p9 = rpm / 60, rpm * 3 / 60, rpm * 6 / 60, rpm * 9 / 60

        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Configure the figure and grid layout
        fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')
        ax.set_facecolor('white')

        # Plot heatmap
        f_min, f_max = frequencies.min(), frequencies.max()
        offset = 0.5 if not wind_speed[0].is_integer() else 0
        wind_speed_min, wind_speed_max = wind_speeds[0] - offset, wind_speeds[
            -1] + offset
        img = ax.imshow(np.log10(psd_heatmap),
                        cmap='jet',
                        origin='lower',
                        extent=[wind_speed_min, wind_speed_max, f_min, f_max],
                        aspect='auto',
                        interpolation='none')

        # Overlay characteristic frequency lines
        for freq, label in zip([p1, p3, p6, p9], ['1P', '3P', '6P', '9P']):
            ax.plot(wind_speed,
                    freq,
                    color='black',
                    linestyle='--',
                    linewidth=1,
                    label=label)
            ax.text(xmax * 0.95,
                    freq.iloc[-1],
                    f"{label}",
                    color='black',
                    fontsize=10,
                    verticalalignment='bottom')

        # Customize axes and appearance
        ax.set_xlabel('Wind speed (m/s)', fontsize=10, labelpad=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10, labelpad=10)
        ax.set_xlim(xmin + offset, xmax - offset)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(False)

        # Add a colorbar
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Log(PSD Amplitude)', fontsize=10, labelpad=10)
        cbar.ax.tick_params(axis='both',
                            which='major',
                            length=4,
                            width=0.7,
                            labelsize=10)

        # Hide unnecessary spines
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        # Save the plot
        os.makedirs(self.analysis_dir, exist_ok=True)
        plot_dir = os.path.join(self.analysis_dir, "plot_psd")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir, f"{metric.replace('_', '')}_psdheatmap_by_windspeed.png")
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)
        plt.close(fig)

        # Save CSV
        if save_csv:

            # Save PSD matrix
            psd_df = pd.DataFrame(np.log10(psd_heatmap),
                                  index=frequencies,
                                  columns=wind_speeds)
            psd_df.index.name = "Frequency [Hz]"
            psd_df.columns.name = "Wind Speed [m/s]"
            psd_df.to_csv(
                os.path.join(plot_dir,
                             f"{metric.replace('_', '')}_psd_matrix.csv"))

            # Save rotor speeds if the file does not exist
            rpm_df = grouped[["Wind1VelX", "RotSpeed"]]
            file_path = os.path.join(plot_dir, "rot_speed.csv")

            if not os.path.exists(file_path):
                rpm_df.to_csv(file_path, index=False)

    def plot_psd_heatmap_metrics_by_windspeed(
            self,
            plot_configs: dict,
            rpm_by_wind_speed_filepath: str = None) -> None:
        """Plots PSD heatmaps for multiple metrics, organized by wind speed.

        Iterates through multiple metrics provided in `plot_configs` and 
        generates a separate PSD heatmap for each.

        Args:
            plot_configs (dict): Configuration dictionary containing:
                - "metrics" (list[str], optional): List of metric names for PSD
                computation. Defaults to ["tower_bottom_mfa"].
                - "xmin" (float, optional): Minimum wind speed. Defaults to 3.
                - "xmax" (float, optional): Maximum wind speed. Defaults to 25.
                - "ymin" (float, optional): Minimum frequency. Defaults to None.
                - "ymax" (float, optional): Maximum frequency. Defaults to 1.75.
            rpm_by_wind_speed_filepath (str, optional): Path to CSV file
              containing wind speed and rotor speed data. If not provided, mean
              metrics will be computed internally.
        """
        # Set default values
        metrics = [
            'tower_bottom_mfa', 'tower_top_afa'
        ] if not plot_configs.get("metrics") else plot_configs.get("metrics")
        xmin = 3 if not plot_configs.get("xmin") else plot_configs.get("xmin")
        xmax = 25 if not plot_configs.get("xmax") else plot_configs.get("xmax")
        ymax = 1.75 if not plot_configs.get("ymax") else plot_configs.get(
            "ymax")

        for metric in metrics:

            plot_params = {
                "metrics": metric,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": plot_configs.get("ymin"),
                "ymax": ymax,
            }
            self.plot_psd_heatmap_metric_by_windspeed(
                plot_params, rpm_by_wind_speed_filepath, True)

    def plot_psd_by_frequency(self, plot_configs: dict) -> None:
        """
        Plots the PSD of selected metrics from OpenFAST outputs.

        Args:
            plot_configs (dict): Configuration dictionary with:
                - "metrics": List of metric names to plot.
                - "ylabels": Corresponding y-axis labels.
                - "plot_labels": Labels for the legend.
                - "grid_shape": Tuple (rows, cols) for subplot layout.
                - "xmin", "xmax", "ymin", "ymax": Axis limits.
        """
        # Extract plot parameters
        metrics = plot_configs.get("metrics")
        ylabels = plot_configs.get("ylabels", None)
        plot_labels = plot_configs.get("plot_labels", None)
        xmin = plot_configs.get("xmin") or [None]
        xmax = plot_configs.get("xmax") or [None]
        ymin = plot_configs.get("ymin") or [None]
        ymax = plot_configs.get("ymax") or [None]
        metrics = plot_configs.get("metrics", [])
        numplots = len(metrics)
        xmin = xmin if xmin and len(xmin) == numplots else [xmin[0]] * numplots
        xmax = xmax if xmax and len(xmax) == numplots else [xmax[0]] * numplots
        ymin = ymin if ymin and len(ymin) == numplots else [ymin[0]] * numplots
        ymax = ymax if ymax and len(ymax) == numplots else [ymax[0]] * numplots

        # Compute PSD values for each metric
        (frequencies_list_metrics, psd_values_list_metrics,
         labels) = self.compute_psd_metrics(metrics)

        # Extract grid shape and convert to integers
        grid_shape = plot_configs.get("grid_shape")
        rows, cols = map(int, grid_shape)

        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Configure the figure and grid layout
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.4)

        # Define subplots
        axes = []
        for i in range(numplots):
            row, col = divmod(i, cols)  # Compute row and column index
            ax = plt.subplot(gs[row, col], facecolor='white')
            axes.append(ax)

        # Iterate over subplots to plot data
        for (ax, frequencies_list, psd_values_list, ylabel, xlim,
             ylim) in zip(axes, frequencies_list_metrics,
                          psd_values_list_metrics, ylabels, zip(xmin, xmax),
                          zip(ymin, ymax)):

            for i, (frequencies, psd_values, label) in enumerate(
                    zip(frequencies_list, psd_values_list, labels)):
                label = plot_labels[i] if plot_labels is not None else label

                ax.semilogy(
                    frequencies,
                    psd_values,
                    '-',
                    linewidth=1.2,
                    markersize=4,
                    label=label,
                    alpha=0.9,
                )

            # Hide unnecessary spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)

            ax.set_ylabel(ylabel, fontsize=10, labelpad=10)
            ax.set_xlim(xlim or axes[0].get_xlim())
            ax.set_ylim(ylim)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

            if ax in axes[-cols:]:
                ax.set_xlabel('Frequency (Hz)', fontsize=10, labelpad=10)
            else:
                ax.tick_params(axis='x', labelbottom=False)

            # Hide unnecessary spines
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)

        # Add a shared legend for all subplots
        handles, legend_labels = axes[0].get_legend_handles_labels()
        if legend_labels:
            max_columns = min(len(labels), 3)
        else:
            max_columns = 1
        fig.legend(
            handles,
            legend_labels,
            loc='upper center',
            fontsize=10,
            frameon=False,
            ncol=max_columns,
            bbox_to_anchor=(0.5, 0.0),
        )

        # Save the plot
        os.makedirs(self.analysis_dir, exist_ok=True)
        plot_dir = os.path.join(self.analysis_dir, "plot_psd")
        os.makedirs(plot_dir, exist_ok=True)
        metrics_str = "_".join([metric.replace("_", "") for metric in metrics])
        plot_filename = f'{metrics_str}_psd_by_frequency.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        fig.savefig(plot_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)
        plt.close(fig)


class OpenFASTDatasetSplitByWindWave:
    """Splits OpenFAST simulation datasets based on wind and wave."""

    def __init__(self, dataset_csv_path: str, source_root: str,
                 destination_root: str, wind_combinations: list[int],
                 wave_combinations: list[int]):
        """
        Initializes the dataset splitter for OpenFAST simulations.

        Args:
            dataset_csv_path (str): Path to the dataset CSV file.
            source_root (str): Root directory containing simulation folders.
            destination_root (str): Destination base folder to copy filtered
              datasets.
            wind_combinations (list[int]): Desired wind speed combinations
              (e.g., [22, 8, 4, 2]).
            wave_combinations (list[int]): Desired wave combinations
              (e.g., [7, 4, 3, 2]).
        """
        self.dataset_csv_path = dataset_csv_path
        self.source_root = source_root
        self.destination_root = destination_root
        self.wind_combinations = wind_combinations
        self.wave_combinations = wave_combinations

        self.df = pd.read_csv(dataset_csv_path)
        self.df["wind_speed"] = self.df["file_name_bts"].str.extract(
            r"W(\d{4})")[0].astype(float) / 100
        self.df["seed"] = self.df["file_name_bts"].str.extract(r"_(S\d+)\.bts$")

    def split_dataset(self):
        """
        Splits the dataset into valid combinations and copies the corresponding
        simulation folders.
        """
        for comb_wind in self.wind_combinations:
            for comb_waves in self.wave_combinations:
                if self._can_support_combination(comb_wind, comb_waves):
                    continue

                sim_ids = self._get_selected_sim_ids(comb_wind, comb_waves)
                folder_name = f"wind{comb_wind}_waves{comb_waves}x{comb_waves}"
                destination = os.path.join(self.destination_root, folder_name)
                self._filter_and_copy(sim_ids, destination)

    def _can_support_combination(self, comb_wind: int, comb_waves: int) -> bool:
        """
        Verifies if all seeds have enough wind and wave data to support the
        combination.

        Args:
            comb_wind (int): Number of wind speeds desired.
            comb_waves (int): Number of wave conditions (hs and tp) desired.

        Returns:
            bool: True if supported by all seeds, False otherwise.
        """
        for _, group_ws in self.df.groupby("seed"):
            wind_speeds = group_ws["wind_speed"].unique()

            if len(wind_speeds) != comb_wind:
                return False

            for ws in wind_speeds:
                group_ws_ws = group_ws[group_ws["wind_speed"] == ws]
                wave_hs = group_ws_ws["wave_hs"].unique()
                if len(wave_hs) != comb_waves:
                    return False

                for hs in wave_hs:
                    group_ws_hs = group_ws_ws[group_ws_ws["wave_hs"] == hs]
                    wave_tp = group_ws_hs["wave_tp"].unique()
                    if len(wave_tp) != comb_waves:
                        return False

        return True

    def _get_selected_sim_ids(self, combinations_wind: int,
                              combination_waves: int) -> list[str]:
        selected_sim_ids = []

        for _, group_ws in self.df.groupby("seed"):
            unique_ws = sorted(group_ws["wind_speed"].unique())
            ws_selected = self._select_evenly_spaced(unique_ws,
                                                     combinations_wind)

            for ws in ws_selected:
                group_ws_ws = group_ws[group_ws["wind_speed"] == ws]
                unique_hs = sorted(group_ws_ws["wave_hs"].unique())
                hs_selected = self._select_evenly_spaced(
                    unique_hs, combination_waves)

                for hs in hs_selected:
                    group_ws_hs = group_ws_ws[group_ws_ws["wave_hs"] == hs]
                    unique_tp = sorted(group_ws_hs["wave_tp"].unique())
                    tp_selected = self._select_evenly_spaced(
                        unique_tp, combination_waves)

                    group_tp = group_ws_hs[group_ws_hs["wave_tp"].isin(
                        tp_selected)]
                    selected_sim_ids.extend(
                        group_tp["sim_id"].astype(str).tolist())

        return list(set(selected_sim_ids))

    def _select_evenly_spaced(self, values: list[float], n: int) -> list[float]:
        step = (len(values) - 1) / (n - 1)
        indices = [round(i * step) for i in range(n)]
        return [values[i] for i in indices]

    def _filter_and_copy(self, selected_sim_ids: list[str], destination: str):
        os.makedirs(destination, exist_ok=True)
        df_filtered = self.df[self.df["sim_id"].astype(str).isin(
            selected_sim_ids)]

        for sim_id in tqdm(
                selected_sim_ids,
                desc=f"📦 Copiando para {os.path.basename(destination)}"):
            for item in os.listdir(self.source_root):
                if item.startswith(sim_id + "_") and os.path.isdir(
                        os.path.join(self.source_root, item)):
                    src_path = os.path.join(self.source_root, item)
                    dst_path = os.path.join(destination, item)
                    if not os.path.exists(dst_path):
                        shutil.copytree(src_path, dst_path)
                    break

        # Save filtered CSV
        name, ext = os.path.splitext(os.path.basename(self.dataset_csv_path))
        df_filtered.to_csv(os.path.join(destination, f"{name}_filtered{ext}"),
                           index=False)


class TowerPSDComparisonAnalysis:
    """
    Compare PSD heatmaps of a tower metric between tower designs.

    Attributes:
        comparison_dir (str): Directory to save the comparison plots and
          outputs.
        output_dir (str): Subdirectory within comparison_dir for storing
            comparison results.
        psd_dirs (List[str]): List of directories containing PSD matrices and
            rotor speed CSV files for each tower design.
        metric (str): Metric used for PSD comparison (underscores are removed
          for file lookup).
        psd (List[np.ndarray]): List of PSD matrices for each tower design.
        rot (List[pd.DataFrame]): List of rotor speed DataFrames for each
            tower design.
    """

    def __init__(self,
                 comparison_dir: str,
                 psd_dirs: List[str],
                 metric: str = "tower_bottom_mfa") -> None:
        """
        Initializes the TowerPSDComparisonAnalysis object.

        Args:
            comparison_dir (str): Directory where comparison plots will be
              saved.
            psd_dirs (List[str]): List of directories containing PSD matrices
                and rotor speed CSV files for each tower design. 
            metric (str): Name of the metric (underscores will be removed for
              file lookup).
        """
        self.metric = metric.replace("_", "")
        self.comparison_dir = comparison_dir
        self.output_dir = os.path.join(self.comparison_dir,
                                       "psd_comparison_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.psd_dirs = psd_dirs

        # Load data
        self._get_data()

    def _get_data(self) -> None:
        """
        Loads PSD matrices and rotor speed data from CSV files.
        """
        self.psd = []
        self.rot = []
        for psd_dir in self.psd_dirs:
            psd = pd.read_csv(os.path.join(psd_dir,
                                           f"{self.metric}_psd_matrix.csv"),
                              skiprows=1,
                              header=None).values
            rot = pd.read_csv(os.path.join(psd_dir, "rot_speed.csv"))
            self.psd.append(psd)
            self.rot.append(rot)

    def compare_psds(
        self,
        plot_configs: dict,
        save_svg: bool = False,
    ) -> None:
        """
        Generate and save a comparison plot of PSD heatmaps.

        The function compares the Power Spectral Density (PSD) of a given metric
        across designs, plotted as heatmaps over wind speed and frequency.
        Characteristic frequencies (1P, 3P, 6P, 9P) based on rotor speed are
        overlaid.

        Args:
            plot_configs (dict): Dictionary with optional plot limits:
                - "xmin" (float): Minimum wind speed.
                - "xmax" (float): Maximum wind speed.
                - "ymin" (float): Minimum frequency (Hz).
                - "ymax" (float): Maximum frequency (Hz).
                - "plot_labels" (list of str, optional): Custom labels for the
                  two designs. Default is ["Ref Tower", "New Tower"].
            save_svg (bool): If True, saves the plot as an SVG file.
        """
        # Extract plot parameters
        xmin = 3 if not plot_configs.get("xmin") else plot_configs.get("xmin")
        xmax = 25 if not plot_configs.get("xmax") else plot_configs.get("xmax")
        ymax = 1.75 if not plot_configs.get("ymax") else plot_configs.get(
            "ymax")
        ymin = 0 if not plot_configs.get("ymin") else plot_configs.get("ymin")
        plot_labels = ["Ref Tower", "New Tower"] if not plot_configs.get(
            "plot_labels") else plot_configs.get("plot_labels")

        # Set global plot style
        plt.style.use('fivethirtyeight')

        # Create the figure and axes with white background
        num_subplots = len(self.psd) + 1
        width_ratios = [1] * len(self.psd) + [0.05]
        fig, axs = plt.subplots(1,
                                num_subplots,
                                figsize=(8 * (num_subplots - 1) - num_subplots,
                                         10),
                                facecolor='white',
                                constrained_layout=True,
                                gridspec_kw={"width_ratios": width_ratios})

        # Determine global vmin and vmax for consistent color scaling
        vmin = min(psd[:, 1:].min() for psd in self.psd)
        vmax = max(psd[:, 1:].max() for psd in self.psd)

        for ax, psd, rot, label in zip(axs[0:(num_subplots - 1)], self.psd,
                                       self.rot, plot_labels):
            f_min, f_max = psd[:, 0][0], psd[:, 0][-1]
            wind_speeds = rot["Wind1VelX"].values
            offset = 0.5 if not wind_speeds[0].is_integer() else 0
            wind_speed_min, wind_speed_max = wind_speeds[
                0] - offset, wind_speeds[-1] + offset
            img = ax.imshow(
                psd[:, 1:],
                origin='lower',
                aspect='auto',
                cmap="jet",
                extent=[wind_speed_min, wind_speed_max, f_min, f_max],
                interpolation='none',
                vmin=vmin,
                vmax=vmax)

            # Extract wind speed and corresponding average rotor speed
            rpm = rot["RotSpeed"].values
            wind_speed = rot["Wind1VelX"].values

            # Convert rotor speed to multiples of rotational frequency
            p1, p3, p6, p9 = rpm / 60, rpm * 3 / 60, rpm * 6 / 60, rpm * 9 / 60

            # Overlay characteristic frequency lines
            for freq, label_freq in zip([p1, p3, p6, p9],
                                        ['1P', '3P', '6P', '9P']):
                ax.plot(wind_speed,
                        freq,
                        color='black',
                        linestyle='--',
                        linewidth=1,
                        label=label_freq)
                ax.text(wind_speeds[-1] * 0.95,
                        freq[-1],
                        f"{label_freq}",
                        color='black',
                        fontsize=10,
                        verticalalignment='bottom')

            # Customize axes and appearance
            ax.set_title(label)
            ax.set_xlabel("Wind Speed (m/s)", fontsize=10, labelpad=10)
            ax.grid(False)
            ax.tick_params(axis='both',
                           which='major',
                           length=4,
                           width=1,
                           labelsize=10)
            ax.set_xlim(xmin + offset, xmax - offset)
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(np.arange(ymin, ymax + 0.01, 0.25))
            ax.set_xticks(np.arange(xmin, xmax + 1, 1))

        # Hide unnecessary spines
        for ax in axs:
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)

        # Customize axes and appearance
        axs[0].set_ylabel("Frequency (Hz)", fontsize=10, labelpad=10)
        for i, ax in enumerate(axs[:-1]):
            if i != 0:
                ax.set_yticklabels([])

        # Remove all content from the third axis (colorbar axis)
        axs[num_subplots - 1].axis('off')

        # Add a colorbar
        cbar = fig.colorbar(img,
                            ax=axs[num_subplots - 2],
                            orientation='vertical')
        cbar.set_label('Log(PSD Amplitude)', fontsize=10, labelpad=10)
        cbar.ax.tick_params(axis='both',
                            which='major',
                            length=4,
                            width=0.7,
                            labelsize=10)
        cbar.outline.set_visible(False)

        # Save the plot
        plot_path = os.path.join(
            self.output_dir,
            f"{self.metric}_psdheatmap_by_windspeed_comparison.png")
        plt.savefig(plot_path,
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                    transparent=True)

        if save_svg:
            svg_path = os.path.join(
                self.output_dir,
                f"{self.metric}_psdheatmap_by_windspeed_comparison.svg")
            plt.savefig(svg_path,
                        format='svg',
                        bbox_inches="tight",
                        pad_inches=0,
                        transparent=True)
        plt.close(fig)
