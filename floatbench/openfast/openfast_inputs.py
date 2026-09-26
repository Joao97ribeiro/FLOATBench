# pylint: skip-file
# Released post-processing pipeline of the FLOATBench labels, kept verbatim
# (only the colour import points to floatbench.colors).
# pylint: disable=import-error
# pylint: disable=duplicate-code
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
"""Analysis of OpenFAST Input files."""

import os
import glob
import logging
from typing import Dict, Optional

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

ELASTO_FILENAME = 'IEA-22-280-RWT-Semi_ElastoDyn.dat'
ELASTO_BLADE_FILENAME = 'IEA-22-280-RWT_ElastoDyn_blade.dat'
ELASTO_TOWER_FILENAME = 'IEA-22-280-RWT-Semi_ElastoDyn_tower.dat'
HYDRO_FILENAME = 'IEA-22-280-RWT-Semi_HydroDyn_PotMod.dat'

OPENFAST_PROPERTIES = {
    # Blade-related properties
    'blade_tip_radius': 'TipRad',
    'blade_precone_angle': 'PreCone(1)',

    # Hub-related properties
    'hub_radius': 'HubRad',
    'hub_mass': 'HubMass',
    'shaft_tilt_angle': 'ShftTilt',
    'rotor_overhang': 'OverHang',
    'tower_to_shaft_distance': 'Twr2Shft',

    # Nacelle-related properties
    'nacelle_cmx': 'NacCMxn',
    'nacelle_cmz': 'NacCMzn',
    'nacelle_mass': 'NacMass',

    # Tower-related properties
    'tower_base_z': 'TowerBsHt',
    'tower_top_z': 'TowerHt',

    # Platform-related properties
    'platform_cmx': 'PtfmCMxt',
    'platform_cmz': 'PtfmCMzt',
    'platform_mass': 'PtfmMass',
}

COLORS = [
    "midnightblue",  # Azul escuro intenso
    "firebrick",  # Vermelho terroso
    "darkgreen",  # Verde escuro
    "goldenrod",  # Amarelo queimado
    "darkorange",  # Laranja forte
    "slategray",  # Cinza azulado
    "indigo",  # Roxo escuro
    "teal",  # Azul petróleo
    "crimson",  # Vermelho vibrante
    "steelblue",  # Azul acinzentado
]


class ElastoDynInputLoader:
    """Loads and processes data from an OpenFAST ElastoDyn input file."""

    def __init__(self, input_dir: str, input_filename: str = ELASTO_FILENAME):
        """
        Initializes an ElastoDynInputLoader instance.

        Args:
            input_dir (str): Directory containing the OpenFAST ElastoDyn input
              file.
            input_filename (str, optional): The name of the input file.
              Defaults to 'ELASTO_FILENAME'.
        """
        self.input_dir = input_dir
        self.input_filename = input_filename
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the OpenFAST ElastoDyn input file and extracts the relevant data.

        Returns:
            pd.DataFrame: Data containing the relevant properties.
        """
        # Locate the input file
        file_path = glob.glob(os.path.join(self.input_dir,
                                           f"**/*{self.input_filename}"),
                              recursive=True)[0]

        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Dictionary to store extracted values
        extracted_values = {key: None for key in OPENFAST_PROPERTIES}

        # Loop through the file lines and extract values for each property
        for line in lines:
            for key, openfast_key in OPENFAST_PROPERTIES.items():
                if openfast_key in line:
                    extracted_values[key] = float(line.split()[0])

        return pd.DataFrame([extracted_values])


class ElastoDynBladeInputLoader:
    """Loads and processes data from an OpenFAST ElastoDyn blade input file."""

    def __init__(self,
                 input_dir: str,
                 input_filename: str = ELASTO_BLADE_FILENAME):
        """
        Initializes an ElastoDynBladeInputLoader instance.

        Args:
            input_dir (str): Directory containing the OpenFAST ElastoDyn blade
              input file.
            input_filename (str, optional): The name of the input file.
              Defaults to 'ELASTO_BLADE_FILENAME'.
        """
        self.input_dir = input_dir
        self.input_filename = input_filename
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the OpenFAST ElastoDyn blade input file and extracts the data.

        Returns:
            pd.DataFrame: Data containing the blade properties.
        """
        # Locate the input file
        file_path = glob.glob(os.path.join(self.input_dir,
                                           f"**/*{self.input_filename}"),
                              recursive=True)[0]

        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        data = []
        start_idx = None

        # Locate the start of the blade properties section
        for i, line in enumerate(lines):
            if "DISTRIBUTED BLADE PROPERTIES" in line:
                start_idx = i + 3
                headers = lines[i + 1].split()
                break

        # Parse the data into numeric values
        for line in lines[start_idx:]:
            if "BLADE MODE SHAPES" in line:
                break
            values = line.split()
            numeric_values = [float(val) for val in values]
            if len(numeric_values) == len(headers):
                data.append(numeric_values)

        # Convert the extracted data into a Pandas DataFrame
        column_mapping = {
            "BlFract": "blade_fract",
            "PitchAxis": "pitch_axis",
            "StrcTwst": "structural_twist",
            "BMassDen": "blade_mass_den",
            "FlpStff": "flap_stiffness",
            "EdgStff": "edge_stiffness"
        }
        data_df = pd.DataFrame(data, columns=headers)
        data_df.rename(columns=column_mapping, inplace=True)

        return data_df


class ElastoDynTowerInputLoader:
    """Loads and processes data from an OpenFAST ElastoDyn tower input file."""

    def __init__(self,
                 input_dir: str,
                 input_filename: str = ELASTO_TOWER_FILENAME):
        """
        Initializes an ElastoDynTowerInputLoader instance.

        Args:
            input_dir (str): Directory containing the OpenFAST ElastoDyn tower
              input file.
            input_filename (str, optional): The name of the input file.
              Defaults to 'ELASTO_TOWER_FILENAME'.
        """
        self.input_dir = input_dir
        self.input_filename = input_filename
        self.data = self._load_data()
        self.mode_shapes = self._load_mode_shapes()

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the OpenFAST ElastoDyn tower input file and extracts the data.

        Returns:
            pd.DataFrame: Data containing the tower properties.
        """
        # Locate the input file
        file_path = glob.glob(os.path.join(self.input_dir,
                                           f"**/*{self.input_filename}"),
                              recursive=True)[0]

        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        data = []
        start_idx = None

        # Locate the start of the tower properties section
        for i, line in enumerate(lines):
            if "DISTRIBUTED TOWER PROPERTIES" in line:
                start_idx = i + 3
                headers = lines[i + 1].split()
                break

        # Parse the data into numeric values
        for line in lines[start_idx:]:
            if "TOWER FORE-AFT MODE SHAPES" in line:
                break
            values = line.split()
            numeric_values = [float(val) for val in values]
            if len(numeric_values) == len(headers):
                data.append(numeric_values)

        # Convert the extracted data into a Pandas DataFrame
        column_mapping = {
            "HtFract": "height_fract",
            "TMassDen": "tower_mass_den",
            "TwFAStif": "fa_stiffness",
            "TwSSStif": "ss_stiffness"
        }
        data_df = pd.DataFrame(data, columns=headers)
        data_df.rename(columns=column_mapping, inplace=True)

        return data_df

    def _load_mode_shapes(self) -> dict[str, list[np.ndarray]]:
        """
        Load vibration mode shape coefficients (x^2 to x^6).

        Returns:
            dict: Dictionary with modal shape coefficients grouped into:
                {
                    'fore_aft': [array_mode1, array_mode2],
                    'side_side': [array_mode1, array_mode2]
                }
        """
        # Locate the input file
        file_path = glob.glob(os.path.join(self.input_dir,
                                           f"**/*{self.input_filename}"),
                              recursive=True)[0]

        # Initialize container
        raw_shapes = {
            "TwFAM1Sh": [None] * 5,
            "TwFAM2Sh": [None] * 5,
            "TwSSM1Sh": [None] * 5,
            "TwSSM2Sh": [None] * 5,
        }

        # Regex to match: value, label, degree
        pattern = re.compile(r'\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+'
                             r'(TwFAM1Sh|TwFAM2Sh|TwSSM1Sh|TwSSM2Sh)\((\d)\)')

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    value, key, degree = match.groups()
                    idx = int(degree) - 2
                    if 0 <= idx < 5:
                        raw_shapes[key][idx] = float(value)

        # Output grouped by physical meaning
        mode_shapes = {
            "fore_aft": [
                np.array(raw_shapes["TwFAM1Sh"]),
                np.array(raw_shapes["TwFAM2Sh"])
            ],
            "side_side": [
                np.array(raw_shapes["TwSSM1Sh"]),
                np.array(raw_shapes["TwSSM2Sh"])
            ]
        }

        return mode_shapes

    def update_tower_file(self,
                          new_profile: Optional[Dict[str, np.ndarray]] = None,
                          new_mode_shapes: Optional[Dict[str,
                                                         np.ndarray]] = None,
                          new_dir: str = None,
                          new_filename: str = None,
                          normalize_mode_coeffs: bool = False) -> None:
        """Update the ElastoDyn tower file.

        Args:
            new_profile: Dictionary with updated structural tower properties. 
              May include:
                - 'mass_den': array of mass density values.
                - 'foreaft_stff': array of fore-aft stiffness values.
                - 'sideside_stff': array of side-to-side stiffness values.
                - 'sec_loc' (optional): array of section locations (height
                  fractions).
            new_mode_shapes: Dictionary with updated mode shape coefficients:
                - 'fore_aft': List of two arrays (mode 1 and mode 2), each with
                  5 coefficients.
                - 'side_side': Same structure for side-to-side modes.
            new_dir: Directory to save the updated ElastoDyn file.
            new_filename: Name of the file to save. If None, defaults to
              `ELASTO_TOWER_FILENAME`.
            normalize_mode_coeffs: If True, normalize each set of mode shape
              coefficients so that their sum equals 1.  
        """
        # Read original file lines
        file_path = glob.glob(os.path.join(self.input_dir,
                                           f"**/*{self.input_filename}"),
                              recursive=True)[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Update structural profile
        if new_profile is not None:
            data = self.data.copy()
            data["height_fract"] = new_profile.get("sec_loc",
                                                   self.data["height_fract"])
            data["tower_mass_den"] = new_profile.get(
                "mass_den", self.data["tower_mass_den"])
            data["fa_stiffness"] = new_profile.get("foreaft_stff",
                                                   self.data["fa_stiffness"])
            data["ss_stiffness"] = new_profile.get("sideside_stff",
                                                   self.data["ss_stiffness"])

            for i, line in enumerate(lines):
                if "DISTRIBUTED TOWER PROPERTIES" in line:
                    start_idx = i + 3
                    break
            updated_lines = [
                (f" {row.height_fract:.15e}  {row.tower_mass_den:.15e}  "
                 f"{row.fa_stiffness:.15e}  {row.ss_stiffness:.15e}\n")
                for row in data.itertuples(index=False)
            ]
            lines[start_idx:start_idx + len(updated_lines)] = updated_lines

        # Update mode shapes
        if new_mode_shapes is not None:
            for key in ['fore_aft', 'side_side']:
                for i, mode in enumerate(new_mode_shapes[key]):
                    total = np.sum(mode)
                    if not np.isclose(total, 1.0):
                        logging.warning(
                            "Mode '%s' %d: coefficients sum = %.6f "
                            "(not normalized)", key, i + 1, total)
                    else:
                        logging.info(
                            "Mode '%s' %d: coefficients already normalized "
                            "(sum = %.6f)", key, i + 1, total)

                if normalize_mode_coeffs:
                    new_mode_shapes[key] = [
                        mode / np.sum(mode)
                        if not np.isclose(np.sum(mode), 1.0) else mode
                        for mode in new_mode_shapes[key]
                    ]

            def format_modal_block(mode_data: np.ndarray,
                                   prefix: str,
                                   precision: int = 16) -> list[str]:
                """
                Format modal shape coefficients into FAST-compatible lines.

                Args:
                    mode_data (np.ndarray): Array of shape (n_modes, n_coeffs).
                    prefix (str): Prefix for the coefficient labels (e.g., 
                      "TwFAM" or "TwSSM").
                    precision (int): Decimal precision for the coefficient
                      formatting.

                Returns:
                    List[str]: Formatted strings for inclusion in the FAST input
                      file.
                """
                formatted_lines = []
                for mode_idx, coeffs in enumerate(mode_data):
                    for poly_idx, value in enumerate(coeffs):
                        label = f"{prefix}{mode_idx + 1}Sh({poly_idx + 2})"
                        if poly_idx == 0:
                            comment = (f"Mode {mode_idx + 1}, "
                                       f"coefficient of x^{poly_idx + 2} term")
                        else:
                            comment = ("      , "
                                       f"coefficient of x^{poly_idx + 2} term")
                        line = f"{value:<22.{precision}f} {label} - {comment}\n"
                        formatted_lines.append(line)
                return formatted_lines

            fore_aft = np.array(new_mode_shapes.get("fore_aft"))[:2]
            side_side = np.array(new_mode_shapes.get("side_side"))[:2]

            if fore_aft is not None:
                fa_start = next(i for i, l in enumerate(lines)
                                if "TOWER FORE-AFT MODE SHAPES" in l) + 1
                lines[fa_start:fa_start +
                      5 * fore_aft.shape[0]] = format_modal_block(
                          fore_aft, "TwFAM")

            if side_side is not None:
                ss_start = next(i for i, l in enumerate(lines)
                                if "TOWER SIDE-TO-SIDE MODE SHAPES" in l) + 1
                lines[ss_start:ss_start +
                      5 * side_side.shape[0]] = format_modal_block(
                          side_side, "TwSSM")

        # Save updated file
        output_path = os.path.join(new_dir, new_filename or
                                   ELASTO_TOWER_FILENAME)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


class HydroDynInputLoader:
    """Loads and processes data from an OpenFAST HydroDyn input file."""

    def __init__(self, input_dir: str, input_filename: str = HYDRO_FILENAME):
        """
        Initializes an HydroDynInputLoader instance.

        Args:
            input_dir (str): Directory containing the OpenFAST HydroDyn input
              file.
            input_filename (str, optional): The name of the input file.
              Defaults to 'HYDRO_FILENAME'.
        """
        self.input_dir = input_dir
        self.input_filename = input_filename
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the OpenFAST HydroDyn input file and extracts the relevant data.

        Returns:
            pd.DataFrame: Data containing the relevant properties.
        """
        # Locate the input file
        file_path = glob.glob(os.path.join(self.input_dir,
                                           f"**/*{self.input_filename}"),
                              recursive=True)[0]

        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        data = {}

        # Locate and extract key values
        for line in lines:
            values = line.split()

            if "! Btm of upwind col" in line:
                data["col_x"] = float(values[1])
                data["col_base_z"] = float(values[3])

            elif "! Top of upwind col" in line:
                data["col_top_z"] = float(values[3])

            elif "! Upwind col" in line and len(values) == 6:
                data["col_out_diam"] = float(values[1])
                data["col_thickness"] = float(values[2])

            elif "! Upwind col" in line and len(values) == 11:
                data["col_div_size"] = float(values[5])
                break

        # Convert the extracted data into a Pandas DataFrame
        data_df = pd.DataFrame([data])

        return data_df


class BladeSystem:
    """Loads and processes properties of 3 blades from OpenFAST input files."""

    def __init__(self, input_dir: str):
        """
        Initializes a BladeSystem instance.

        Args:
            input_dir (str): Directory containing OpenFAST input files.
        """
        self.input_dir = input_dir

        # Load blade-related data
        self.elasto_data = ElastoDynInputLoader(input_dir).data
        self.blade_data = ElastoDynBladeInputLoader(input_dir).data

        # Extract blade properties
        self.tip_radius = self.elasto_data["blade_tip_radius"].iloc[0]
        self.pre_cone_angle = self.elasto_data["blade_precone_angle"].iloc[0]

        # Compute blade length, mass, and center of mass
        self.blade_length = self._compute_blade_length()
        self.balde_mass = self._compute_blade_mass()
        self.blade_local_cm = self._compute_blade_local_center_of_mass()

        # Compute blade system mass, and center of mass
        self.mass = 3 * self.balde_mass
        self.cmx, self.cmz = self._compute_center_of_mass()

    def _compute_blade_length(self) -> float:
        """
        Computes the blade length.

        Returns:
            float: Blade length in meters.
        """
        hub = Hub(self.input_dir)
        return self.tip_radius - hub.radius

    def _compute_blade_mass(self):
        """
        Computes the total mass of a blade using numerical integration.

        Returns:
            float: Blade mass in kilograms.
        """
        blade_fraction = self.blade_data["blade_fract"].values
        mass_density = self.blade_data["blade_mass_den"].values
        blade_positions = blade_fraction * self.blade_length

        return np.trapezoid(mass_density, blade_positions)

    def _compute_blade_local_center_of_mass(self) -> float:
        """
        Computes the local center of mass along the blade length.

        Returns:
            float: Local center of mass position along the blade in meters.
        """
        blade_fraction = self.blade_data["blade_fract"].values
        mass_density = self.blade_data["blade_mass_den"].values
        blade_positions = blade_fraction * self.blade_length

        return np.trapezoid(blade_positions * mass_density,
                            blade_positions) / self.balde_mass

    def _compute_center_of_mass(self) -> tuple:
        """
        Computes the global center of mass (CM) of the blade system.

        Returns:
            tuple: (cmx, cmz), where:
                - cmx (float): X coordinate of the CM in meters (m).
                - cmz (float): Z coordinate of the CM in meters (m).
        """
        tower = Tower(self.input_dir)
        hub = Hub(self.input_dir)

        cmx = ((self.blade_local_cm + hub.radius) *
               np.sin(np.radians(self.pre_cone_angle)) + hub.overhang) * np.cos(
                   np.radians(hub.shaft_tilt))
        cmz = (tower.top_z + hub.tower_to_shaft +
               ((self.blade_local_cm + hub.radius) *
                np.sin(np.radians(self.pre_cone_angle)) + hub.overhang) *
               np.sin(np.radians(hub.shaft_tilt)))

        return cmx, cmz


class Nacelle:
    """Loads and processes nacelle properties from OpenFAST input files."""

    def __init__(self, input_dir: str):
        """
        Initializes a Nacelle instance.

        Args:
            input_dir (str): Directory containing OpenFAST input files.
        """
        self.input_dir = input_dir

        # Load nacelle-related data from ElastoDyn input
        self.elasto_data = ElastoDynInputLoader(input_dir).data

        # Extract nacelle properties
        self.local_cmx = self.elasto_data["nacelle_cmx"].iloc[0]
        self.local_cmz = self.elasto_data["nacelle_cmz"].iloc[0]
        self.mass = self.elasto_data["nacelle_mass"].iloc[0]

        # Compute global center of mass
        self.cmx, self.cmz = self._compute_center_of_mass()

    def _compute_center_of_mass(self) -> tuple:
        """
        Computes the global center of mass (CM) coordinates for the nacelle.

        Returns:
            tuple: (cmx, cmz), where:
                - cmx (float): X coordinate of the CM in meters (m).
                - cmz (float): Z coordinate of the CM in meters (m).
        """
        tower = Tower(self.input_dir)

        cmx = self.local_cmx
        cmz = tower.base_z + tower.height + self.local_cmz

        return cmx, cmz


class Hub:
    """Loads and processes hub properties from OpenFAST input files."""

    def __init__(self, input_dir: str):
        """
        Initializes a Hub instance.

        Args:
            input_dir (str): Directory containing OpenFAST input files.
        """
        self.input_dir = input_dir

        # Load hub-related data from ElastoDyn input
        self.elasto_data = ElastoDynInputLoader(input_dir).data

        # Extract hub properties
        self.radius = self.elasto_data["hub_radius"].iloc[0]
        self.shaft_tilt = self.elasto_data["shaft_tilt_angle"].iloc[0]
        self.overhang = self.elasto_data["rotor_overhang"].iloc[0]
        self.tower_to_shaft = self.elasto_data["tower_to_shaft_distance"].iloc[
            0]
        self.mass = self.elasto_data["hub_mass"].iloc[0]

        # Compute center of mass
        self.cmx, self.cmz = self._compute_center_of_mass()

    def _compute_center_of_mass(self) -> tuple:
        """
        Computes the global center of mass (CM) coordinates for the hub.

        Returns:
            tuple: (cmx, cmz), where:
                - cmx (float): X coordinate of the CM in meters (m).
                - cmz (float): Z coordinate of the CM in meters (m).
        """
        tower = Tower(self.input_dir)

        cmx = self.overhang * np.cos(np.radians(self.shaft_tilt))
        cmz = (tower.top_z + self.tower_to_shaft +
               np.abs(self.overhang * np.sin(np.radians(self.shaft_tilt))))

        return cmx, cmz


class Tower:
    """Loads and processes tower properties from OpenFAST input files."""

    def __init__(self, input_dir: str):
        """
        Initializes a Tower instance.

        Args:
            input_dir (str): Directory containing OpenFAST input files.
        """
        self.input_dir = input_dir

        # Load tower-related data
        self.elasto_data = ElastoDynInputLoader(input_dir).data
        self.tower_data = ElastoDynTowerInputLoader(input_dir).data

        # Extract tower properties
        self.top_z = self.elasto_data["tower_top_z"].iloc[0]
        self.base_z = self.elasto_data["tower_base_z"].iloc[0]
        self.height = self.top_z - self.base_z

        # Compute mass and center of mass
        self.mass = self._compute_mass()
        self.local_cmz = self._compute_local_center_of_mass_z()
        self.cmx, self.cmz = self._compute_center_of_mass()

    def _compute_mass(self) -> float:
        """
        Computes the total mass of the tower using numerical integration.

        Returns:
            float: Tower mass in kilograms (kg).
        """
        height_fraction = self.tower_data["height_fract"].values
        mass_density = self.tower_data["tower_mass_den"].values
        height_positions = height_fraction * self.height

        return np.trapezoid(mass_density, height_positions)

    def _compute_local_center_of_mass_z(self) -> float:
        """
        Computes the local center of mass (Z coordinate) of the tower.

        Returns:
            float: Center of mass height in meters (m).
        """
        height_fraction = self.tower_data["height_fract"].values
        mass_density = self.tower_data["tower_mass_den"].values
        height_positions = height_fraction * self.height

        return np.trapezoid(height_positions * mass_density,
                            height_positions) / self.mass

    def _compute_center_of_mass(self) -> float:
        """
        Computes the global center of mass (CM) coordinates for the tower.

        Returns:
            tuple: (cmx, cmz), where:
                - cmx (float): X coordinate of the CM in meters (m).
                - cmz (float): Z coordinate of the CM in meters (m).
        """
        cmx = 0
        cmz = self.local_cmz + self.base_z

        return cmx, cmz


class Platform:
    """Loads and processes platform properties from OpenFAST input files."""

    def __init__(self, input_dir: str):
        """
        Initializes a Platform instance.

        Args:
            input_dir (str): Directory containing OpenFAST input files.
        """
        self.input_dir = input_dir

        # Load platform-related data
        self.elasto_data = ElastoDynInputLoader(input_dir).data
        self.hydro_data = HydroDynInputLoader(input_dir).data

        # Extract platform properties
        self.col_out_diam = self.hydro_data["col_out_diam"].iloc[0]
        self.col_thickness = self.hydro_data["col_thickness"].iloc[0]
        self.col_in_diam = self.col_out_diam - 2 * self.col_thickness
        self.col_div_size = self.hydro_data["col_div_size"].iloc[0]
        self.col_x = self.hydro_data["col_x"].iloc[0]
        self.col_base_z = self.hydro_data["col_base_z"].iloc[0]
        self.col_top_z = self.hydro_data["col_top_z"].iloc[0]
        self.col_height = self.col_top_z - self.col_base_z
        self.mass = self.elasto_data["platform_mass"].iloc[0]
        self.cmx = self.elasto_data["platform_cmx"].iloc[0]
        self.cmz = self.elasto_data["platform_cmz"].iloc[0]


class WindTurbine:
    """Processes the properties of a complete WT from OpenFAST input files."""

    def __init__(self, input_dir: str):
        """
        Initializes an WindTurbine instance.

        Args:
            input_dir (str): Directory containing OpenFAST input files.
        """
        self.input_dir = input_dir

        # Load individual components
        self.tower = Tower(self.input_dir)
        self.blades = BladeSystem(self.input_dir)
        self.hub = Hub(self.input_dir)
        self.nacelle = Nacelle(self.input_dir)
        self.platform = Platform(self.input_dir)

        # Compute total mass and center of mass
        self.mass = self._compute_mass()
        self.cmx, self.cmz = self._compute_center_of_mass()

    def _compute_mass(self) -> float:
        """
        Computes the total mass of the wind turbine.

        Returns:
            float: Total mass in kilograms.
        """
        return (self.tower.mass + self.blades.mass + self.hub.mass +
                self.nacelle.mass + self.platform.mass)

    def _compute_center_of_mass(self) -> float:
        """
        Computes the z-coordinate of the center of mass (CM) for the WT.

        Returns:
            float: Z-coordinate of the center of mass in meters.
        """
        cmx = (self.tower.mass * self.tower.cmx +
               self.blades.mass * self.blades.cmx + self.hub.mass * self.hub.cmx
               + self.nacelle.mass * self.nacelle.cmx +
               self.platform.mass * self.platform.cmx) / self.mass

        cmz = (self.tower.mass * self.tower.cmz +
               self.blades.mass * self.blades.cmz + self.hub.mass * self.hub.cmz
               + self.nacelle.mass * self.nacelle.cmz +
               self.platform.mass * self.platform.cmz) / self.mass

        return cmx, cmz


class ElastoDynTowerInputComparator:
    """Compares ElastoDyn tower input files."""

    def __init__(
        self,
        elastodyn_tower_loader_list: list[ElastoDynTowerInputLoader],
        labels: Optional[list[str]],
        output_dir: str,
    ):
        """
        Initializes an ElastoDynTowerInputComparer instance.
        
        Args:
            elastodyn_tower_loader_list (list[ElastoDynTowerInputLoader]): List
              of ElastoDynTowerInputLoader instances to compare.
            labels (Optional[list[str]], optional): Labels for the input files.
            output_dir (str): Directory to save the comparison plots.
        """
        self.elasto_tower_loaders = elastodyn_tower_loader_list
        self.labels = labels or [
            f"Loader {i + 1}" for i in range(len(elastodyn_tower_loader_list))
        ]
        self.output_dir = output_dir

    def plot_structuralprofiles_comparation(self) -> None:
        """
        Compare structural profiles (mass density and stiffness).
        """

        profiles = []
        for loader in self.elasto_tower_loaders:
            profiles.append(loader.data)

        # Set global plot style
        plt.style.use("fivethirtyeight")

        # Create a DataFrame for plotting
        combined_df = pd.concat(profiles, keys=self.labels)
        props = [
            ("tower_mass_den", "Tower Mass Density"),
            ("fa_stiffness", "Tower Stiffness (FA/SS)"),
        ]
        # Configure the figure and grid layout
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        gs = gridspec.GridSpec(1, 2, hspace=0.2, wspace=0.2)

        # Define subplots
        ax1 = fig.add_subplot(gs[0, 0], facecolor='white')
        ax2 = fig.add_subplot(gs[0, 1], facecolor='white')
        axes = [ax1, ax2]

        # Plot each property
        for ax, (key, ylabel) in zip(axes, props):
            for i, (label, group) in enumerate(combined_df.groupby(level=0)):
                ax.plot(group[key],
                        group["height_fract"],
                        label=label,
                        linewidth=1.5,
                        markersize=4,
                        color=COLORS[i % len(COLORS)])

            # Customize axes and appearance
            ax.set_xlabel(ylabel, fontsize=9, labelpad=10)
            if ax in [axes[0]]:
                ax.set_ylabel("Height Fraction", fontsize=9, labelpad=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(9)

        # Hide borders
        for ax in axes:
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)

        # Legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   loc="lower center",
                   ncol=min(len(labels), 4),
                   bbox_to_anchor=(0.5, -0.05 * (len(labels) // 6 + 1)),
                   fontsize=9,
                   frameon=False)

        # Save figure
        plot_dir = os.path.join(self.output_dir, "plot_tower_comparison")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "tower_structural_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True)

        plt.close(fig)

    def plot_modeshapes_comparation(self, normalize_shapes_plot=False) -> None:
        """
        Compare modal shapes.

        Args:
            output_dir (str): Directory to save the plot.
            normalize_shapes_plot (bool): If True, normalize the shape profiles.
        """

        def poly_shape(coeffs: list[float],
                       x: np.ndarray,
                       normalize: bool = False) -> np.ndarray:
            """
            Compute the polynomial shape profile from a list of coefficients.

            Args:
                coeffs (list[float]): Polynomial coefficients, starting from the
                  x^2 term.
                x (np.ndarray): Normalized height positions (typically in [0,
                  1]).
                normalize (bool): If True, normalize the shape to have max abs
                  value of 1 and preserve sign.

            Returns:
                np.ndarray: Polynomial shape profile (normalized if specified).
            """
            shape = sum(c * x**(i + 2) for i, c in enumerate(coeffs))
            max_idx = np.argmax(np.abs(shape))
            scale = np.abs(shape[max_idx])
            sign = np.sign(shape[max_idx])

            if normalize:
                return shape / (scale * sign)
            return shape

        mode_shapes = []
        for loader in self.elasto_tower_loaders:
            mode_shapes.append(loader.mode_shapes)

        # Set global plot style
        plt.style.use("fivethirtyeight")

        # Configure the figure and grid layout
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        gs = gridspec.GridSpec(2, 2, hspace=0.2, wspace=0.2)

        # Define subplots
        ax1 = fig.add_subplot(gs[0, 0], facecolor='white')
        ax2 = fig.add_subplot(gs[0, 1], facecolor='white')
        ax3 = fig.add_subplot(gs[1, 0], facecolor='white')
        ax4 = fig.add_subplot(gs[1, 1], facecolor='white')
        axes = [ax1, ax2, ax3, ax4]

        mode_keys = [("fore_aft", 0), ("fore_aft", 1), ("side_side", 0),
                     ("side_side", 1)]

        x = np.linspace(0, 1, 200)

        # Plot each mode shape
        for i, (key, idx) in enumerate(mode_keys):
            ax = axes[i]
            for j, (loader, label) in enumerate(
                    zip(self.elasto_tower_loaders, self.labels)):
                shape = poly_shape(loader.mode_shapes[key][idx], x,
                                   normalize_shapes_plot)
                ax.plot(x,
                        shape,
                        label=label,
                        linewidth=1.5,
                        markersize=4,
                        color=COLORS[j % len(COLORS)])

            # Customize axes and appearance
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=9)
            if normalize_shapes_plot:
                y_label = "Normalized Shape" if ax in [axes[0], axes[2]] else ""
            else:
                y_label = "Shape" if ax in [axes[0], axes[2]] else ""

            if ax in [axes[-1], axes[-2]]:
                ax.set_xlabel("Normalized Tower Height",
                              fontsize=10,
                              labelpad=10)
            ax.set_ylabel(y_label, fontsize=10, labelpad=10)

        # Customize axes and appearance
        axes[0].set_title("Mode 1", fontsize=12)
        axes[1].set_title("Mode 2", fontsize=12)
        fig.text(0.02,
                 0.7,
                 "Fore-Aft (FA)",
                 va='center',
                 ha='center',
                 rotation='vertical',
                 fontsize=11)
        fig.text(0.02,
                 0.25,
                 "Side-to-Side (SS)",
                 va='center',
                 ha='center',
                 rotation='vertical',
                 fontsize=11)

        # Hide borders
        for ax in axes:
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_visible(False)

        # Legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   loc="lower center",
                   ncol=min(len(labels), 4),
                   bbox_to_anchor=(0.5, -0.05 * (len(labels) // 6 + 1)),
                   fontsize=9,
                   frameon=False)

        # Save figure
        plot_dir = os.path.join(self.output_dir, "plot_tower_comparison")
        os.makedirs(plot_dir, exist_ok=True)
        suffix = "_normalized" if normalize_shapes_plot else ""
        plot_path = os.path.join(plot_dir,
                                 f"tower_modeshapes_comparison{suffix}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True)

        plt.close(fig)
