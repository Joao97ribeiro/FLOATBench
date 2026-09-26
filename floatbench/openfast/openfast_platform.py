# pylint: skip-file
# Released post-processing pipeline of the FLOATBench labels, kept verbatim
# (only the colour import points to floatbench.colors).
# pylint: disable=import-error
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods
"""Platform Column Water Balancing and OpenFAST Stability Analysis."""

import pandas as pd
import numpy as np

from .openfast_inputs import WindTurbine

GRAVITY = 9.81
WATER_DENSITY = 1025.0


class PlatformWaterBalancer:
    """Estimates platform column water adjustments needed for partial balance.

    This class uses aerodynamic forces and platform properties to determine how
    much water must be added to upwind or port-stbd columns to help balance the
    platform.
    """

    def __init__(self, metrics: pd.DataFrame, input_dir: str):
        """Initializes the PlatformWaterBalancer instance.

        Args:
            metrics (pd.DataFrame): Aerodynamic force and moment data.
            input_dir (str): Directory containing OpenFAST input files.
        """
        self.metrics = metrics
        self.input_dir = input_dir
        self.wind_turbine = WindTurbine(self.input_dir)

    def compute_colwater_parameters(self,
                                    safety_factor: float = 0.9) -> pd.DataFrame:
        """"Computes the required column water parameters for platform balance.

        The method determines how much water needs to be added to 
        specific columns to counteract moment imbalances.

        Args:
            safety_factor (float): A multiplier for the computed water height 
                           (e.g., 0.9 applies 90% of the computed value).

        Returns:
            DataFrame with additional columns:
            - 'plat_mass_init' (kg): Initial platform mass.
            - 'colwater_moment' (kN·m): Moment contribution due to column water
              adjustment.
            - 'colwater_type' (str): Column selected for water adjustment
              ("upwind", "port-stbd", or "-").
            - 'colwater_height_init' (m): Initial computed water height before
              adjustment.
            - 'colwater_height' (m): Final adjusted water height after
              constraints.
            - 'colwater_mass' (kg): Required water mass for balance.
            - 'plat_mass' (kg): Updated platform mass after water adjustment.
            - 'colwater_z' (m): Z-coordinate of the final water surface level in
              the column.
        """
        df_result = self.metrics.copy()

        # Wind Turbien Components
        platform = self.wind_turbine.platform
        nacelle = self.wind_turbine.nacelle
        blades = self.wind_turbine.blades
        hub = self.wind_turbine.hub
        turbine = self.wind_turbine

        # Store initial platform mass
        df_result["plat_mass_init"] = platform.mass

        # Compute aerodynamic load and moment contributions
        rotor_fxh = df_result["RtAeroFxh"]
        rotor_myh = df_result["RtAeroMyh"]
        rotor_fxh_x = rotor_fxh * np.cos(np.radians(hub.shaft_tilt))
        rotor_fxh_y = np.abs(rotor_fxh * np.sin(np.radians(hub.shaft_tilt)))

        # Compute gravitational moment (N·m)
        gravity_moment = (nacelle.mass * nacelle.cmx + hub.mass * hub.cmx +
                          blades.mass * blades.cmx) * GRAVITY

        # Compute aerodynamic moment (N·m)
        aero_moment = rotor_fxh_x * (
            hub.cmz - turbine.cmz) + rotor_fxh_y * hub.cmx + rotor_myh

        # Compute total moment imbalance (N·m), convert to kN·m
        df_result["colwater_moment"] = -(gravity_moment + aero_moment) / 1000.0

        # Determine column to be filled
        df_result["colwater_type"] = np.where(df_result["colwater_moment"] < 0,
                                              "upwind", "port-stbd")

        # Compute required load (kN), water mass (kg), water volume (m3),
        # water section area (m2)
        col_water_load = np.abs(df_result["colwater_moment"]) / np.abs(
            platform.col_x)
        col_water_mass = col_water_load * 1000 / GRAVITY
        col_water_vol = col_water_mass / WATER_DENSITY
        col_water_secarea = np.pi * (platform.col_in_diam**2) / 4

        # Compute required height (m)
        df_result["colwater_height_init"] = (col_water_vol / col_water_secarea *
                                             safety_factor)

        # Initialize adjusted height
        df_result["colwater_height"] = df_result["colwater_height_init"].iloc[0]

        # Adjust water height
        if df_result["colwater_height_init"].iloc[0] < platform.col_div_size:
            df_result["colwater_height"] = platform.col_div_size * round(
                df_result["colwater_height_init"].iloc[0] /
                platform.col_div_size)
        else:
            df_result["colwater_height"] = df_result[
                "colwater_height_init"].iloc[0]

        # Compute water mass and updated platform mass
        df_result["colwater_mass"] = df_result["colwater_height"].iloc[
            0] * col_water_secarea * WATER_DENSITY
        df_result["plat_mass"] = df_result["plat_mass_init"].iloc[
            0] - df_result["colwater_mass"].iloc[0]

        # Compute final water surface location
        df_result["colwater_z"] = platform.col_base_z + df_result[
            "colwater_height"].iloc[0]

        # If column is "port-stbd" and colwater_height > 0, double mass water
        if df_result["colwater_type"].iloc[0] == "port-stbd" and df_result[
                "colwater_height"].iloc[0] > 0:
            df_result["colwater_mass"] *= 2
            df_result["plat_mass"] = df_result["plat_mass_init"].iloc[
                0] - df_result["colwater_mass"].iloc[0]

        # If colwater_height == 0, reset values
        if df_result["colwater_height"].iloc[0] == 0:
            df_result["colwater_type"] = "-"
            df_result["colwater_mass"] = 0.0
            df_result["plat_mass"] = df_result["plat_mass_init"].iloc[0]
            df_result["colwater_z"] = "-"

        return df_result
