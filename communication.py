import numpy as np


class Communication:
    def __init__(self, distance_matrix):
        self.B = 20e6  # 20 MHz
        self.B_sub = 540e3  # 500 kHz
        self.shadow_std = 8  # Shadow fading standard deviation in dB
        self.transmission_power_dBm = 30  # Transmission power in dBm
        self.noise_power_dBm = -114  # Noise power in dBm
        self.decorrelation_distance = 50  # Decorrelation distance in meters
        self.distance_matrix = distance_matrix
        self.num_vehicle, self.num_rsu = distance_matrix.shape
        self.current_shadowing = self.get_shadowing()

    def compute_path_loss(self, distance):
        if distance <= 0:
            raise ValueError("Distance must be greater than zero.")
        return 128.1 + 37.6 * np.log10(distance * 1e-3)

    def get_shadowing(self):
        shadowing_values = np.zeros((self.num_vehicle, self.num_rsu))
        for i in range(self.num_vehicle):
            # Generate a base shadowing value for each vehicle
            base_shadowing = np.random.normal(0, self.shadow_std)
            for j in range(self.num_rsu):
                # Calculate the delta distance from the distance matrix
                delta_distance = self.distance_matrix[i][j]
                # Calculate shadowing using the decorrelation model
                shadowing_decay = np.exp(
                    -1 * (delta_distance / self.decorrelation_distance)
                )
                additional_shadowing = np.sqrt(
                    1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))
                ) * np.random.normal(0, 4)
                shadowing_values[i][j] = (
                    base_shadowing * shadowing_decay + additional_shadowing
                )
        return shadowing_values

    def dBm_to_Watt(self, dBm):
        # Convert dBm to Watts
        return 10 ** ((dBm - 30) / 10)

    def calculate_V2R_data_rate(self, vi, rj, i, j):
        # Compute distance between vehicle and RSU
        distance_value = self.distance_matrix[i][j]
        # Compute path loss in dB
        path_loss_dB = self.compute_path_loss(distance_value)
        shadowing = self.current_shadowing[i][j]
        # Get shadowing in dB
        # shadowing_dB = 10 * np.log10(self.current_shadowing[i][j])
        shadowing_dB = shadowing
        # Total attenuation (Path loss + Shadowing) in dB
        total_loss_dB = path_loss_dB + shadowing_dB
        # Convert total loss from dB to linear scale (channel gain in Watts)
        channel_gain_linear = 10 ** (-total_loss_dB / 10)
        # Convert transmission power and noise power from dBm to Watts
        transmission_power_watt = self.dBm_to_Watt(self.transmission_power_dBm)
        noise_power_watt = self.dBm_to_Watt(self.noise_power_dBm)

        # Calculate data rate using Shannon capacity formula
        data_rate = self.B_sub * np.log2(
            1 + (transmission_power_watt * channel_gain_linear) / (noise_power_watt)
        )
        return data_rate
