import numpy as np


class Communication:
    def __init__(self, distance_matrix):

        self.distance_matrix = distance_matrix
        self.num_vehicle, self.num_rsu = distance_matrix.shape

        self.B_sub = 540e3  # in Hz
        self.shadow_std = 8  # in dB
        self.transmission_power_dBm = 30  # in dBm
        self.noise_power_dBm = -114  # in dBm
        self.decorrelation_distance = 50  # in meters

        self.current_shadowing = self.get_shadowing()  # in dB
        self.current_fast_fading = self.get_fast_fading()  # in dB

    def compute_path_loss(self, distance):
        return 128.1 + 37.6 * np.log10(distance * 1e-3)

    def get_shadowing(self):
        shadowing_values = np.random.normal(
            0, self.shadow_std, (self.num_vehicle, self.num_rsu)
        )
        delta_distances = self.distance_matrix / self.decorrelation_distance
        shadowing_decay = np.exp(-delta_distances)
        additional_shadowing = np.sqrt(
            1 - np.exp(-2 * delta_distances)
        ) * np.random.normal(0, self.shadow_std, (self.num_vehicle, self.num_rsu))
        shadowing_values = shadowing_values * shadowing_decay + additional_shadowing
        return shadowing_values

    def get_fast_fading(self):
        real_part = np.random.normal(0, 1, (self.num_vehicle, self.num_rsu))
        imag_part = np.random.normal(0, 1, (self.num_vehicle, self.num_rsu))
        fast_fading = -20 * np.log10(np.abs(real_part + 1j * imag_part) / np.sqrt(2))
        return fast_fading

    def get_data_rate(self, i, j):
        distance_value = self.distance_matrix[i][j]
        path_loss_dB = self.compute_path_loss(distance_value)
        shadowing_dB = self.current_shadowing[i][j]
        fast_fading_dB = self.current_fast_fading[i][j]

        # Total channel gain
        channel_gain = 10 ** (
            (self.transmission_power_dBm - path_loss_dB - shadowing_dB + fast_fading_dB)
            / 10
        )

        noise_power_watt = 10 ** (self.noise_power_dBm / 10)

        # Calculate data rate using Shannon capacity formula
        data_rate = self.B_sub * np.log2(1 + np.divide(channel_gain, noise_power_watt))

        return data_rate
