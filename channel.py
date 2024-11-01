import numpy as np


class V2X:
    def __init__(self, args=None) -> None:
        self.B_sub = 1e6  # 540 kHz
        self.P_bs = 43
        self.P_rsu = 30
        self.sigma2 = -114
        self.shadow_std = 4
        self.decorrelation_distance = 50

        self.h_bs = 25 - 1.5  # antenna height of BS
        self.h_rsu = 5 - 1.5  # antenna height of RSU

        self.v_atten_gain = 3  # vehicle antenna gain
        self.r_atten_gain = 5
        self.bs_atten_gain = 14  # BS antenna gain

        self.noise_power = 10 ** (self.sigma2 / 10)

        self.distance = None
        self.channel_gain = None

        self.data_rate = None

    def path_loss(self, distance: float, mode: str = "rsu") -> float:
        if mode == "rsu":
            return 128.1 + 37.6 * np.log10(np.sqrt(distance**2 + self.h_rsu**2) * 1e-3)
        else:
            return 128.1 + 37.6 * np.log10(np.sqrt(distance**2 + self.h_bs**2) * 1e-3)

    def get_shadowing(self):
        """
        distance shape: (num_vehicle, 2) in which
        the first column is the distance between vehicle and BS
        the second column is the distance between vehicle and their corresponding RSU
        """

        shadowing_values = np.random.normal(
            0, self.shadow_std, (self.distance.shape[0], 2)
        )
        delta_distances = self.distance / self.decorrelation_distance
        shadowing_decay = np.exp(-delta_distances)
        additional_shadowing = np.sqrt(
            1 - np.exp(-2 * delta_distances)
        ) * np.random.normal(0, self.shadow_std, (self.distance.shape[0], 2))
        shadowing_values = shadowing_values * shadowing_decay + additional_shadowing
        return shadowing_values

    def get_fast_fading(self):
        real_part_bs = np.random.normal(0, 1, self.distance.shape[0])
        imag_part_bs = np.random.normal(0, 1, self.distance.shape[0])
        fading_bs = -20 * np.log10(
            np.abs(real_part_bs + 1j * imag_part_bs) / np.sqrt(2)
        )

        real_part_rsu = np.random.normal(0, 1, self.distance.shape[0])
        imag_part_rsu = np.random.normal(0, 1, self.distance.shape[0])
        fading_rsu = -20 * np.log10(
            np.abs(real_part_rsu + 1j * imag_part_rsu) / np.sqrt(2)
        )

        fast_fading = np.column_stack((fading_bs, fading_rsu))
        return fast_fading

    def reset(self, distance: np.ndarray) -> None:
        return self.step(distance)

    def step(self, distance: np.ndarray) -> np.ndarray:
        self.distance = distance

        path_loss_values = np.zeros_like(self.distance)
        path_loss_values[:, 0] = np.array(
            [self.path_loss(d, mode="bs") for d in self.distance[:, 0]]
        )
        path_loss_values[:, 1] = np.array(
            [self.path_loss(d, mode="rsu") for d in self.distance[:, 1]]
        )

        shadowing_values = self.get_shadowing()
        fast_fading_values = self.get_fast_fading()

        channel_gain_bs = 10 ** (
            (
                self.P_bs
                - path_loss_values[:, 0]
                - shadowing_values[:, 0]
                - fast_fading_values[:, 0]
                + 6
            )
            / 10
        )

        channel_gain_rsu = 10 ** (
            (
                self.P_rsu
                - path_loss_values[:, 1]
                - shadowing_values[:, 1]
                - fast_fading_values[:, 1]
                + 6
            )
            / 10
        )

        self.channel_gain = np.concatenate(
            [channel_gain_bs.reshape(1, -1), channel_gain_rsu.reshape(1, -1)], axis=0
        )

        data_rate_bs = self.B_sub * np.log2(1 + channel_gain_bs / self.noise_power)
        data_rate_rsu = self.B_sub * np.log2(1 + channel_gain_rsu / self.noise_power)

        self.data_rate = np.concatenate(
            [data_rate_bs.reshape(1, -1), data_rate_rsu.reshape(1, -1)], axis=0
        )

        return self.channel_gain, self.data_rate


if __name__ == "__main__":
    channel = V2X()
    distance = np.array([[1000, 1000], [1200, 1000]])

    channel.reset(distance)
    print(channel.distance)
    print(channel.data_rate)
