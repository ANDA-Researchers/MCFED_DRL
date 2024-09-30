avg the state dict by keys

env init: poisson -> init velocity -> init vehicle velocity
env update: vehicle position -> add/remove -> update velocity -> update vehicle states

# data rate

$$
\huge
S_{V_{i}^r} = B_{i, R_j} \cdot \log_2(1 + \frac{P_{R_j} \cdot h_i^r(V_{i}^r, R_j)}{\sigma^2})
$$

- $S_{V_{i}^r}$: the data rate of vehicle $V_{i}^r$
- $B_{i, R_j}$: the bandwidth of the channel between vehicle $V_{i}^r$ and RSU $R_j$
- $P_{R_j}$: the transmission power of RSU $R_j$
- $h_i^r(V_{i}^r, R_j)$: the channel gain between vehicle $V_{i}^r$ and RSU $R_j$
- $\sigma^2$: the noise power

# channel gain

$$
\huge
128.1 + 37.6 \cdot \log_{10}(d(V_{i}^r, R_j)) + X
$$

- $d(V_{i}^r, R_j)$: the distance between vehicle $V_{i}^r$ and RSU $R_j$
- $X$ (in dB): represents shadowing channel fading, which follows Log-normal distribution

<!-- #     plt.figure(figsize=(12, 3))

#     for _ in range(1000):
#         print(f"Time: {env.current_time}")
#         print(f"Number of vehicles: {len(env.vehicles)}")
#         # for idx, vehicle in enumerate(env.vehicles):
#         #     print(vehicle)

#         if len(env.vehicles) == 0:
#             continue
#         else:
#             env.vehicles[0].local_update()
#             weights = env.vehicles[0].get_weights()

#         plt.clf()
#         x_positions = [vehicle.position[0] for vehicle in env.vehicles]
#         y_positions = [vehicle.position[1] for vehicle in env.vehicles]

#         plt.hlines([0, width], 0, length, colors="gray", linestyles="solid")
#         plt.scatter(x_positions, y_positions, c="blue", s=100, label="Points")

#         plt.title(f"Iteration {_+1}, num vehicles: {len(env.vehicles)}")
#         plt.xlim(0, length)
#         plt.ylim(0 - 10, width + 10)
#         plt.xlabel("Position along the Road (m)")
#         plt.ylabel("Width of the Road (m)")

#         # Display the updated plot without creating a new figure
#         plt.pause(0.1)

#         env.small_step()

# plt.show() -->
