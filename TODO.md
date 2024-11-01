Metrics:
	- [x] Avg Delay
	- [x] Hit-ratio

### Federated Learning ??? very ??? (may be adapt the FedFast)
[] Integrate 2 other branches // quick

### Simulation
[x] Fix the environment
[x] Add the interrupt

### DRL
[x] Action Branching
[] Action Masking
[] Attention Output
[] Prioritized Memory

### Experiment
[x] Training results 40 vehicle
	- w interrupt
	- w/o interrupt

### Setup baselines
	Cache Strategy
	- [] Greedy: Cache the topmost requested content for the last time step.
	- [50%] AvgFed: cache replacement based on Avg federated learning CPP
	- [x] McFed: cache replacement based on multi-cluster federated learning CPP
	- [x] Random: cache randomly replaced

	Delivery Strategy
	- [] Random Delivery 
	- [] Cache
	- [] W/Wo Attention
	- [] Greedy Delivery

[] Run the comparison experiment


[] Ablation study // Design after done the FedFast





Ablation Study
	- [] No Cache
	- [] Cache 1 Branch

[] Different Cache Capacity [50, 100, 150, 200]
[] Different Vehicle Density [10, 20, 40, 80, 100]
[] w/wo Interrupt Rate


Writing ===

- Recheck the communication model, problem formulation
- Mobility Model
- Interrupt Model
- Simulation Setup
- Baseline
- Metric
- Experiment results


