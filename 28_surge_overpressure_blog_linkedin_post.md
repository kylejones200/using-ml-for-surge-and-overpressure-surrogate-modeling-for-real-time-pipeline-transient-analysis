A control room operator initiates emergency shutdown on a 30-inch crude oil pipeline. The automated system closes the block valve in 1.8 seconds. The pressure transient propagates upstream at 1,200 m/s. Peak overpressure reaches 287 psig above normal—exceeding the 250 psig MAOP pipeline rating. Pipe ruptures 400 meters upstream releasing 8,500 barrels. Total cost: $50M. The operator followed procedure, the system worked as designed—but no one knew what the peak overpressure would be for that specific closure time.

This tutorial covers surge prediction surrogate modeling: training XGBoost regressors on transient hydraulic simulation results (OLGA, PIPESIM) achieving R²=0.94, predicting peak overpressure in milliseconds enabling operators to answer what-if questions instantly, and replacing 15-30 minute full simulations that are too slow for real-time decision support during emergency scenarios.

This matters because pressure surges (water hammer) occur when fluid velocity changes rapidly. Full transient simulation is accurate but slow. Operations need instant answers during emergencies: "If I close this valve in 5 seconds instead of 2, what will the peak pressure be?" Surrogate models provide real-time predictions enabling safer operational decisions.

https://lnkd.in/example

#pipeline #hydraulics #machinelearning #transient #surgeanalysis

