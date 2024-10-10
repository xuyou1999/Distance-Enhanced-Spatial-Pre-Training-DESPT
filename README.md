# Distance-Enhanced Spatial Pre-Training (DESPT)

Accurate traffic prediction is crucial for managing congestion and optimizing infrastructure in modern urban environments. A key challenge is predicting traffic for unseen sensors—newly installed or relocated sensors that existing models are unfamiliar with. This research introduces the Distance-Enhanced Spatial Pre-Training (DESPT) framework, which uses Spatial Encoding and Contrastive Learning (CL) to improve prediction accuracy for these sensors. DESPT captures sensor dependencies based on routing distances through spatial encoding and employs CL to pre-train the encoder, enriching input data with historical sensor information and spatial context. This enables the model to generalize effectively to unseen sensors. Extensive experiments on datasets like METR-LA, PeMS-BAY, and an exclusive Hague dataset demonstrate DESPT's superiority, achieving reductions in Mean Absolute Error (MAE) of 3.38% on METR-LA, 3.68% on PeMS-BAY, and 9.17% on the Hague dataset. The DESPT framework provides a robust solution for dynamic traffic networks, advancing urban traffic management.

## Instructions

Ensure the data files are located in the `data/DATASET_NAME` folder, with the following required files:
- `data.h5`: A file storing numerical data in a pandas DataFrame-like format.
- `location.csv`: A file containing the location data for each sensor.

The Hague dataset is exclusively available to the project team and is not publicly accessible due to privacy concerns.

To process the data and generate the adjacency matrix, execute the relevant sections in `explore_data.ipynb`.

### Running Experiments

Use `experiments.py` to execute the experiments. Follow these steps to replicate the research study:

1. Run `execution_tune.py` for all datasets and models. Adjust the parameter sets within the file to run experiments for a specific dataset or model.
2. Run `execution_test.py` for all datasets and models. Adjust the parameter sets based on the best tuning results to ensure a fair comparison. This file is also used for ablation and runtime experiments by setting the appropriate parameters.

Parameter sets and results are saved in the `save` folder under `parameters.csv` and `results.csv`, respectively. For all experiments, remember to set a unique `trace_id` to differentiate the results.

Refer to the code comments for details on configuring parameters and understanding the results.

### Storing Results in MongoDB Atlas

To store results in a MongoDB Atlas database, set the `username`, `password`, `cluster_url`, and `db_name` in `config.json`.

### Analyzing and Visualizing Results

Once the experiments are completed, use `results.ipynb` to analyze and visualize the outcomes.

## Data Processing

### Hague Dataset

The Hague dataset includes sensor readings on vehicle counts at various intersections, with each intersection equipped with multiple sensors. Four specific trajectories, based on prior studies, are identified. Each road consists of two trajectories, categorized by direction (North or South).

For this project, sensor locations were manually identified on Google Maps. The `location_std.csv` file documents this information with the following columns:
- **Index**: A unique number assigned to each sensor.
- **Trajectory**: Identifies one of the four trajectories, with sensors contributing to multiple trajectories indicated by identifiers separated by a "+".
- **Sensor ID**: A unique identifier created by combining the intersection identifier and the sensor identifier within the intersection using a "-".
- **Latitude**: The sensor's latitude.
- **Longitude**: The sensor's longitude.

Driving route distances between sensor pairs on the same trajectory were calculated using OSRM, and the results are stored in `distances.csv`, with the following columns:
- **Trajectory**: The trajectory ID.
- **From**: The starting point's Sensor ID.
- **To**: The ending point's Sensor ID.
- **Cost**: The driving distance in meters from the starting to the ending point.

## Contact

For questions or concerns, please feel free to reach out.