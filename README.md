# Master Graduation Project

## Data Processing

### Hague Dataset

The Hague dataset comprises readings from sensors about the number of vehicles at various intersections. Each intersection is equipped with a different number of sensors.

Tom Mertens has identified four specific trajectories of sensor groups as part of a research project on the OBIS model. The configurations for these trajectories can be found at: [OBIS Configurations](https://github.com/Tom-Mertens/OBIS/blob/main/configs.json).

For the purposes of this project, the location of each sensor has been manually identified on Google Maps. This information is documented in the `location_std.csv` file, which contains the following columns:
- **Index**: A unique index number for each sensor.
- **Trajectory**: Identifies one of the four trajectories. Each trajectory consists of two routes, each with a direction (North or South). Sensors contributing to the indicators of two trajectories are denoted with identifiers connected by a "+".
- **Sensor ID**: A unique identifier for each sensor, formed by concatenating the intersection identifier and the sensor identifier within that intersection with a "-".
- **Latitude**: The latitude of the sensor's location.
- **Longitude**: The longitude of the sensor's location.

Furthermore, the driving route distances between each pair of sensors on the same trajectory were calculated using OSRM. The results are stored in `distances.csv`, which includes the following columns:
- **Trajectory**: The ID of the trajectory.
- **From**: The starting point, identified by the Sensor ID.
- **To**: The ending point, also identified by the Sensor ID.
- **Cost**: The driving route distance between the start and end points, in meters.
