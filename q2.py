import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import norm, t

# Load the highway network from the GML file
assignmentGraph = nx.read_gml('networkAssignment.gml')

# Define the arrival rates per hour (vehicles per hour)
arrival_rates = {
    0: 314.2, 1: 162.4, 2: 138.6, 3: 148.8, 4: 273.2, 5: 1118.8, 6: 2773.8, 7: 4036.2,
    8: 4237.4, 9: 3277.0, 10: 2843.0, 11: 2876.4, 12: 3143.0, 13: 3277.8, 14: 3546.2, 15: 4335.0,
    16: 4945.4, 17: 4525.8, 18: 2847.8, 19: 1828.0, 20: 1378.4, 21: 1271.2, 22: 1171.2, 23: 767.6
}

# Define vehicle types and their properties
vehicle_types = {
    'car': {'probability': 0.9, 'max_speed': 100},  # 90% cars, max speed 100 km/h
    'truck': {'probability': 0.1, 'max_speed': 80}  # 10% trucks, max speed 80 km/h
}

# Define City A and City B
city_A = '43108886'
city_B = '44996729'

# Function to calculate travel time for a link
def calculate_travel_time(link_length, max_speed):
    mean_travel_time = link_length / max_speed  # in hours
    std_dev = mean_travel_time / 20  # standard deviation
    travel_time = norm.rvs(loc = mean_travel_time, scale = std_dev)  # normally distributed
    return travel_time * 60  # convert to minutes

# Function to simulate vehicle arrivals and travel
def simulate_traffic(G, arrival_rates, vehicle_types, city_A, city_B, num_runs):
    all_travel_times = []
    all_truck_travel_times = []
    all_car_travel_times = []
    all_AB_travel_times = []  # Travel times for cars from city A to city B

    for _ in range(num_runs):
        travel_times = []
        truck_travel_times = []
        car_travel_times = []
        AB_travel_times = []

        for hour, rate in arrival_rates.items():
            num_vehicles = np.random.poisson(rate)
            for _ in range(num_vehicles):
                # Randomly select origin and destination
                nodes = list(G.nodes())
                origin, destination = np.random.choice(nodes, size = 2, replace = False)

                # Determine vehicle type
                vehicle_type = np.random.choice(list(vehicle_types.keys()), p = [v['probability'] for v in vehicle_types.values()])
                max_speed = vehicle_types[vehicle_type]['max_speed']

                # Find the shortest path
                shortest_path = nx.shortest_path(G, source = origin, target = destination, weight = 'length')
                total_travel_time = 0

                # Calculate travel time for each link in the path
                for i in range(len(shortest_path) - 1):
                    u, v = shortest_path[i], shortest_path[i + 1]
                    link_length = G[u][v]['length']
                    total_travel_time += calculate_travel_time(link_length, max_speed)

                # Store travel times
                travel_times.append(total_travel_time)
                if vehicle_type == 'truck':
                    truck_travel_times.append(total_travel_time)
                else:
                    car_travel_times.append(total_travel_time)
                    if origin == city_A and destination == city_B:  # Track travel time from A to B
                        AB_travel_times.append(total_travel_time)

        # Store results for this run
        all_travel_times.append(travel_times)
        all_truck_travel_times.append(truck_travel_times)
        all_car_travel_times.append(car_travel_times)
        all_AB_travel_times.append(AB_travel_times)

    return all_travel_times, all_truck_travel_times, all_car_travel_times, all_AB_travel_times

# Run the simulation
num_runs = 10  # Number of simulation runs for confidence intervals
all_travel_times, all_truck_travel_times, all_car_travel_times, all_AB_travel_times = simulate_traffic(assignmentGraph, arrival_rates, vehicle_types, city_A, city_B, num_runs)

# Calculate mean, standard deviation, and 95% confidence interval for each performance measure
def calculate_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data, ddof = 1)
    confidence_interval = t.interval(0.95, len(data) - 1, loc = mean, scale = std_dev/np.sqrt(len(data)))
    return mean, std_dev, confidence_interval

# Calculate statistics for each performance measure
total_vehicles = [len(times) for times in all_travel_times]
mean_total_vehicles, std_total_vehicles, ci_total_vehicles = calculate_statistics(total_vehicles)

mean_travel_time, std_travel_time, ci_travel_time = calculate_statistics([item for sublist in all_travel_times for item in sublist])
mean_truck_travel_time, std_truck_travel_time, ci_truck_travel_time = calculate_statistics([item for sublist in all_truck_travel_times for item in sublist])
mean_car_travel_time, std_car_travel_time, ci_car_travel_time = calculate_statistics([item for sublist in all_car_travel_times for item in sublist])
mean_AB_travel_time, std_AB_travel_time, ci_AB_travel_time = calculate_statistics([item for sublist in all_AB_travel_times for item in sublist])

# Print the results in a table format
print("Performance Measure | Mean | Standard Deviation | 95% Confidence Interval")
print("---------------------------------------------------------------")
print(f"Total number of vehicles | {mean_total_vehicles:.2f} | {std_total_vehicles:.2f} | {ci_total_vehicles}")
print(f"Travel time (arbitrary vehicle) | {mean_travel_time:.2f} | {std_travel_time:.2f} | {ci_travel_time}")
print(f"Travel time (truck) | {mean_truck_travel_time:.2f} | {std_truck_travel_time:.2f} | {ci_truck_travel_time}")
print(f"Travel time (car) | {mean_car_travel_time:.2f} | {std_car_travel_time:.2f} | {ci_car_travel_time}")
print(f"Travel time from A to B (car) | {mean_AB_travel_time:.2f} | {std_AB_travel_time:.2f} | {ci_AB_travel_time}")

# Plot a histogram of the travel time for cars from city A to city B
plt.hist([item for sublist in all_AB_travel_times for item in sublist], bins = 30, edgecolor = 'black')
plt.title('Travel Time Distribution for Cars from City A to City B')
plt.xlabel('Travel Time (minutes)')
plt.ylabel('Frequency')
plt.show()