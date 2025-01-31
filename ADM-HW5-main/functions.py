import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque
from queue import Queue
import heapq
from scipy.stats import gaussian_kde
from itertools import product
import re
from difflib import get_close_matches

# Import required packages
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def analyze_graph_features(flight_network):
    """
    Analyze graph features from the flight network represented as a DataFrame.

    Args:
        flight_network (pd.DataFrame): A DataFrame where rows and columns represent airports,
                                          and values represent the number of flights between them.

    Returns:
        dict: A dictionary containing graph features including the number of airports, flights,
              graph density, and identified hub airports by total degree.
    """
    # Count the number of airports (nodes)
    num_airports = flight_network.shape[0]  # Number of rows (or columns, since it's square)

    # Count the number of flights (edges) by summing non-zero entries
    num_flights = np.count_nonzero(flight_network.values)  # Non-zero values represent the flights

    # Calculate graph density
    density = num_flights / (num_airports * (num_airports - 1))

    # In-degree and out-degree calculations
    in_degrees = flight_network.sum(axis=0)  # Sum along columns for in-degrees
    out_degrees = flight_network.sum(axis=1)  # Sum along rows for out-degrees
    tot_degrees = in_degrees + out_degrees  # Total degree

    # Identify hubs by total degree (90th percentile)
    tot_degrees_90th = np.percentile(tot_degrees, 90)
    hub_airports = tot_degrees[tot_degrees >= tot_degrees_90th].index.tolist()

    # Create a dictionary to hold all features
    features = {
        'num_airports': num_airports,
        'num_flights': num_flights,
        'density': density,
        'hub_airports': hub_airports,  # Airports with total degree above 90th percentile
        'in_degrees': in_degrees,#.to_dict(),  # Convert to dictionary for readability
        'out_degrees': out_degrees,#.to_dict(),  # Convert to dictionary for readability
        'tot_degrees': tot_degrees.to_dict(),  # Convert to dictionary for readability
    }

    return features

def summarize_graph_features(flight_network):
    """
    Summarize graph features and create visualizations from the flight network DataFrame.

    Args:
        flight_network (pd.DataFrame): A DataFrame where rows and columns represent airports,
                                          and values represent the number of flights between them.

    Returns:
        dict: A summary dictionary containing graph metrics and hub information.
    """
    # Get the analyzed graph features
    features = analyze_graph_features(flight_network)
    
    # Create the summary report
    summary = {
        'Number of airports (nodes)': features['num_airports'],
        'Number of flights (edges)': features['num_flights'],
        'Graph Density': features['density'],
        'Hub Airports (90th Percentile Total Degree)': features['hub_airports'],  # Airport names
    }
    
    # Print the summary
    print("Graph Summary Report:")
    for key, value in summary.items():
        if isinstance(value, list):  # Handle lists (e.g., hub airports) for better display
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    
    # Determine if the graph is sparse or dense
    density = features['density']
    graph_type = "sparse" if density < 0.1 else "dense"
    print(f"Graph Type: {graph_type}")

    # Convert degree dictionaries to Series for plotting
    in_degrees = pd.Series(features['in_degrees'])
    out_degrees = pd.Series(features['out_degrees'])

    # Plot in-degree and out-degree distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with two subplots
    
    axes[0].hist(in_degrees.values, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title('In-degree Distribution')
    axes[0].set_xlabel('In-degree')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(out_degrees.values, bins=20, color='salmon', edgecolor='black')
    axes[1].set_title('Out-degree Distribution')
    axes[1].set_xlabel('Out-degree')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    
    # Return the summary dictionary
    return summary



def bfs(adjacency_list, airport):
    # Create a set containing all the airports in the dataset
    airport_list = set(adjacency_list.keys())

    explored = {airport}

    # Initialize with 0 the distance from the input airport to itself and to inf the distance from it and each different airport
    distances = dict.fromkeys(airport_list, float('inf'))
    distances[airport] = 0

    # Create a dictionary to store the predecessors of each node
    predecessors = defaultdict(list)

    # Initialize BFS queue and the explored airports set
    q = deque([airport])

    while q:
        # Extract vertex from the queue
        v = q.popleft()
        for w in adjacency_list[v]:
            if distances[w] == float('inf'):
                # Add w to explored airports
                explored.add(w)
                # Update distance from v
                distances[w] = distances[v] + 1
                # Add w to the end of the queue
                q.append(w)
                # Record the predecessor
                predecessors[w].append(v)
            elif distances[w] == distances[v] + 1:
                # If we found another shortest path, record the predecessor
                predecessors[w].append(v)
    
    return distances, predecessors


def betweenness_centrality(adjacency_list, airport=None, normalized=False):
    
    # Initialize betweenness centrality to zero for all nodes
    betweenness = {node: 0 for node in adjacency_list.keys()}

    # Iterate through each node in the adjacency_list as the node
    for node in adjacency_list.keys():
        # Step 1: Find all shortest paths from 'node' using BFS
        distances, predecessors = bfs(adjacency_list, node)
        
        # Initialize dependency
        dependency = {node: 0 for node in adjacency_list.keys()}
        
        # Process nodes in order of decreasing distance (reverse topological order)
        sorted_nodes = sorted(
            [(node, dist) for node, dist in distances.items() if dist != float('inf')], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Dependency accumulation
        for w, _ in sorted_nodes:
            for v in predecessors[w]:
                if distances[w] == distances[v] + 1:
                    dependency[v] += (1 + dependency[w]) / len(predecessors[w])
            
            # Accumulate betweenness for nodes except node
            if w != node:
                betweenness[w] += dependency[w]
    
    # Normalize betweenness centrality if requested
    n = len(adjacency_list.keys())
    if airport is not None:
        if normalized:
            return betweenness[airport]/((n - 1) * (n - 2))
    
        else:
            return betweenness[airport]

    return {node: betweenness[node] / ((n - 1) * (n - 2)) if normalized else betweenness[node] for node in betweenness}




def closeness_centrality(adjacency_list, airport, normalized=False):
    
    # Use BFS to compute the shortest paths starting from the node airport to all other nodes
    distances = bfs(adjacency_list, airport)[0]
    
    # Distances from reachable nodes
    reachable_nodes = [dist for dist in distances.values() if dist != float('inf')]
    total_distance = sum(reachable_nodes)

    if normalized and total_distance > 0 and len(reachable_nodes) > 2:
        closeness = (len(reachable_nodes) - 1) / total_distance
    else:
        if total_distance > 0:
            closeness = 1/total_distance
        else:
            closeness = 0
    return closeness


def degree_centrality(adjacency_list, airport, normalized=False, tot=False):
    
    # Number of nodes
    n = len(adjacency_list.keys())

    # Find out degree as the number of the node'node neighbors
    out_degree_count = len(adjacency_list[airport])
    
    # Initialize in-degree for all nodes to 0
    in_degree = {key: 0 for key in adjacency_list.keys()}  
    # Iterate over the adjacency_list to calculate in-degrees
    for neighbors in adjacency_list.values():
        for neighbor in neighbors:
            in_degree[neighbor] += 1
    in_degree_count = in_degree[airport]

    if tot:
        total = in_degree_count + out_degree_count
        return round(total/(n-1), 2) if normalized else total
    else:
        if normalized:
            return (round(in_degree_count/(n-1), 2), round(out_degree_count/(n-1), 2))
        else:
            return (in_degree_count, out_degree_count)



def pagerank(adjacency_list, airport, damp = 0.85, epsilon=1e-3):
    
    n = len(adjacency_list.keys())
    
    # Initialize rank probabilities equally for all nodes
    probs = {k: 1 / n for k in adjacency_list.keys()}

    diff = float('inf')

    inlinks = defaultdict(list)

    # Compute inlinks
    for node, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            inlinks[neighbor].append(node)
    
    out_degrees = {node:len(adjacency_list[node]) for node in adjacency_list.keys()}

    while diff>=epsilon:
        diff = 0
        new_probs = {}

        for node in adjacency_list.keys():
            rank = 0

            for neighbor in inlinks.get(node, []):
                if out_degrees.get(neighbor, 0) > 0:
                    neighbor_contribution = (1/ out_degrees[neighbor]) * probs[neighbor]
                    rank += neighbor_contribution

            new_probs[node] = (1-damp)/n+damp*rank

            diff += abs(new_probs[node]-probs[node])

        probs = new_probs
    
    return probs[airport]


def analyze_centrality(adjacency_list, airport, plus=False):

    print(f"{airport}'s betweenness centrality is {betweenness_centrality(adjacency_list, airport)}")
    print(f"\n{airport}'s normalized betweenness centrality is {betweenness_centrality(adjacency_list, airport, normalized=True)}")
    print(f"\n{airport}'s closeness centrality is {closeness_centrality(adjacency_list, airport)}")
    print(f"\n{airport}'s normalized closeness centrality is {closeness_centrality(adjacency_list, airport, normalized=True)}")
    print(f"\n{airport}'s in-degree centrality is {degree_centrality(adjacency_list, airport)[0]}")
    print(f"\n{airport}'s out-degree centrality is {degree_centrality(adjacency_list, airport)[1]}")
    print(f"\n{airport}'s total degree centrality is {degree_centrality(adjacency_list, airport, tot=True)}")
    print(f"\n{airport}'s total normalized degree centrality is {degree_centrality(adjacency_list, airport, normalized=True, tot=True)}")
    print(f"\n{airport}'s PageRank score is {pagerank(adjacency_list, airport)}")
    if plus:
        hits_results = hits(adjacency_list, airport)
        print(f"\n{airport}'s Authority score is {hits_results[0]} and its Hub score is {hits_results[1]}")


def compute_centralities(adjacency_list):
    data = {'Airport': list(adjacency_list.keys()), 
            'Betweenness': list(betweenness_centrality(adjacency_list).values()),
            'Normalized betweenness':list(betweenness_centrality(adjacency_list, normalized=True).values()),
            'Closeness': [closeness_centrality(adjacency_list, i) for i in adjacency_list.keys()],
            'Normalized closeness': [closeness_centrality(adjacency_list, i, normalized=True) for i in adjacency_list.keys()],
            'In-degree': [degree_centrality(adjacency_list, i)[0] for i in adjacency_list.keys()],
            'Out-degree': [degree_centrality(adjacency_list, i)[1] for i in adjacency_list.keys()],
            'Total degree': [degree_centrality(adjacency_list, i, tot=True) for i in adjacency_list.keys()],
            'Normalized total degree': [degree_centrality(adjacency_list, i, normalized=True, tot=True) for i in adjacency_list.keys()],
            'PageRank':[pagerank(adjacency_list, i) for i in adjacency_list.keys()]}
    
    return pd.DataFrame.from_dict(data)


def plot_distributions(df):

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    axs = axs.flatten()

    colors = ['#5698ad', '#56ad83', '#8b56ad', '#9cad56', '#ad5674', '#565bad', '#ad7b56', '#ad5670', '#6656ad']
    
    # Loop over columns in the DataFrame starting from the second column
    for i, column in enumerate(df.columns[1:]):
        data = df[column].dropna()
        
        axs[i].hist(data, bins=30, color=colors[i], edgecolor='black', density=True, alpha=0.6)

        data = df[column].dropna()
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 500)
        axs[i].plot(x, kde(x), color='black', linestyle='-', linewidth=1.5, label='Density Curve')

        axs[i].set_title(f'{column} histogram')
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(data.min(), data.max())

    plt.tight_layout()

    plt.show()


def print_top_airports(df, n=5):
   
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()

    colors = ['#5698ad', '#56ad83', '#8b56ad', '#9cad56', '#ad5674', '#565bad', '#ad7b56', '#ad5670', '#6656ad']
    
    # Loop through all numeric columns and corresponding axes
    for i, column in enumerate(numeric_columns):
        # Sort the DataFrame by the current column
        top = df.nlargest(n, column)
        
        # Extract IDs and their corresponding values
        ids = top.iloc[:, 0]
        values = top[column]
        
        # Print the top IDs and values
        print(f"Top {n} airports by '{column}':")
        for idx, value in zip(ids, values):
            print(f"Airport: {idx}, {column}: {value}")
        print("_" * 40)
        
        # Plot the bar chart
        axs[i].bar(ids, values, color=colors[i], alpha=0.8, edgecolor='black')
        axs[i].set_title(f"Top {n} airports by '{column}'", fontsize=12)
        axs[i].set_xlabel('Airport', fontsize=10)
        axs[i].set_ylabel(column, fontsize=10)
        axs[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def compare_centralities(adjacency_list, n=5, centralities=False):
    
    centralities = compute_centralities(adjacency_list)

    plot_distributions(centralities)

    print_top_airports(centralities, n=n)


def hits(adjacency_list, airport=None, max_iter=1000, epsilon=1e-6):
    
    # Initialize the authority and hub scores for each node
    auth = {k: 1.0 for k in adjacency_list.keys()}
    hubs = {k: 1.0 for k in adjacency_list.keys()}

    for _ in range(max_iter):
        # Store previous scores to check for convergence
        prev_auth = auth.copy()
        prev_hubs = hubs.copy()

        norm = 0

        # Update authority scores
        for node in adjacency_list.keys():
            auth[node] = 0
            for other_node in adjacency_list.keys():
                if node in adjacency_list[other_node]:
                    auth[node] += hubs[other_node]
            norm += auth[node]**2
        norm = math.sqrt(norm)

        # Normalize authority scores
        for node in adjacency_list.keys():
            if norm == 0:
                norm = 1
            auth[node] /= norm

        norm = 0

        # Update hub scores
        for node in adjacency_list.keys():
            hubs[node] = 0
            for neighbor in adjacency_list[node]:
                hubs[node] += auth[neighbor]
            norm += hubs[node]**2
        norm = math.sqrt(norm)

        # Normalize hub scores
        for node in adjacency_list.keys():
            if norm == 0:
                norm = 1
            hubs[node] /= norm

        # Check for convergence
        auth_diff = sum(abs(auth[node] - prev_auth[node]) for node in auth.keys())
        hubs_diff = sum(abs(hubs[node] - prev_hubs[node]) for node in hubs.keys())

        if auth_diff < epsilon and hubs_diff < epsilon:
            break


    if airport is not None:
        return auth.get(airport, 0), hubs.get(airport, 0)
    else:
        return (auth, hubs)


# Function to normalize city names
def clean_city_input(city_input):
    """
    Preprocesses the city name for robust matching.
    - Removes state information (e.g., ', TX')
    - Removes content inside parentheses
    - Removes extra spaces
    - Converts to lowercase
    """
    city_cleaned = re.sub(r'\(.*?\)', '', city_input)  # Removes content inside parentheses
    city_cleaned = re.sub(r',\s*\w+$', '', city_cleaned)  # Removes state information
    return city_cleaned.strip().lower()  # Removes extra spaces and converts to lowercase


# Robust matching function
def match_city(city_input, unique_cities):
    """
    Finds the corresponding city in the dataset given a partially or fully formatted input.
    - Searches for an exact match
    - If no exact match is found, searches for partial matches
    """
    city_input_cleaned = clean_city_input(city_input)
    preprocessed_cities = {clean_city_input(city): city for city in unique_cities}
    
    # Check for exact match
    if city_input_cleaned in preprocessed_cities:
        return preprocessed_cities[city_input_cleaned]
    
    # Check for partial matches
    matches = get_close_matches(city_input_cleaned, preprocessed_cities.keys(), n=1)
    if matches:
        return preprocessed_cities[matches[0]]
    
    # No match found
    return None


# Function to find the best routes between two cities
def find_best_routes(df, origin_city, destination_city, fly_date):
    """Finds the best routes between two cities using flight data."""
    
    # Dataset normalization
    df = df.assign(
        Origin_city_clean=df['Origin_city'].apply(clean_city_input),
        Destination_city_clean=df['Destination_city'].apply(clean_city_input)
    )
    
    origin_city = clean_city_input(origin_city)
    destination_city = clean_city_input(destination_city)

    # 1. Filter flights by date
    flights_filtered = df[df['Fly_date'] == fly_date]
    if flights_filtered.empty:
        print(f"Error: No flights are available for the date {fly_date}. Searching for alternative dates...")
        
        # Check for the next closest date
        next_date = df[df['Fly_date'] > fly_date]['Fly_date'].min()
        prev_date = df[df['Fly_date'] < fly_date]['Fly_date'].max()

        if not pd.isna(next_date):
            print(f"We suggest the next closest alternative date: {next_date}")
            flights_filtered = df[df['Fly_date'] == next_date]
        elif not pd.isna(prev_date):
            print(f"We suggest the previous closest alternative date: {prev_date}")
            flights_filtered = df[df['Fly_date'] == prev_date]
        else:
            print("No alternative dates available.")
            return pd.DataFrame()  # Returns an empty DataFrame

    # 2. Create city-to-airport mapping
    city_to_airports = {
        city: list(flights_filtered[flights_filtered['Origin_city_clean'] == city]['Origin_airport'].unique())
        for city in flights_filtered['Origin_city_clean'].unique()
    }
    city_to_airports.update({
        city: list(flights_filtered[flights_filtered['Destination_city_clean'] == city]['Destination_airport'].unique())
        for city in flights_filtered['Destination_city_clean'].unique()
    })

    # 3. Create flight graph
    G = nx.DiGraph()
    for _, row in flights_filtered.iterrows():
        G.add_edge(row['Origin_airport'], row['Destination_airport'], weight=row['Distance'])

    # 4. Find all airport-to-airport combinations
    origin_airports = city_to_airports.get(origin_city, [])
    destination_airports = city_to_airports.get(destination_city, [])
    
    if not origin_airports:
        print(f"Error: No airports are available for the origin city '{origin_city}'.")
    if not destination_airports:
        print(f"Error: No airports are available for the destination city '{destination_city}'.")

    if not origin_airports or not destination_airports:
        return pd.DataFrame()  # Returns an empty DataFrame

    airport_pairs = product(origin_airports, destination_airports)

    # 5. Calculate the best routes
    results = []
    for org, dest in airport_pairs:
        try:
            # Find the shortest path (minimizing distance)
            path = nx.shortest_path(G, source=org, target=dest, weight='weight')
            results.append({
                'Origin_city_airport': org,
                'Destination_city_airport': dest,
                'Best_route': ' → '.join(path)
            })
        except nx.NetworkXNoPath:
            # No route found
            results.append({
                'Origin_city_airport': org,
                'Destination_city_airport': dest,
                'Best_route': 'No route found'
            })

    # 6. Convert results into a DataFrame
    return pd.DataFrame(results)



def airline_partitioning(df):
    G = nx.DiGraph()  
    for _, row in df.iterrows():
        G.add_edge(row['Origin_airport'], row['Destination_airport'], capacity=1)  # Weight of 1 for each edge
        G.add_edge(row['Destination_airport'], row['Origin_airport'], capacity=1)  # Makes the graph undirected

    #Source and Sink Nodes as default for Minimum Cut
    source = list(G.nodes)[0]  # First node as source
    sink = list(G.nodes)[-1]   # Last node as sink
    
    # calculating Minimum Cut
    cut_value, partition = nx.minimum_cut(G, source, sink)
    reachable, non_reachable = partition
    removed_edges = [(u, v) for u in reachable for v in non_reachable if G.has_edge(u, v)]
    
    print(f"Minimum Cut Value (number of edges removed): {cut_value}")
    print("Flights (Edges) removed to disconnect the graph:")
    for edge in removed_edges:
        print(edge)
    
    # 4. Visualization of the Original Network
    plt.figure(figsize=(12, 6))
    plt.title("Original flight network")
    pos = nx.spring_layout(G, seed=50)  # Layout for the graph
    nx.draw(G, pos, with_labels=True, node_color='purple', edge_color='black', node_size=500)
    plt.show()
    
    # 5. Creating the Resulting Subgraphs
    G_removed = G.copy()
    G_removed.remove_edges_from(removed_edges)
    
    # 6. Visualization of the Resulting Subgraphs
    plt.figure(figsize=(12, 6))
    plt.title("Flight network after Minimum Cut")
    
    # Color nodes based on their partition
    node_colors = ['green' if node in reachable else 'yellow' for node in G_removed.nodes()]
    nx.draw(G_removed, pos, with_labels=True, node_color=node_colors, edge_color='black', node_size=500)
    plt.show()


def balance_partitioning(graph_data):
    G = nx.Graph()
    for _, row in graph_data.iterrows():
        G.add_edge(row['Origin_airport'], row['Destination_airport'])

    #Partitioning using Kernighan-Lin
    partition = nx.community.kernighan_lin_bisection(G)
    group_1, group_2 = partition

    #edges between partitions
    edges_cut = [(u, v) for u in group_1 for v in group_2 if G.has_edge(u, v)]
    
    print("Balanced Partition Results:")
    print(f"Number of nodes in Partition 1: {len(group_1)}")
    print(f"Number of nodes in Partition 2: {len(group_2)}")
    print(f"Number of edges between partitions: {len(edges_cut)}")

    # Visualization of the partitions
    plt.figure(figsize=(12, 8))
    plt.title("Balanced Flight Network Partitioning")
    pos = nx.spring_layout(G, seed=50) 

    #Nodes based on their partition
    node_colors = ['lightgreen' if node in group_1 else 'yellow' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=50, font_size=5, edge_color='black')

    # Highlight edges between partitions
    nx.draw_networkx_edges(G, pos, edgelist=edges_cut, edge_color='red', width=2)
    plt.show()


# Function to check if two cities are in the same community
def are_in_same_community(city1, city2):
    return partition.get(city1) == partition.get(city2)

# Function to check if two cities are in the same community
def are_in_same_community_infomap(city1, city2):
    for community in communities:
        if city1 in community and city2 in community:
            return True
    return False

def find_connected_components(flight_network, start_date, end_date):
 
    # dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    #Edges based on the date range
    edges_in_date_range = [
        (u, v)
        for u, v, data in flight_network.edges(data=True)
        if start_date <= datetime.strptime(data['Fly_date'], "%Y-%m-%d") <= end_date
    ]

    # Create a new graph with only the filtered edges
    filtered_graph = nx.Graph()
    filtered_graph.add_edges_from(edges_in_date_range)

    # Find connected components
    connected_components = list(nx.connected_components(filtered_graph))

    # Number of connected components
    num_components = len(connected_components)

    # Size of each connected component
    component_sizes = [len(component) for component in connected_components]

    # Largest connected component
    largest_component = max(connected_components, key=len)

    print(f"Number of connected components: {num_components}")
    print(f"Size of each connected component: {component_sizes}")
    print(f"Largest connected component: {largest_component} (size: {len(largest_component)})")

    return num_components, component_sizes, largest_component