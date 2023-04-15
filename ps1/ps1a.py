###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    # TODO: Your code here
    cow_dict = {}
    with open(filename, mode='r') as file:
        for record in file:
            name, weight = record.strip().split(',')
            cow_dict[name] = int(weight)
    return cow_dict

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    transport_li = []
    import copy
    cow_list = list(copy.deepcopy(cows).items())#list with (name, weight) as element
    cow_list.sort(key= lambda x:x[1], reverse=True)#sort cows according to weight
    cows_transported = set()#using a set to automatically remove duplicate cows
    current_trip = []
    
    while len(cows_transported) < len(cows):
        w = 0
        for i in range(len(cow_list)):
           if cow_list[i][0] not in cows_transported:
               if w + cow_list[i][1] <= limit:
                   w += cow_list[i][1]
                   current_trip.append(cow_list[i][0])
                   cows_transported.add(cow_list[i][0])
        
        transport_li.append(current_trip)
        current_trip = []
            
    return transport_li

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
        
    def is_valid_partition(partition):
        res = True
        for i in range(len(partition)):
            total_weight = 0
            for j in range(len(partition[i])):
                total_weight += cows[partition[i][j]]
            #print(total_weight)
            if total_weight > limit:
                res = False
                break
        return res    
    
    partitions_lt_limit = []
    import copy
    cow_names = list(copy.deepcopy(cows).keys())
    for partition in get_partitions(cow_names):
        #print(partition)
        if is_valid_partition(partition):
            partitions_lt_limit.append(partition)
    #print(partitions_lt_limit)
    
    fewest_trips = len(partitions_lt_limit[0])
    res = partitions_lt_limit[0]
    for element in partitions_lt_limit:
        if len(element) <= fewest_trips:
            fewest_trips = len(element)
            res = element
    #print(res)
    return res
        
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # TODO: Your code here
    cows = load_cows('ps1_cow_data.txt')
    import time
    start = time.time()
    greedy = greedy_cow_transport(cows,limit=10)
    end = time.time()
    time_greedy = end - start
    
    start = time.time()
    brutal = brute_force_cow_transport(cows,limit=10)
    end = time.time()
    time_brutal = end - start
    print(f'''the greedy method uses {time_greedy} seconds
          it returns {len(greedy)} trips: {greedy}''')
    print(f'''the brutal force method uses {time_brutal} seconds
          it returns {len(brutal)} trips: {brutal}''')
    return None










