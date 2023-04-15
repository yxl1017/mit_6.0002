###########################
# 6.0002 Problem Set 1b: Space Change
# Name:
# Collaborators:
# Time:
# Author: 

#================================
# Part B: Golden Eggs
# Dynamic programming involves breaking a larger problem into smaller,
# simpler subproblems, solving the subproblems, and storing their solutions. 
# What are the subproblems in this case? What values do you want to store?
# Answer: the subproblems is smaller target weights, I want to store the number
# of eggs and their weights in the memo.

# This problem is analogous to the knapsack problem, which you saw in lecture. 
# Imagine the eggs are items you are packing. What is the objective function? 
# What is the weight limit in this case? What are the values of each item? 
# What is the weight of each item?
# Answer: 







#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    # TODO: Your code here
    if target_weight == 0:
        return 0

    try:
        return memo[target_weight]

    except KeyError:
        for egg in egg_weights:
            new_weight = target_weight - egg
            if new_weight >= 0:
                result = 1 + dp_make_weight(egg_weights, new_weight, memo)
                memo[target_weight] = result
    return result

# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 20)
    n = 99
    print(f"Egg weights = {egg_weights}")
    print("n = 99")
    #print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()
    

# =============================================================================
# 1. Explain why it would be difficult to use a brute force algorithm to solve
#    this problem if there were 30 different egg weights. You do not need to 
#    implement a brute force algorithm in order to answer this.
# 2. If you were to implement a greedy algorithm for finding the minimum number
#    of eggs needed, what would the objective function be? What would the 
#    constraints be? What strategy would your greedy algorithm follow to pick 
#    which eggs to take? You do not need to implement a greedy algorithm in 
#    order to answer this.
# 3. Will a greedy algorithm always return the optimal solution to this problem?
#    Explain why it is optimal or give an example of when it will not return 
#    the optimal solution. Again, you do not need to implement a greedy 
#    algorithm in order to answer this.
# =============================================================================















