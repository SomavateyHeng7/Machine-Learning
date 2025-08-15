#Genetic Algorithm code

import random 

POPULATION_SIZE = 5
GENES = 4  
GEN_MIN = 1
GEN_MAX = 30
TARGET = 30

def create_chromosome():
    return [random.randint(GEN_MIN, GEN_MAX) for _ in range(GENES)]

def calculate_fitness(chromosome):
    a, b, c, d = chromosome
    result = a + 2*b + 3*c + 4*d
    return abs(TARGET - result)

def create_initial_population():
    population = []
    for i in range(POPULATION_SIZE):
        chromosome = create_chromosome()
        population.append(chromosome)
    return population

def calculate_fitness_rates(fitness_values):
    inverses = [1/(fitness + 0.001) for fitness in fitness_values]
    total_inverse = sum(inverses)
    
    # Calculate percentages
    fitness_rates = [(inv/total_inverse) * 100 for inv in inverses]
    return fitness_rates

#Select chromosome with lowest fitness rate for crossover
def select_worst_chromosome(fitness_values, fitness_rates):
    min_rate = min(fitness_rates)
    min_idx = fitness_rates.index(min_rate)
    return min_idx

#perform crossover between two parents
def crossover(parent1, parent2):
    crossover_point = random.randint(1, GENES-1)
    
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2

#mutations randomly to the population
def mutate_population(population):
    new_population = []
    for chromosome in population:
        new_chromosome = chromosome[:]
        # Random mutation with low probability
        if random.random() < 0.2:
            mut_idx = random.randint(0, GENES-1)
            new_chromosome[mut_idx] = random.randint(GEN_MIN, GEN_MAX)
        new_population.append(new_chromosome)
    return new_population

def print_population_table(population, fitness_values, fitness_rates, crossover_num):
    print(f"Crossover = {crossover_num}")
    print("Chromosome No.  Chromosome              Fitness FitnessRate(%)")
    
    chromosome_labels = ['C1', 'C2', 'C3', 'C4', 'C5']
    
    for i in range(len(population)):
        a, b, c, d = population[i]
        fitness = fitness_values[i]
        rate = fitness_rates[i]
        
        print(f"{chromosome_labels[i]}              ({a} {b} {c} {d})               {fitness}       {rate:.2f}")

def genetic_algorithm():

    # Create initial population
    population = create_initial_population()
    crossover_count = 0
    
    while True:
        # Calculate fitness values and rates
        fitness_values = [calculate_fitness(chromosome) for chromosome in population]
        fitness_rates = calculate_fitness_rates(fitness_values)
        
        # Check for solution
        if min(fitness_values) == 0:
            best_idx = fitness_values.index(0)
            best_chromosome = population[best_idx]
            print(f"\nSOLUTION FOUND!")
            a, b, c, d = best_chromosome
            print(f"Chromosome: ({a}, {b}, {c}, {d})")
            print(f"Verification: {a} + 2×{b} + 3×{c} + 4×{d} = {a + 2*b + 3*c + 4*d}")
            break
        
        print_population_table(population, fitness_values, fitness_rates, crossover_count)
        
        # Select chromosome with lowest fitness rate
        worst_idx = select_worst_chromosome(fitness_values, fitness_rates)
        chromosome_labels = ['C1', 'C2', 'C3', 'C4', 'C5']
        selected_label = chromosome_labels[worst_idx]
        a, b, c, d = population[worst_idx]
        rate = fitness_rates[worst_idx]
        
        print(f"Chromosome {selected_label}:({a} {b} {c} {d}) is invalid for cross-over (fitnessRate = {rate:.1f}%)")
        print()
        
        # Select two best chromosomes for crossover
        best_fitness1 = min(fitness_values)
        best_idx1 = fitness_values.index(best_fitness1)
        
        # Find second best
        remaining_fitness = fitness_values[:]
        remaining_fitness[best_idx1] = float('inf')  # Exclude the best one
        best_fitness2 = min(remaining_fitness)
        best_idx2 = fitness_values.index(best_fitness2)
        
        parent1 = population[best_idx1]
        parent2 = population[best_idx2]
        
        # Perform crossover
        offspring1, offspring2 = crossover(parent1, parent2)
        
        # Create new population - replace worst chromosome and apply mutations
        new_population = population[:]
        new_population[worst_idx] = offspring1  # Replace worst with offspring
        
        # Apply random mutations
        new_population = mutate_population(new_population)
        
        population = new_population
        crossover_count += 1
        
        # Stop after reasonable number of iterations
        if crossover_count > 50:
            print("Stopped after maximum crossovers")
            break

genetic_algorithm()