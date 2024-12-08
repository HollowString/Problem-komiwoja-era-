# Problem-komiwoja-era-

import random
import numpy as np
import matplotlib.pyplot as plt

POP_SIZE = 100 
GEN_MAX = 200  
MUTATION_RATE = 0.1  
CROSSOVER_RATE = 0.7  
CHROMOSOME_LENGTH = 10 

def fitness_function(chromosome):
    return sum(chromosome)

def initialize_population(pop_size, chromosome_length):
    return [np.random.randint(0, 2, size=chromosome_length).tolist() for _ in range(pop_size)]

def tournament_selection(population, fitness_scores, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(list(enumerate(fitness_scores)), tournament_size)
        winner = max(participants, key=lambda x: x[1])[0]
        selected.append(population[winner])
    return selected

def one_point_crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def genetic_algorithm():
    population = initialize_population(POP_SIZE, CHROMOSOME_LENGTH)
    best_fitness_per_gen = []
    avg_fitness_per_gen = []
    
    for generation in range(GEN_MAX):
        fitness_scores = [fitness_function(chrom) for chrom in population]
        
        best_fitness_per_gen.append(max(fitness_scores))
        avg_fitness_per_gen.append(np.mean(fitness_scores))
        
        selected_population = tournament_selection(population, fitness_scores)
        
        next_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < len(selected_population) else selected_population[0]
            child1, child2 = one_point_crossover(parent1, parent2)
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))
        
        population = next_population[:POP_SIZE]
    
    return best_fitness_per_gen, avg_fitness_per_gen

best_fitness, avg_fitness = genetic_algorithm()

plt.figure(figsize=(10, 6))
plt.plot(range(GEN_MAX), best_fitness, label='Najlepszy fitness', color='blue')
plt.plot(range(GEN_MAX), avg_fitness, label='Średni fitness', color='orange')
plt.xlabel('Pokolenie')
plt.ylabel('Fitness')
plt.title('Postęp algorytmu ewolucyjnego dla 200 pokoleń')
plt.legend()
plt.grid()
plt.show()
