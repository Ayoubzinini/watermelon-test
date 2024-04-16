import random
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Fitness function (evaluate the subset of variables)
def evaluate_subset(subset, X, y):
    X_subset = X[:, subset]
    # Using SVR as an example, you can replace it with any other model
    model = SVR(kernel='linear')
    scaler = StandardScaler()
    scores = cross_val_score(model, scaler.fit_transform(X_subset), y, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

# Genetic Algorithm
def genetic_algorithm(X, y, population_size=20, generations=50, mutation_rate=0.1):
    num_variables = X.shape[1]
    population = [random.sample(range(num_variables), random.randint(1, num_variables)) for _ in range(population_size)]

    for gen in range(generations):
        # Evaluate fitness of each individual in the population
        fitness_scores = [evaluate_subset(individual, X, y) for individual in population]

        # Select top individuals
        top_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:5]

        # Crossover
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.choices(top_indices, k=2)
            crossover_point = random.randint(1, min(len(population[parent1]), len(population[parent2])))
            child = population[parent1][:crossover_point] + population[parent2][crossover_point:]
            new_population.append(child)

        # Mutation
        for i in range(population_size):
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, num_variables-1)
                if mutation_point in new_population[i]:
                    new_population[i].remove(mutation_point)
                else:
                    new_population[i].append(mutation_point)

        population = new_population

    # Select best individual
    best_individual = max(population, key=lambda x: evaluate_subset(x, X, y))
    return best_individual

# Example usage
best_subset = genetic_algorithm(X_train_scaled, y_train)
print("Best Subset of Variables:", best_subset)
