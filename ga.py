import numpy as np

DIMENSION = 6
LOWER_BOUND = -20.0
UPPER_BOUND = 20.0

POPULATION_SIZE = 150
EVALUATION_BUDGET = 30000

CROSSOVER_PROBABILITY = 0.90
MUTATION_PROBABILITY = 0.30
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2

MUTATION_SIGMA_START = 1.0
MUTATION_SIGMA_END = 0.001


def happy_cat(x):  # вариант 19
    x = np.array(x, dtype=float)

    d = x.shape[-1]
    sum_sq = np.sum(x ** 2, axis=-1)
    sum_x = np.sum(x, axis=-1)

    return np.abs(sum_sq - d) ** 0.25 + (0.5 * sum_sq + sum_x) / d + 0.5


def initialize_population(rng):  # начальная популяция
    return rng.uniform(LOWER_BOUND, UPPER_BOUND, size=(POPULATION_SIZE, DIMENSION))


def tournament_selection(population, values, rng, n_parents):  # В каждом турнире побеждает особь с минимальным f(x)
    parents = []

    for _ in range(n_parents):
        candidates = rng.choice(
            len(population),
            size=TOURNAMENT_SIZE,
            replace=False
        )

        candidate_values = values[candidates]
        winner_local_index = np.argmin(candidate_values)
        winner_index = candidates[winner_local_index]

        parents.append(population[winner_index].copy())

    return np.array(parents)


def arithmetic_crossover(parent1, parent2, rng): # Арифметический crossover для двух родителей
    if rng.random() >= CROSSOVER_PROBABILITY:
        return parent1.copy(), parent2.copy()

    alpha = rng.random(DIMENSION)

    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2

    return child1, child2


def adaptive_sigma(generation, total_generations): # уменьшение масштаба мутации 
    if total_generations <= 1:
        return MUTATION_SIGMA_END

    progress = generation / (total_generations - 1)

    sigma = MUTATION_SIGMA_START * (
        MUTATION_SIGMA_END / MUTATION_SIGMA_START
    ) ** progress

    return sigma


def gaussian_mutation(children, rng, mutation_probability, sigma): #  гауссовская мутация генов
    children = children.copy()

    mutation_mask = rng.random(children.shape) < mutation_probability
    noise = rng.normal(0, sigma, size=children.shape)
    children += mutation_mask * noise

    return children


def handle_bounds(children):
    return np.clip(children, LOWER_BOUND, UPPER_BOUND)


def make_next_population(population, values, rng, mutation_probability, sigma):
    elite_indices = np.argsort(values)[:ELITE_COUNT] # Элитизм
    elites = population[elite_indices].copy()

    n_children = POPULATION_SIZE - ELITE_COUNT  # Сколько детей нужно создать
    if n_children % 2 == 0:
        n_parents = n_children
    else:
        n_parents = n_children + 1

    parents = tournament_selection(  # Выбор родителей
        population,
        values,
        rng,
        n_parents
    )

    children = []  # Скрещиваем родителей попарно

    for i in range(0, n_parents, 2):
        child1, child2 = arithmetic_crossover(
            parents[i],
            parents[i + 1],
            rng
        )
        children.append(child1)
        children.append(child2)

    children = np.array(children[:n_children])

    children = gaussian_mutation( # Мутация
        children,
        rng,
        mutation_probability,
        sigma
    )
    children = handle_bounds(children)
    new_population = np.vstack((elites, children))
    return new_population


def run_ga(seed, mutation_probability=MUTATION_PROBABILITY): # Один полный запуск 
    rng = np.random.default_rng(seed)

    population = initialize_population(rng)
    values = happy_cat(population)

    evaluations_used = POPULATION_SIZE

    best_index = np.argmin(values)
    best_x = population[best_index].copy()
    best_value = float(values[best_index])

    history_best = [best_value]
    history_mean = [float(np.mean(values))]
    history_median = [float(np.median(values))]
    evaluations = [evaluations_used]

    total_population_evaluations = EVALUATION_BUDGET // POPULATION_SIZE
    total_generations = total_population_evaluations - 1

    for generation in range(total_generations):
        sigma = adaptive_sigma(generation, total_generations)

        population = make_next_population(
            population,
            values,
            rng,
            mutation_probability,
            sigma
        )

        values = happy_cat(population)
        evaluations_used += POPULATION_SIZE

        current_best_index = np.argmin(values)
        current_best_value = float(values[current_best_index])

        if current_best_value < best_value:
            best_value = current_best_value
            best_x = population[current_best_index].copy()

        history_best.append(best_value)
        history_mean.append(float(np.mean(values)))
        history_median.append(float(np.median(values)))
        evaluations.append(evaluations_used)

    return {
        "seed": seed,
        "best_x": best_x,
        "best_value": best_value,
        "history_best": np.array(history_best),
        "history_mean": np.array(history_mean),
        "history_median": np.array(history_median),
        "evaluations": np.array(evaluations)
    }

def run_random_search(seed): # Случайный поиск с тем же числом вычислений HappyCat, что и у ГА
    rng = np.random.default_rng(seed)

    points = rng.uniform(
        LOWER_BOUND,
        UPPER_BOUND,
        size=(EVALUATION_BUDGET, DIMENSION)
    )

    values = happy_cat(points)

    best_index = np.argmin(values)
    best_x = points[best_index].copy()
    best_value = float(values[best_index])

    return best_x, best_value


if __name__ == "__main__": # тестовый запуск
    test_x = -np.ones(DIMENSION)
    print("HappyCat(-1, ..., -1) =", happy_cat(test_x))

    result = run_ga(seed=42)
    print("Best f(x) =", result["best_value"])
    print("Best x =", result["best_x"])
