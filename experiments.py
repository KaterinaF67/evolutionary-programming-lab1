import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ga import (
    DIMENSION,
    LOWER_BOUND,
    UPPER_BOUND,
    POPULATION_SIZE,
    EVALUATION_BUDGET,
    CROSSOVER_PROBABILITY,
    MUTATION_PROBABILITY,
    TOURNAMENT_SIZE,
    ELITE_COUNT,
    MUTATION_SIGMA_START,
    MUTATION_SIGMA_END,
    run_ga,
    run_random_search
)

N_RUNS = 20
SEEDS = list(range(202600, 202600 + N_RUNS))

LOW_MUTATION = 0.15
MAIN_MUTATION = MUTATION_PROBABILITY  # 0.30

RESULTS_DIR = "results"


def summarize(values):
    return {
        "best": np.min(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values, ddof=1),
        "worst": np.max(values)
    }


def save_parameters():
    rows = [
        {
            "method": "GA_mutation_0.15",
            "dimension": DIMENSION,
            "lower_bound": LOWER_BOUND,
            "upper_bound": UPPER_BOUND,
            "population_size": POPULATION_SIZE,
            "evaluation_budget": EVALUATION_BUDGET,
            "crossover_probability": CROSSOVER_PROBABILITY,
            "mutation_probability": LOW_MUTATION,
            "tournament_size": TOURNAMENT_SIZE,
            "elite_count": ELITE_COUNT,
            "mutation_sigma_start": MUTATION_SIGMA_START,
            "mutation_sigma_end": MUTATION_SIGMA_END,
            "n_runs": N_RUNS,
            "seed_start": SEEDS[0],
            "seed_end": SEEDS[-1]
        },
        {
            "method": "GA_mutation_0.30",
            "dimension": DIMENSION,
            "lower_bound": LOWER_BOUND,
            "upper_bound": UPPER_BOUND,
            "population_size": POPULATION_SIZE,
            "evaluation_budget": EVALUATION_BUDGET,
            "crossover_probability": CROSSOVER_PROBABILITY,
            "mutation_probability": MAIN_MUTATION,
            "tournament_size": TOURNAMENT_SIZE,
            "elite_count": ELITE_COUNT,
            "mutation_sigma_start": MUTATION_SIGMA_START,
            "mutation_sigma_end": MUTATION_SIGMA_END,
            "n_runs": N_RUNS,
            "seed_start": SEEDS[0],
            "seed_end": SEEDS[-1]
        },
        {
            "method": "Random_search",
            "dimension": DIMENSION,
            "lower_bound": LOWER_BOUND,
            "upper_bound": UPPER_BOUND,
            "population_size": "",
            "evaluation_budget": EVALUATION_BUDGET,
            "crossover_probability": "",
            "mutation_probability": "",
            "tournament_size": "",
            "elite_count": "",
            "mutation_sigma_start": "",
            "mutation_sigma_end": "",
            "n_runs": N_RUNS,
            "seed_start": SEEDS[0],
            "seed_end": SEEDS[-1]
        }
    ]

    parameters = pd.DataFrame(rows)
    parameters.to_csv(
        os.path.join(RESULTS_DIR, "parameters.csv"),
        index=False
    )

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_parameters()

    rows = []
    main_histories = []
    main_evaluations = None

    for seed in SEEDS:
        # ГА с вероятностью мутации 0.15
        result_low = run_ga(
            seed=seed,
            mutation_probability=LOW_MUTATION
        )

        row = {
            "method": "GA_mutation_0.15",
            "seed": seed,
            "best_value": result_low["best_value"]
        }

        for i, value in enumerate(result_low["best_x"]):
            row[f"x{i + 1}"] = value

        rows.append(row)

        # ГА с вероятностью мутации 0.30
        result_main = run_ga(
            seed=seed,
            mutation_probability=MAIN_MUTATION
        )

        row = {
            "method": "GA_mutation_0.30",
            "seed": seed,
            "best_value": result_main["best_value"]
        }

        for i, value in enumerate(result_main["best_x"]):
            row[f"x{i + 1}"] = value

        rows.append(row)

        main_histories.append(result_main["history_best"])
        main_evaluations = result_main["evaluations"]

        # Случайный
        random_x, random_value = run_random_search(seed)

        row = {
            "method": "Random_search",
            "seed": seed,
            "best_value": random_value
        }

        for i, value in enumerate(random_x):
            row[f"x{i + 1}"] = value

        rows.append(row)

    runs = pd.DataFrame(rows)
    runs.to_csv(
        os.path.join(RESULTS_DIR, "runs.csv"),
        index=False
    )

    summary_rows = []

    for method in runs["method"].unique():
        method_values = runs.loc[
            runs["method"] == method,
            "best_value"
        ].to_numpy()

        stats = summarize(method_values)
        stats["method"] = method
        summary_rows.append(stats)

    summary = pd.DataFrame(summary_rows)

    histories = np.vstack(main_histories)

    best_min = np.min(histories, axis=0)
    best_mean = np.mean(histories, axis=0)
    best_max = np.max(histories, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(main_evaluations, best_mean, label="Среднее лучшее значение")
    plt.fill_between(
        main_evaluations,
        best_min,
        best_max,
        alpha=0.20,
        label="Диапазон min-max по 20 запускам"
    )
    plt.xlabel("Число вычислений целевой функции")
    plt.ylabel("Лучшее найденное значение f(x)")
    plt.title("Сходимость генетического алгоритма")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "convergence.png"),
        dpi=180
    )
    plt.close()

    values_low = runs.loc[
        runs["method"] == "GA_mutation_0.15",
        "best_value"
    ].to_numpy()

    values_main = runs.loc[
        runs["method"] == "GA_mutation_0.30",
        "best_value"
    ].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [values_low, values_main],
        tick_labels=["p_mut=0.15", "p_mut=0.30"]
    )
    plt.ylabel("Итоговое лучшее значение f(x)")
    plt.title("Сравнение вероятностей мутации")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "mutation_comparison.png"),
        dpi=180
    )
    plt.close()

    random_values = runs.loc[
        runs["method"] == "Random_search",
        "best_value"
    ].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [values_main, random_values],
        tick_labels=["Генетический алгоритм", "Случайный поиск"]
    )
    plt.ylabel("Итоговое лучшее значение f(x)")
    plt.title("Сравнение со случайным поиском")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "ga_vs_random.png"),
        dpi=180
    )
    plt.close()


    print(summary.to_string(index=False))
    print("\nResults saved to:", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
