from src.build.classify_articles import fit_clusters
import optuna


def objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 10, 80)
    n_components = trial.suggest_int("n_components", 5, 30)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
    min_cluster_size = trial.suggest_int("min_cluster_size", 4, 20)
    min_samples = trial.suggest_int("min_samples", 1, 5)
    cluster_epsilon = trial.suggest_float("cluster_epsilon", 0.0, 0.20)
    min_topic_size = trial.suggest_int("min_topic_size", 4, 20)
    threshold = trial.suggest_float("reassign_threshold", 0.05, 0.25)

    n_outliers = fit_clusters(
        n_neighbors, n_components, min_dist,
        min_cluster_size, min_samples, cluster_epsilon,
        min_topic_size, top_n_words=10,
        nr_bins=30, threshold=threshold
    )
    return n_outliers


def sweep():
    # Bayesian parameter sweep
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)  # Bayesian TPE
    study = optuna.create_study(direction="minimize",
                                sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=100, timeout=8 * 60 * 60)

if __name__ == "__main__":
    sweep()
