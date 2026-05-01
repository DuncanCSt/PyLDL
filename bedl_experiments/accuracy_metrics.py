from helpers import write_results, load_data_fold, _resolve_model_cls, fit_best_model

def run_accuracy_metrics(model_cls_name, dataset_name, fold):
    """Run accuracy metrics for a given model, dataset, and fold. This is
    separate from the main hypertuning loop since we want to run it after
    refitting the best model on the full training set (train + val).
    """
    from pyldl.utils import kl_divergence
    from .helpers import load_data_fold, read_results, write_results

    X_train, D_train, X_test, D_test = load_data_fold(dataset_name, fold)
    results = read_results(model_cls_name, dataset_name, fold)
    hyperparameters = results['hyperparameters']
    
    model = fit_best_model(
        model_cls_name, dataset_name, fold
    )
    
    D_pred = model.score(X_test)

    write_results(
        model=model_cls_name,
        dataset=dataset_name,
        fold=fold,
        results={
            'test_kl_divergence': float(kl_div),
        }
    )