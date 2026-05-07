import shutil
from pathlib import Path
from helpers import write_results, load_data_fold, _resolve_model_cls

DEFAULT_HYPERTUNING_SETTING = {
        'n_hidden': [8, 16, 32, 64],
        'n_latent': 16,
        'learning_rate': [1e-3],
        'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
        'dropout_rate': [0.0, 0.2, 0.4],
        'patience': 50,
        'minimum': 100,
        'batch_size': [16, 128],
        'max_epochs': 1500,
    }

def _coerce(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)



def _build_tuner(model_cls_name, dataset_name, fold, directory, project_name,
                 max_trials, executions_per_trial, hypertuning_settings,
                 X_train, D_train, X_val, D_val,
                 extra_init_kwargs=None, extra_fit_kwargs=None,
                 overwrite=False):
    import keras
    import keras_tuner as kt

    from pyldl.utils import LDLEarlyStopping, LossHistory

    model_cls = _resolve_model_cls(model_cls_name)
    extra_init_kwargs = dict(extra_init_kwargs or {})
    extra_fit_kwargs  = dict(extra_fit_kwargs  or {})

    class _PyLDLHyperModel(kt.HyperModel):
        def build(self, hp):
            return model_cls(
                n_hidden=hp.Choice('n_hidden', hypertuning_settings[dataset_name]['n_hidden']),
                n_latent=hp.Fixed('n_latent', hypertuning_settings[dataset_name]['n_latent']),
                **extra_init_kwargs,
            )

        def fit(self, hp, model, **kwargs):
            lr       = hp.Choice('learning_rate', hypertuning_settings[dataset_name]['learning_rate'])
            wd       = hp.Choice('weight_decay', hypertuning_settings[dataset_name]['weight_decay'])
            dropout  = hp.Choice('dropout_rate', hypertuning_settings[dataset_name]['dropout_rate'])
            patience = hp.Fixed('patience', hypertuning_settings[dataset_name]['patience'])
            minimum  = hp.Fixed('minimum', hypertuning_settings[dataset_name]['minimum'])
            bs       = hp.Choice('batch_size', hypertuning_settings[dataset_name]['batch_size'])
            max_epochs = hp.Fixed('max_epochs', hypertuning_settings[dataset_name]['max_epochs'])

            keras.backend.clear_session()

            history = LossHistory()
            model.fit(
                X_train, D_train,
                X_val=X_val, D_val=D_val,
                epochs=max_epochs,
                batch_size=bs,
                optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd),
                dropout_rate=dropout,
                callbacks=[LDLEarlyStopping(monitor='kl_divergence',
                                            patience=patience, minimum=minimum),
                           history],
                verbose=0,
                **extra_fit_kwargs,
            )
            epochs_run = len(history.history.get('loss', []))
            scores = model.score(X_val, D_val,
                                 metrics=['kl_divergence'], return_dict=True)
            return {
                'kl_divergence': float(scores['kl_divergence']),
                'epochs_run': float(epochs_run),
            }

    return kt.BayesianOptimization(
        hypermodel=_PyLDLHyperModel(),
        objective=kt.Objective('kl_divergence', direction='min'),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        num_initial_points=6,
        directory=directory,
        project_name=project_name,
        overwrite=overwrite,
    )


def run_search(model_cls_name, dataset_name, fold,
               max_trials, executions_per_trial,
               hypertuning_settings=None,
               extra_init_kwargs=None, extra_fit_kwargs=None,
               val_split=0.2, random_state=0):
    """Run one BO search for (model_cls_name, dataset_name) on this worker's GPU.

    Loads the dataset inside the worker so large arrays don't get shipped
    across processes. Writes the best HPs + objective to disk and returns
    a small summary dict for the main process to aggregate.
    """
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    from sklearn.model_selection import train_test_split

    X, D, _, _ = load_data_fold(dataset_name, fold)
    X_train, X_valid, D_train, D_valid = train_test_split(
        X, D, test_size=val_split, random_state=random_state)

    directory =  Path(__file__).parent / 'hypertuning_dir'
    project_name = f'{model_cls_name}__{dataset_name}__{fold}'

    if hypertuning_settings is None:
        hypertuning_settings = {
            dataset_name: DEFAULT_HYPERTUNING_SETTING
        }

    tuner = _build_tuner(
        model_cls_name=model_cls_name, dataset_name=dataset_name, fold=fold,
        directory=directory, project_name=project_name,
        max_trials=max_trials, executions_per_trial=executions_per_trial,
        hypertuning_settings=hypertuning_settings,
        X_train=X_train, D_train=D_train, X_val=X_valid, D_val=D_valid,
        extra_init_kwargs=extra_init_kwargs, extra_fit_kwargs=extra_fit_kwargs,
    )
    tuner.search()

    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    epochs_run = best_trial.metrics.get_last_value('epochs_run')
    epochs_run = int(round(epochs_run)) if epochs_run is not None else None

    write_results(
        model=model_cls_name,
        dataset=dataset_name,
        fold=fold,
        results={
            'model': model_cls_name,
            'dataset': dataset_name,
            'val_kl_divergence': float(best_trial.score),
            'epochs_run': epochs_run,
            'hyperparameters': {k: _coerce(v) for k, v in dict(best_hp.values).items()},
        }
    )

    # delete tuner directory to save space; we have the best HPs + objective saved separately
    tuner_dir = Path(directory) / project_name
    if tuner_dir.exists():
        shutil.rmtree(tuner_dir)

    return {
        'model': model_cls_name,
        'dataset': dataset_name,
        'val_kl_divergence': float(best_trial.score),
        'epochs_run': epochs_run,
        'hyperparameters': dict(best_hp.values),
    }


def refit_best(model_cls_name, hyperparameters,
               X_train, D_train, X_val, D_val,
               extra_init_kwargs=None, extra_fit_kwargs=None,
               epochs=2500, minimum=50):
    """Refit a model with the best hyperparameters. `hyperparameters` may be
    either a `keras_tuner.HyperParameters` object or a plain dict (e.g.
    loaded from disk via `load_best_hyperparams(...)['hyperparameters']`).
    """
    import keras
    from pyldl.utils import LDLEarlyStopping

    model_cls = _resolve_model_cls(model_cls_name)
    extra_init_kwargs = dict(extra_init_kwargs or {})
    extra_fit_kwargs  = dict(extra_fit_kwargs  or {})

    if hasattr(hyperparameters, 'get') and not isinstance(hyperparameters, dict):
        get = hyperparameters.get  # keras_tuner.HyperParameters
    else:
        hp = dict(hyperparameters)
        get = hp.__getitem__

    model = model_cls(
        n_hidden=get('n_hidden'),
        n_latent=get('n_latent'),
        **extra_init_kwargs,
    )
    model.fit(
        X_train, D_train,
        X_val=X_val, D_val=D_val,
        epochs=epochs,
        batch_size=get('batch_size'),
        optimizer=keras.optimizers.AdamW(
            learning_rate=get('learning_rate'),
            weight_decay=get('weight_decay'),
        ),
        dropout_rate=get('dropout_rate'),
        callbacks=[LDLEarlyStopping(monitor='kl_divergence',
                                    patience=get('patience'),
                                    minimum=minimum)],
        verbose=0,
        **extra_fit_kwargs,
    )
    return model
