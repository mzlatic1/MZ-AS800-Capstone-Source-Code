# Title: Wildfire Prediction Methodology Part C
# Author: Marko Zlatic
# Date: 2024-05-03 (yyyy-mm-dd)
# University: Johns Hopkins University
# Program: Master of Science in Geographic Information Systems (GIS)
# Purpose: This code was written to uphold the requirements for the AS800 GIS Capstone

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import datetime
import random
import optuna
import sys
import os

np.random.seed(42)
random.seed(42)


def part_c(master_folder):
    # Script Objective: Create a supervised machine learning model that is able to predict wildfire locations within a 5
    # kilometer radius

    output_datasets = os.path.join(master_folder, 'output_datasets')

    ml_parent_df = pd.read_parquet(os.path.join(output_datasets, "finalized_preprocessed_parent_df.parquet"))

    # Exploratory Fields
    x_keep_fields = [
        'num_points_within_meters_1000000',
        'STATE',
        'pop_dist_from_center',
        'camp_cluster_id',
        'cluster_id',
        'sdi_tba_dist_from_center',
        'land_area',
        'cluster_pop_dist_from_center_num_points_within_meters_1000000_cluster_id',
        'cluster_pop_dist_from_center_num_points_within_meters_1000000_dist_from_center',
        'cluster_pop_dist_from_center_sdi_tba_dist_from_center_dist_from_center',
        'dist_from_center',
        'cluster_pop_dist_from_center_sdi_tba_dist_from_center_cluster_id',
        'num_points_within_meters_100000',
        'dist_to_closest_state_boundary',
        'pop_cluster_id',
        'soil_type',
        'num_points_within_meters_10000',
        'water_area',
        'elevation',
        'population_density',
        'population_count'
    ]

    # Prediction Fields
    y_keep_fields = ['mN', 'mE']

    # Train/Test Split
    split_num = int(round(len(ml_parent_df.index) * 0.9))

    parent_train = ml_parent_df.reset_index(names='linear_oid').query(f"linear_oid < {split_num}")
    xtrain = parent_train[x_keep_fields]
    ytrain = parent_train[y_keep_fields]

    parent_test = ml_parent_df.reset_index(names='linear_oid').query(f"linear_oid >= {split_num}").reset_index(drop=True)

    xtest = parent_test[x_keep_fields]
    ytest = parent_test[y_keep_fields]

    # Run each model using base parameters
    print('Running models using base parameters...')
    preliminary_results = []
    models = {
        'Extra Tree Regressor': ExtraTreesRegressor,
        'Random Forest Regressor': RandomForestRegressor,
        'Bagging Regressor': BaggingRegressor,
        'XGBoost Regressor': XGBRegressor,
        'KNeighbors Regressor': KNeighborsRegressor
    }
    n_jobs = os.cpu_count() - 4
    for model in list(models.keys()):
        m = models[model](n_jobs=n_jobs)
        try:
            m.random_state = 42
        except:
            pass
        m.fit(xtrain, ytrain)

        base_pred = m.predict(xtest)

        r_squared = metrics.r2_score(ytest, base_pred)
        mean_absolute_error = metrics.mean_absolute_error(ytest, base_pred)
        root_mean_squared_error = metrics.root_mean_squared_error(ytest, base_pred)

        preliminary_results.append([model, r_squared, mean_absolute_error, root_mean_squared_error])

    print('Preliminary Results:\n', preliminary_results)
    pd.DataFrame(preliminary_results, columns=['Model', 'R2', 'MAE', 'RMSE']) \
        .to_csv(os.path.join(output_datasets, 'preliminary_results.csv'))

    # Optuna hyperparameter optimization using model with lowest RMSE and MAE (in this case, its the Extra Trees Regressor)

    def objective(trial):
        # Used exclusively for the Optuna Study class to help determine the optimal hyperparameters.
        # Code structure source can be found here: https://optuna.org/
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_int("max_depth", 300, 800)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        smla = ExtraTreesRegressor(n_jobs=n_jobs, random_state=42) # Supervised machine learning algorithm (SMLA)
        smla.max_depth = max_depth
        smla.n_estimators = n_estimators
        smla.min_samples_split = min_samples_split
        smla.min_samples_leaf = min_samples_leaf

        smla.fit(xtrain, ytrain)
        preds = smla.predict(xtest)
        return metrics.root_mean_squared_error(ytest, preds)

    print('Running hyperparameter tuning...')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)
    best_params = study.best_params
    print('Best hyperparameters:', best_params)

    # Execute Extra Trees Regressor with optimized hyperparameters
    print('Running ETR with optimized hyperparameters...')
    etr = ExtraTreesRegressor(
        n_jobs=n_jobs,
        random_state=42,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf']
    )

    etr.fit(xtrain, ytrain)

    pred = etr.predict(xtest)

    rmse = metrics.root_mean_squared_error(ytest, pred)
    print(f"RMSE: {rmse}")

    mae = metrics.mean_absolute_error(ytest, pred)
    print(f"MAE: {mae}")

    r2 = metrics.r2_score(ytest, pred)
    print(f"R2: {r2}")

    # Plot feature importance
    print('Plotting feature importance...')
    features = xtrain.columns
    importance = etr.feature_importances_

    feature_importance = pd.Series(importance, features).sort_values(ascending=False)
    plt.figure(figsize=(10, 10))
    ax = feature_importance.plot(kind='bar', title='Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot learning curves (helps determine overfitting)
    # Source code found here: https://github.com/ageron/handson-ml3/blob/main/04_training_linear_models.ipynb
    print('Generating Learning Curves...')
    train_sizes, train_scores, valid_scores = learning_curve(
        etr,
        ml_parent_df[x_keep_fields], ml_parent_df[y_keep_fields],
        train_sizes=np.linspace(0.01, 1, 40),
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    train_errors = -train_scores.mean(axis=1)
    valid_errors = -valid_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors, 'r-+', linewidth=2, label='Train Subset')
    plt.plot(train_sizes, valid_errors, 'b-', linewidth=3, label='Valid Subset')
    plt.title('Learning curve results for Extra Tree Regressor')
    plt.xlabel('Training Set Size')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.legend()
    plt.show()

    # Export predictions for the Part D script
    print('Exporting predictions...')
    state_encoder = pd\
        .read_csv(os.path.join(output_datasets, 'state_ordinal_encoder.csv'))['STATE']\
        .to_dict() # Reverse the ordinal encoder that was applied to the State abbreviation field
    parent_test['STATE'] = parent_test['STATE'].apply(lambda state: state_encoder[state])

    pd \
        .merge(pd.DataFrame(pred, columns=['pred_' + f for f in y_keep_fields]),
               parent_test[['STATE', 'FireDiscoveryDateTime', 'orig_oid', 'mN', 'mE']] \
               .rename(columns={'mN': 'actual_mN', 'mE': 'actual_mE'}),
               left_index=True,
               right_index=True
               )\
        .to_csv(os.path.join(output_datasets, "finalized_predictions.csv"))

    return os.path.split(output_datasets)[0]


if __name__ == '__main__':
    if sys.argv[1] is not None:
        start_time = datetime.datetime.now()
        part_c(sys.argv[1])
        end_time = datetime.datetime.now()
        print('Script complete! Duration:', (end_time - start_time).total_seconds() / 60, 'minutes.')
        sys.exit(0)
    else:
        print('ERROR: Invalid/absent arguments.')
        sys.exit(1)
