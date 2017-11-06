import pandas as pd
import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import brier_score_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error

datafile = "hw4data.csv"

x_col = ["lapse_rate_0to3km_k_m01", "wind_mean_0to6km_magnitude_m_s01", "lapse_rate_850to500mb_k_m01",
         "srw_0to1km_magnitude_m_s01", "echo_top_50dbz_mean_metres", "wind_mean_ebl_cosine",
         "reflectivity_m10celsius_max_dbz", "reflectivity_lowest_altitude_prctile25gradient_dbz_m01",
         "lapse_rate_700to500mb_k_m01", "reflectivity_lowest_altitude_percentile95_dbz",
         "low_level_shear_stdev_s01", "wind_mean_0to6km_cosine", "enhanced_stretching_potential",
         "microburst_composite_param", "cape_0to6km_j_kg01", "theta_e_difference_kelvins", "derecho_composite_param",
         "lifted_index_surface_to_500mb_kelvins", "wind_shear_0to8km_magnitude_m_s01", "vil_gradient_percentile25_mm"]

train_validate_size = 10000
testing_size = 2000
m_s_severe = 25.7222

num_trees_list = [1, 2, 3, 5, 10]
max_depth_list = [1, 2, 3, 5, 10]

def load_data():
    df = pd.read_csv(datafile, header=0)
    df = preprocess(df)
    #Create a new binary column to track whether the wind was severe
    df = df.assign(severe_wind=df.max_wind_speed_m_s01.apply(lambda x: 1 if x > m_s_severe else 0))
    return df


def preprocess(df):
    #Deafualt to replace Nan with mean
    imp = Imputer()
    imp_df = imp.fit_transform(df)
    imp_df = pd.DataFrame(columns=df.columns.values, data=imp_df)
    return imp_df


def split_input_data(df):
    tv_set = df.take(range(train_validate_size))
    test_set = df.take(range(train_validate_size, train_validate_size+testing_size))
    return tv_set, test_set


def decision_tree(df):
    y_col = 'severe_wind'
    tv_set, test_set = split_input_data(df)
    training_set, validation_set = random_split(tv_set, .75)
    d_tree = DecisionTreeClassifier(criterion="entropy")
    fitted_tree = d_tree.fit(training_set[x_col], training_set[y_col])
    # Frequency of severe wind for BSS calc
    counts = training_set[y_col].value_counts().to_dict()
    freq_pos = counts[1] / (counts[0] + counts[1])

    print("Decision Tree")
    # Validation stuff
    y_true = validation_set[y_col]
    y_pred = fitted_tree.predict_proba(validation_set[x_col])[:, 1]
    # Validation BS
    print("Validation BS")
    valid_bs = brier_score_loss(y_true, y_pred)
    print(valid_bs)
    # Validation BSS
    print("Validation BSS")
    valid_bss = calc_BSS(y_true, y_pred, freq_pos)
    print(valid_bss)

    # Test stuff
    y_true = test_set[y_col]
    y_pred = fitted_tree.predict_proba(test_set[x_col])[:, 1]
    # Test BS
    print("Test BS")
    test_bs = brier_score_loss(y_true, y_pred)
    print(test_bs)
    # Test BSS
    print("Test BSS")
    test_bss = calc_BSS(y_true, y_pred, freq_pos)
    print(test_bss)
    print("\n")


def random_forest(df):
    y_col = 'severe_wind'
    tv_set, test_set = split_input_data(df)
    training_set, validation_set = random_split(tv_set, .75)
    forest = RandomForestClassifier(criterion="entropy", n_estimators=500)
    forest.fit(training_set[x_col], training_set[y_col])
    counts = training_set[y_col].value_counts().to_dict()
    freq_pos = counts[1] / (counts[0] + counts[1])

    print("Random Forest")
    # Validation stuff
    y_true = validation_set[y_col]
    y_pred = forest.predict_proba(validation_set[x_col])[:, 1]
    # Validation BS
    print("Validation BS")
    valid_bs = brier_score_loss(y_true, y_pred)
    print(valid_bs)
    # Validation BSS
    print("Validation BSS")
    valid_bss = calc_BSS(y_true, y_pred, freq_pos)
    print(valid_bss)

    # Test stuff
    y_true = test_set[y_col]
    y_pred = forest.predict_proba(test_set[x_col])[:, 1]
    # Test BS
    print("Test BS")
    test_bs = brier_score_loss(y_true, y_pred)
    print(test_bs)
    # Test BSS
    print("Test BSS")
    test_bss = calc_BSS(y_true, y_pred, freq_pos)
    print(test_bss)
    print("\n")


def gradient_boosted_trees(df):
    y_col = 'severe_wind'
    tv_set, test_set = split_input_data(df)
    training_set, validation_set = random_split(tv_set, .75)
    gradient_trees = GradientBoostingClassifier(n_estimators=500, loss="exponential")
    gradient_trees.fit(training_set[x_col], training_set[y_col])
    counts = training_set[y_col].value_counts().to_dict()
    freq_pos = counts[1] / (counts[0] + counts[1])

    print("Gradient Boosted Trees")
    # Validation stuff
    y_true = validation_set[y_col]
    y_pred = gradient_trees.predict_proba(validation_set[x_col])[:, 1]
    # Validation BS
    print("Validation BS")
    valid_bs = brier_score_loss(y_true, y_pred)
    print(valid_bs)
    # Validation BSS
    print("Validation BSS")
    valid_bss = calc_BSS(y_true, y_pred, freq_pos)
    print(valid_bss)

    # Test stuff
    y_true = test_set[y_col]
    y_pred = gradient_trees.predict_proba(test_set[x_col])[:, 1]
    # Test BS
    print("Test BS")
    test_bs = brier_score_loss(y_true, y_pred)
    print(test_bs)
    # Test BSS
    print("Test BSS")
    test_bss = calc_BSS(y_true, y_pred, freq_pos)
    print(test_bss)


def regression_tree(df):
    y_col = 'max_wind_speed_m_s01'
    tv_set, test_set = split_input_data(df)
    training_set, validation_set = random_split(tv_set, .75)
    dec_tree_reg = DecisionTreeRegressor()
    fitted_regressor = dec_tree_reg.fit(training_set[x_col], training_set[y_col])

    y_true = validation_set[y_col]
    y_pred = fitted_regressor.predict(validation_set[x_col])

    print("Regression tree")
    print("Validation MAE")
    valid_mae = mean_absolute_error(y_true, y_pred)
    print(valid_mae)
    print("Validation RMSE")
    valid_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(valid_rmse)

    y_true = test_set[y_col]
    y_pred = fitted_regressor.predict(test_set[x_col])
    print("Test MAE")
    valid_mae = mean_absolute_error(y_true, y_pred)
    print(valid_mae)
    print("Test RMSE")
    valid_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(valid_rmse)
    print("\n")


def regression_forest(df):
    y_col = 'max_wind_speed_m_s01'
    tv_set, test_set = split_input_data(df)
    training_set, validation_set = random_split(tv_set, .75)
    dec_tree_reg = RandomForestRegressor(n_estimators=500)
    fitted_regressor = dec_tree_reg.fit(training_set[x_col], training_set[y_col])

    y_true = validation_set[y_col]
    y_pred = fitted_regressor.predict(validation_set[x_col])
    print("Regression Forest")
    print("Validation MAE")
    valid_mae = mean_absolute_error(y_true, y_pred)
    print(valid_mae)
    print("Validation RMSE")
    valid_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(valid_rmse)

    y_true = test_set[y_col]
    y_pred = fitted_regressor.predict(test_set[x_col])
    print("Test MAE")
    valid_mae = mean_absolute_error(y_true, y_pred)
    print(valid_mae)
    print("Test RMSE")
    valid_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(valid_rmse)
    print("\n")


def gb_regressor(df):
    y_col = 'max_wind_speed_m_s01'
    tv_set, test_set = split_input_data(df)
    training_set, validation_set = random_split(tv_set, .75)
    dec_tree_reg = GradientBoostingRegressor(n_estimators=500)
    fitted_regressor = dec_tree_reg.fit(training_set[x_col], training_set[y_col])

    y_true = validation_set[y_col]
    y_pred = fitted_regressor.predict(validation_set[x_col])
    print("Regression Forest")
    print("Validation MAE")
    valid_mae = mean_absolute_error(y_true, y_pred)
    print(valid_mae)
    print("Validation RMSE")
    valid_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(valid_rmse)

    y_true = test_set[y_col]
    y_pred = fitted_regressor.predict(test_set[x_col])
    print("Test MAE")
    valid_mae = mean_absolute_error(y_true, y_pred)
    print(valid_mae)
    print("Test RMSE")
    valid_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(valid_rmse)
    print("\n")


def grid_search_forest(df):
    global num_trees_list, max_depth_list
    y_col = 'severe_wind'
    tv_set, test_set = split_input_data(df)
    models = list(itertools.product(num_trees_list, max_depth_list))
    valid_bs_dict = {}
    valid_bss_dict = {}
    test_bs_dict = {}
    test_bss_dict = {}
    for m in models:
        num_trees = m[0]
        max_depth = m[1]
        for i in range(30):
            print(m)
            training_set, validation_set = random_split(tv_set, .75)
            forest = RandomForestClassifier(criterion="entropy", n_estimators=num_trees)
            forest.fit(training_set[x_col], training_set[y_col])
            counts = training_set[y_col].value_counts().to_dict()
            freq_pos = counts[1] / (counts[0] + counts[1])

            # Validation stuff
            y_true = validation_set[y_col]
            y_pred = forest.predict_proba(validation_set[x_col])[:, 1]
            # Validation BS
            valid_bs = brier_score_loss(y_true, y_pred)
            # Validation BSS
            valid_bss = calc_BSS(y_true, y_pred, freq_pos)
            # Test stuff
            y_true = test_set[y_col]
            y_pred = forest.predict_proba(test_set[x_col])[:, 1]
            # Test BS
            test_bs = brier_score_loss(y_true, y_pred)
            # Test BSS
            test_bss = calc_BSS(y_true, y_pred, freq_pos)

            valid_bs_dict.setdefault(m, []).append(valid_bs)
            valid_bss_dict.setdefault(m, []).append(valid_bss)
            test_bs_dict.setdefault(m, []).append(test_bs)
            test_bss_dict.setdefault(m, []).append(test_bss)

    for key in valid_bs_dict:
        valid_bs_dict[key] = np.mean(valid_bs_dict[key])
    for key in valid_bss_dict:
        valid_bss_dict[key] = np.mean(valid_bss_dict[key])
    for key in test_bs_dict:
        test_bs_dict[key] = np.mean(test_bs_dict[key])
    for key in test_bss_dict:
        test_bss_dict[key] = np.mean(test_bss_dict[key])

    valid_bs_list = np.array(list(valid_bs_dict.values()))
    z = valid_bs_list.reshape((len(num_trees_list), len(max_depth_list)))
    plt.figure()
    contour_plot = plt.contourf(max_depth_list, num_trees_list, z, extend="both")
    plt.clabel(contour_plot, inline=True, fontsize=10)
    plt.title("Brier Score for Classification Forest")
    plt.xlabel("Max depth")
    plt.ylabel("Num trees")
    plt.colorbar(contour_plot).set_label("Brier Score")

    valid_bss_list = np.array(list(valid_bss_dict.values()))
    z = valid_bss_list.reshape((len(num_trees_list), len(max_depth_list)))
    plt.figure()
    contour_plot = plt.contourf(max_depth_list, num_trees_list, z, extend="both")
    plt.clabel(contour_plot, inline=True, fontsize=10)
    plt.title("Brier Skill Score for Classification Forest")
    plt.xlabel("Max depth")
    plt.ylabel("Num trees")
    plt.colorbar(contour_plot).set_label("Brier Skill Score")

    min_test_bs = min(test_bs_dict.values())
    max_test_bss = max(test_bss_dict.values())
    print("Best test bs")
    print(min_test_bs)
    print("Best test bss")
    print(max_test_bss)

def random_split(df, ratio):
    ### Split the dataset into training and validation
    # Number of elements in the 1st set
    num_validation_set = int(ratio * len(df.index))
    # Which elements to split off
    index_list = random.sample(range(len(df.index)), num_validation_set)
    set1, set2 = split_dataframe(df, index_list)
    return set1, set2


def split_dataframe(df, index_list):
    pulled_df = df.take(index_list)
    source_df = df.drop(df.index[index_list],inplace = False)
    return pulled_df, source_df


def calc_BSS(y_true, y_pred, freq_pos):
    bs = brier_score_loss(y_true, y_pred)
    freq_series = pd.Series([freq_pos for i in range(len(y_pred))])
    bs_pos_class = brier_score_loss(y_true, freq_series)
    return (bs_pos_class - bs)/bs_pos_class


if __name__ == "__main__":
    df = load_data()
    # decision_tree(df)
    # random_forest(df)
    # gradient_boosted_trees(df)
    #regression_tree(df)
    #regression_forest(df)
    #gb_regressor(df)

    # Part 5 using Random forest for regression
    grid_search_forest(df)
    plt.show()

