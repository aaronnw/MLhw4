import pandas as pd
import random
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import brier_score_loss

datafile = "hw4data.csv"

x_col = ["lapse_rate_0to3km_k_m01", "wind_mean_0to6km_magnitude_m_s01", "lapse_rate_850to500mb_k_m01",
         "srw_0to1km_magnitude_m_s01", "echo_top_50dbz_mean_metres", "wind_mean_ebl_cosine",
         "reflectivity_m10celsius_max_dbz", "reflectivity_lowest_altitude_prctile25gradient_dbz_m01",
         "lapse_rate_700to500mb_k_m01", "reflectivity_lowest_altitude_percentile95_dbz",
         "low_level_shear_stdev_s01", "wind_mean_0to6km_cosine", "enhanced_stretching_potential",
         "microburst_composite_param", "cape_0to6km_j_kg01", "theta_e_difference_kelvins", "derecho_composite_param",
         "lifted_index_surface_to_500mb_kelvins", "wind_shear_0to8km_magnitude_m_s01", "vil_gradient_percentile25_mm"]

y_col = 'severe_wind'


train_validate_size = 10000
testing_size = 2000
m_s_severe = 25.7222

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
    decision_tree(df)
    random_forest(df)
    gradient_boosted_trees(df)

