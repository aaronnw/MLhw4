import pandas as pd
import random
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier

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
    tv_set, test_set = split_input_data(df)
    decision_tree(tv_set, test_set)

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

def decision_tree(tv_set, test_set):
    training_set, validation_set = random_split(tv_set, .75)
    d_tree = DecisionTreeClassifier()
    fitted_tree = d_tree.fit(training_set[x_col], training_set[y_col])

    #Validation BS
    print("Validation BS")
    print(len(validation_set[x_col]))
    y_probs = fitted_tree.predict_proba(validation_set[x_col])
    print(y_probs)
    #Validation BSS

    #Test BS
    #Test BSS

    print(training_set.shape)
    print(validation_set.shape)



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

if __name__ == "__main__":
    load_data()
