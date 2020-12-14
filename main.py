# import library
from numpy.ma import count
import pandas as pd
from sklearn import preprocessing, tree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tabulate import tabulate
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
pdtabulate = lambda df: tabulate(df, headers='keys', tablefmt='psql')


def data_cleaning():
    # establish dataframe for both train and test data

    # data source
    source_train, source_test = pd.read_csv('train.csv'), pd.read_csv('test.csv')

    print("Read train and test data ")
    print("Training data (show first 5 rows)", "\n", pdtabulate(source_train.head()))
    print("Testing data (show first 5 rows)", "\n", pdtabulate(source_test.head()))

    print("Data cleaning begins")
    print("Check if there are any missing values")
    print("Columns in Training data", "\n", source_train.isnull().any())
    print("Columns in Testing data", "\n", source_test.isnull().any())

    # create dictionary to store the keys in each string type columns
    train_dictionary_for_feature, test_dictionary_for_feature = {}, {}

    # store all columns name
    train_column_name, test_column_name = source_train.columns, source_test.columns

    print("Columns in training data", train_column_name)
    print("Columns in testing data", test_column_name)

    # check difference
    difference_columns_name = list(
        list(set(train_column_name) - set(test_column_name))
        + list(set(test_column_name) - set(train_column_name))
    )
    print("Check difference between two list of columns names", "\n", difference_columns_name)

    # store all feature name and label name
    feature_name, label_name = test_column_name, difference_columns_name
    print("Feature columns:", "\n", feature_name)
    print("Label columns", "\n", label_name)

    # check data type
    print("Checking data type")
    print("The data type of each columns in source_train", "\n", source_train.dtypes)
    print("The data type of each columns in source_test", "\n", source_test.dtypes)

    # separate string and integers into different dataframe
    train_int, train_obj = source_train.select_dtypes(include='int64'), source_train.select_dtypes(include='object')
    test_int, test_obj = source_test.select_dtypes(include='int64'), source_test.select_dtypes(include='object')

    print("Separate different data type ")
    print("The integer columns in source_train (show first 5 rows)", "\n", pdtabulate(train_int.head()))
    print("The string columns in source_train (show first 5 rows)", "\n", pdtabulate(train_obj.head()))
    print("The integer columns in source_test (show first 5 rows)", "\n", pdtabulate(test_int.head()))
    print("The string columns in source_test (show first 5 rows)", "\n", pdtabulate(test_obj.head()))

    # store all name of the object-type columns
    train_obj_name, test_obj_name = train_obj.columns, test_obj.columns

    # change string to integer for each text columns
    for i in range(0, count(train_obj_name)):
        # extract the unique columns values
        train_columns_unique_value = train_obj[train_obj_name[i]].unique()
        test_columns_unique_value = test_obj[test_obj_name[i]].unique()

        # establish a train dictionary (Key:each unique values in columns, Value: continuous numbers)
        enum_train = enumerate(train_columns_unique_value)
        train_dict_value = dict((j, i) for i, j in enum_train)
        train_dictionary_for_feature[train_obj_name[i]] = train_dict_value

        # establish a test dictionary (Key:each unique values in columns, Value: continuous numbers)
        enum_test = enumerate(test_columns_unique_value)
        test_dict_value = dict((j, i) for i, j in enum_test)
        test_dictionary_for_feature[test_obj_name[i]] = test_dict_value

        # change the text to the corresponding numbers
        train_obj[train_obj_name[i]] = train_obj[train_obj_name[i]].map(train_dict_value)
        test_obj[test_obj_name[i]] = test_obj[test_obj_name[i]].map(train_dict_value)

    # Iterate over key / value pairs of parent dictionary
    print("Nested dictionary for string columns in training data")
    for key, value in train_dictionary_for_feature.items():
        print(key, 'dictionary')
        # Again iterate over the nested dictionary
        for sub_key, sub_value in value.items():
            print(sub_key, ':', sub_value)

    print("Nested dictionary for string columns in testing data")
    for key, value in test_dictionary_for_feature.items():
        print(key, 'dictionary')
        # Again iterate over the nested dictionary
        for sub_key, sub_value in value.items():
            print(sub_key, ':', sub_value)

    # combine both integer data frame and text data frame to become a pure integer dataframe
    train_pure_int = pd.concat([train_int, train_obj], axis=1, sort=False).reindex(columns=train_column_name)
    print("Covert to integer:")
    print("Training data (show first 5 rows)", "\n", pdtabulate(train_pure_int.head()))

    test_pure_int = pd.concat([test_int, test_obj], axis=1, sort=False).reindex(columns=test_column_name)
    print("Testing data (show first 5 rows)", "\n", pdtabulate(test_pure_int.head()))

    print("Checking if there are any missing values")
    print("The total amount of missing values in train_pure_int", "\n", train_pure_int.isnull().sum())
    print("The total amount of missing values in test_pure_int", "\n", test_pure_int.isnull().sum())

    # native-country has some blank data
    # the reason is the string(s) in this column cannot match the key in temporary dictionary
    # the value (99) is to be added in these missing values.

    print("Verify the amount of key values in both training data and test data ")
    print("The amount of key values in training data", "\n", len(train_dictionary_for_feature["native-country"]))
    print("The amount of key values in testing data", "\n", len(test_dictionary_for_feature["native-country"]))

    missing_series = pd.isnull(test_pure_int["native-country"])
    print("The rows with missing values", "\n", pdtabulate(test_pure_int[missing_series]))
    print("Input missing values to be 99")

    miss_input = int(99)
    test_pure_int = test_pure_int.fillna(miss_input)
    test_dictionary_for_feature[''] = miss_input
    print("Refreshing...")
    print(pdtabulate(test_pure_int[missing_series]))
    print("Check again...")
    print("The string columns in test_pure_int")
    print(test_pure_int.isnull().sum())

    # Establish source data
    x_train = train_pure_int.drop(columns="exceeds50K")
    y_train = train_pure_int.iloc[:, -1:]
    x_test = test_pure_int
    print("Overview of x_train (show first 5 rows):", "\n", pdtabulate(x_train.head()))
    print("Overview of y_train (show first 5 rows):", "\n", pdtabulate(y_train.head()))
    print("Overview of x_test (show first 5 rows):", "\n", pdtabulate(x_test.head()))

    print("Data cleaning is completed.")

    # Bar chart
    y_train['exceeds50K'].value_counts().plot(kind='bar')
    print("")
    print(y_train['exceeds50K'].value_counts())
    plt.title("")
    plt.show()


    # Histogram
    print("Histogram is going to be generated")
    x1 = source_train['relationship'].to_numpy()
    x2 = source_test['relationship'].to_numpy()

    plt.hist([x1, x2], 10, label=['Train data', 'Test data'])
    plt.legend(loc='upper right')
    plt.title("The relationship ")
    plt.show()

    # Scatter
    print("Scatter is going to be generated")
    source_train.plot(kind='scatter', x='age', y='capital-gain', color='red')
    source_test.plot(kind='scatter', x='age', y='capital-gain', color='blue')
    plt.show()

    return x_train, y_train, x_test, feature_name, source_test


def choose():
    learning = input("Please choose your learning. (decision tree/knn/svm)")
    if learning == "decision tree":
        decision_tree()
    if learning == "knn":
        knn()
    if learning == "svm":
        svm()


def normalization():
    x_train, y_train, x_test, feature_name, source_test = data_cleaning()

    # normalization
    print("Normalization is required")
    min_max = preprocessing.MinMaxScaler()
    x_train_scaled, y_train_scaled = min_max.fit_transform(x_train), min_max.fit_transform(y_train)
    x_test_scaled = min_max.fit_transform(x_test)

    df_x_train_scaled = pd.DataFrame(data=x_train_scaled, columns=feature_name)
    df_y_train_scaled = pd.DataFrame(data=y_train_scaled)
    df_x_test_scaled = pd.DataFrame(data=x_test_scaled, columns=feature_name)

    print("Overview of x_train_scaled", "\n", pdtabulate(df_x_train_scaled.head()))
    print("Overview of y_train_scaled", "\n", pdtabulate(df_y_train_scaled.head()))
    print("Overview of x_test_scaled", "\n", pdtabulate(df_x_test_scaled.head()))

    return df_x_train_scaled, df_y_train_scaled, df_x_test_scaled, feature_name, source_test


def decision_tree():
    # value
    x_train, y_train, x_test, feature_name, source_test = data_cleaning()

    # model
    model_decision_tree = DecisionTreeClassifier(max_depth=8)
    model_decision_tree.fit(x_train, y_train)
    test_predict = model_decision_tree.predict(x_test)

    r = export_text(model_decision_tree, feature_names=feature_name.tolist())
    print(r)

    # save
    filename = "test_predict_decision_tree_max_depth_default"
    save(source_test, test_predict, filename)


def knn():
    # value
    df_x_train_scaled, df_y_train_scaled, df_x_test_scaled, feature_name, source_test = normalization()

    # model
    model_knn = KNeighborsClassifier(n_neighbors=157)
    model_knn.fit(df_x_train_scaled, df_y_train_scaled.values.ravel())
    test_predict = model_knn.predict(df_x_test_scaled)

    # save
    filename = "test_predict_knn_neighbors_155"
    save(source_test, test_predict, filename)


def svm():
    # value
    df_x_train_scaled, df_y_train_scaled, df_x_test_scaled, feature_name, source_test = normalization()

    # model
    model_svm = SVC(kernel="poly")
    model_svm.fit(df_x_train_scaled, df_y_train_scaled.values.ravel())
    test_predict = model_svm.predict(df_x_test_scaled)

    # save
    filename = "test_predict_svm_rbf"
    save(source_test, test_predict, filename)


def save(source_test, test_predict, filename):
    # save to the files
    new_test = source_test
    new_test['exceeds50K'] = test_predict
    new_test.to_csv('new_test.csv', index=False, header=True)
    test_label = new_test['exceeds50K'].astype(int)

    id_no = [i for i in range(1, 24422)]
    id_no_df = pd.DataFrame(id_no)

    df = pd.concat([id_no_df, test_label], axis=1, join='inner')
    df.columns = ['id', 'prediction']
    df.to_csv(filename, index=False, header=True)
    print(df)
    print(df['prediction'].value_counts())
    df['prediction'].value_counts().plot(kind='bar')
    plt.show()


if __name__ == '__main__':
    choose()
