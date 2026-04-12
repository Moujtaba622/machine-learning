from src.utils import load_data, save_data
from src.train_model import train_logistic_model, evaluate_model
from src.preprocessing import apply_pca
from src.preprocessing import (
    drop_useless_columns,
    parse_registration_date,
    feature_engineering,
    impute_missing_values,
    encode_categorical,
    scale_data
    
)
from src.train_model import (
    split_features_target,
    train_test_split_data
)

DATA_PATH = "data/raw/retail.csv"

if __name__ == "__main__":

    # 1 Load
    df = load_data(DATA_PATH)

    # 2 Cleaning
    df = drop_useless_columns(df)
    df = parse_registration_date(df)
    df = feature_engineering(df)
    df = impute_missing_values(df)
    df = encode_categorical(df)

    # 3 Split
    X, y = split_features_target(df)
    print(X.columns)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 4 Scaling
    X_train, X_test = scale_data(X_train, X_test)
    X_train, X_test = apply_pca(X_train, X_test, n_components=20)

    print("Préparation terminée.")
    
model = train_logistic_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
