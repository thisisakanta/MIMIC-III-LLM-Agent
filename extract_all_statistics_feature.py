
##taken help from the logistic model training beforehand ##

import os
import numpy as np
import random

from readers import InHospitalMortalityReader
import common_utils
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import numpy as np
import pandas as pd
import argparse
import json


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    # Get feature names
    feature_names = common_utils.get_feature_names_from_header(ret['header'], period, features)
    return (X, ret['y'], ret['name'], ret['header'], feature_names)


##  this was added primarily to save the features ###

def save_features_to_csv(X, y, names, filename, feature_names=None):
    """Save features to CSV file"""
    df = pd.DataFrame(X)
    
    # Add column names if available
    if feature_names is not None:
        df.columns = feature_names
    else:
        df.columns = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Add patient identifiers and labels
    df.insert(0, 'patient_name', names)
    df.insert(1, 'mortality_label', y)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f'  Saved: {filename} (shape: {X.shape})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), './data/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--save_features', action='store_true', help='Save features at each preprocessing step')
    args = parser.parse_args()
    print(args)

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)

    print('Reading data and extracting features ...')
    (train_X, train_y, train_names, header, feature_names) = read_and_extract_features(train_reader, args.period, args.features)
    (val_X, val_y, val_names, _, _) = read_and_extract_features(val_reader, args.period, args.features)
    (test_X, test_y, test_names, _, _) = read_and_extract_features(test_reader, args.period, args.features)
    print('  train data shape = {}'.format(train_X.shape))
    print('  validation data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))
    print('  number of features = {}'.format(len(feature_names)))

    # Create directory for feature exports
    if args.save_features:
        feature_dir = os.path.join(args.output_dir, 'extracted_features')
        common_utils.create_directory(feature_dir)
        
        print('\n=== STEP 1: Saving RAW EXTRACTED FEATURES ===')
        save_features_to_csv(train_X, train_y, train_names, 
                            os.path.join(feature_dir, '01_train_raw_features.csv'),
                            feature_names=feature_names)
        save_features_to_csv(val_X, val_y, val_names, 
                            os.path.join(feature_dir, '01_val_raw_features.csv'),
                            feature_names=feature_names)
        save_features_to_csv(test_X, test_y, test_names, 
                            os.path.join(feature_dir, '01_test_raw_features.csv'),
                            feature_names=feature_names)
        
        # Save a small sample for inspection
        save_features_to_csv(train_X[:100], train_y[:100], train_names[:100],
                            os.path.join(feature_dir, '01_train_raw_features_sample100.csv'),
                            feature_names=feature_names)

    print('\nImputing missing values ...')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', copy=True)
    imputer.fit(train_X)
    train_X_imputed = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X_imputed = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X_imputed = np.array(imputer.transform(test_X), dtype=np.float32)

    if args.save_features:
        print('\n=== STEP 2: Saving IMPUTED FEATURES (missing values filled) ===')
        save_features_to_csv(train_X_imputed, train_y, train_names,
                            os.path.join(feature_dir, '02_train_imputed_features.csv'),
                            feature_names=feature_names)
        save_features_to_csv(val_X_imputed, val_y, val_names,
                            os.path.join(feature_dir, '02_val_imputed_features.csv'),
                            feature_names=feature_names)
        save_features_to_csv(test_X_imputed, test_y, test_names,
                            os.path.join(feature_dir, '02_test_imputed_features.csv'),
                            feature_names=feature_names)
        
        save_features_to_csv(train_X_imputed[:100], train_y[:100], train_names[:100],
                            os.path.join(feature_dir, '02_train_imputed_features_sample100.csv'),
                            feature_names=feature_names)

    print('\nNormalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X_imputed)
    train_X = scaler.transform(train_X_imputed)
    val_X = scaler.transform(val_X_imputed)
    test_X = scaler.transform(test_X_imputed)

    if args.save_features:
        print('\n=== STEP 3: Saving NORMALIZED FEATURES (standardized) ===')
        save_features_to_csv(train_X, train_y, train_names,
                            os.path.join(feature_dir, '03_train_normalized_features.csv'),
                            feature_names=feature_names)
        save_features_to_csv(val_X, val_y, val_names,
                            os.path.join(feature_dir, '03_val_normalized_features.csv'),
                            feature_names=feature_names)
        save_features_to_csv(test_X, test_y, test_names,
                            os.path.join(feature_dir, '03_test_normalized_features.csv'),
                            feature_names=feature_names)
        
        save_features_to_csv(train_X[:100], train_y[:100], train_names[:100],
                            os.path.join(feature_dir, '03_train_normalized_features_sample100.csv'),
                            feature_names=feature_names)
        
        # Save feature statistics
        print('\n=== Saving FEATURE STATISTICS ===')
        stats_df = pd.DataFrame({
            'feature_name': feature_names,
            'feature_index': range(train_X.shape[1]),
            'mean_after_scaling': np.mean(train_X, axis=0),
            'std_after_scaling': np.std(train_X, axis=0),
            'mean_before_scaling': imputer.statistics_,
            'min_value': np.min(train_X_imputed, axis=0),
            'max_value': np.max(train_X_imputed, axis=0)
        })
        stats_df.to_csv(os.path.join(feature_dir, '04_feature_statistics.csv'), index=False)
        print(f'  Saved: {os.path.join(feature_dir, "04_feature_statistics.csv")}')

if __name__ == '__main__':
    main()
