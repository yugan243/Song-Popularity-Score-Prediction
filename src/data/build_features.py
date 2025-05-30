import pandas as pd
import numpy as np

def create_date_features_train_test(train_df, test_df, date_col='publications_timestamp', target_col='target'):
    """
    Generate consistent date-based features for train and test sets, including popularity-aware features from train.
    
    Parameters:
        train_df (pd.DataFrame): Training data with the target column.
        test_df (pd.DataFrame): Test data without the target column.
        date_col (str): Timestamp column to extract features from.
        target_col (str): Target column used for popularity-based features in train.
    
    Returns:
        train_df, test_df: DataFrames with added time-based features.
    """
    def base_date_features(df):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        df['release_year'] = df[date_col].dt.year
        df['release_month'] = df[date_col].dt.month
        df['release_day'] = df[date_col].dt.day
        df['release_dayofweek'] = df[date_col].dt.dayofweek
        df['release_quarter'] = df[date_col].dt.quarter
        df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
        df['day_of_year'] = df[date_col].dt.dayofyear

        df['is_weekend_release'] = df['release_dayofweek'].isin([5, 6]).astype(int)
        df['is_summer_release'] = df['release_month'].isin([6, 7, 8]).astype(int)
        df['is_end_of_year'] = df['release_month'].isin([11, 12]).astype(int)
        df['is_valentines_week'] = ((df['release_month'] == 2) & (df['release_day'] <= 14)).astype(int)
        df['is_new_year_release'] = ((df['release_month'] == 1) & (df['release_day'] <= 7)).astype(int)

        df['month_sin'] = np.sin(2 * np.pi * df['release_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['release_month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['release_day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['release_day'] / 31)

        df['songs_released_that_year'] = df.groupby('release_year')[date_col].transform('count')
        df['songs_released_that_month'] = df.groupby(['release_year', 'release_month'])[date_col].transform('count')

        df['release_period_in_month'] = pd.cut(
            df['release_day'], bins=[0, 10, 20, 31], labels=['early', 'mid', 'late'], include_lowest=True
        )
        return df

    # Apply base feature creation
    train_df = base_date_features(train_df)
    test_df = base_date_features(test_df)

    # Popularity-aware features (learn from train, map to test)
    month_popularity = train_df.groupby('release_month')[target_col].mean().to_dict()
    dow_popularity = train_df.groupby('release_dayofweek')[target_col].mean().to_dict()

    train_df['release_month_popularity_score'] = train_df['release_month'].map(month_popularity)
    test_df['release_month_popularity_score'] = test_df['release_month'].map(month_popularity)

    train_df['release_dayofweek_popularity_score'] = train_df['release_dayofweek'].map(dow_popularity)
    test_df['release_dayofweek_popularity_score'] = test_df['release_dayofweek'].map(dow_popularity)

    return train_df, test_df
