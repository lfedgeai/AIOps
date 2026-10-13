# app/utils.py
def extract_features(df):
    return df[['cpu_usage', 'memory_usage']].values
