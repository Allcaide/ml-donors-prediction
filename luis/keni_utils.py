import numpy as np
import pandas as pd


def clean_before_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic cleaning — no statistics computed from data.
    Safe to apply to the full dataset before train/val split.
    Covers: impossible value removal, range clipping, binary flags, category mapping.
    """
    df = df.copy()

    # --- Categorical fixes (deterministic) ---
    df['DONOR_GENDER'] = df['DONOR_GENDER'].fillna('U').str.upper()

    df['HOME_OWNER'] = df['HOME_OWNER'].fillna('U').str.upper()
    df['HOME_OWNER'] = df['HOME_OWNER'].map({'H': 1, 'U': 0}).astype(int)

    df['RECENCY_STATUS_96NK'] = df['RECENCY_STATUS_96NK'].fillna('U').str.upper()

    df['SES'] = df['SES'].replace('?', np.nan).astype(float)

    df['URBANICITY'] = df['URBANICITY'].replace('?', None).fillna('Unknown').str.upper()

    # --- PEP_STAR: clip impossible values + missingness flag ---
    df['PEP_STAR'] = df['PEP_STAR'].apply(lambda x: np.nan if x < 0 else (1.0 if x > 1 else x))
    df['PEP_STAR_IS_MISSING'] = np.where(df['PEP_STAR'].isna(), 1, 0)
    df['PEP_STAR'] = np.where(df['PEP_STAR'] > 0.5, 1, 0).astype(int)

    # --- FREQUENCY_STATUS_97NK: clip to [1, 4] + missingness flag ---
    df['FREQUENCY_STATUS_97NK'] = df['FREQUENCY_STATUS_97NK'].apply(lambda x: np.nan if x < 1 else (4.0 if x > 4 else x))
    df['FREQUENCY_STATUS_97NK_IS_MISSING'] = np.where(df['FREQUENCY_STATUS_97NK'].isna(), 1, 0)
    df['FREQUENCY_STATUS_97NK'] = df['FREQUENCY_STATUS_97NK'].fillna(-1).round().astype(int)

    # --- CHILDREN: clip negatives + cap at 4 + missingness flag ---
    df['CHILDREN'] = df['CHILDREN'].apply(lambda x: np.nan if x < 0 else (4.0 if x > 4 else x))
    df['CHILDREN_IS_MISSING'] = np.where(df['CHILDREN'].isna(), 1, 0)
    df['CHILDREN'] = df['CHILDREN'].fillna(0).round().astype(int)

    # --- INCOME_GROUP: clip to valid range [1, 7] ---
    df['INCOME_GROUP'] = np.where(df['INCOME_GROUP'] > 7, 7, df['INCOME_GROUP'])
    df['INCOME_GROUP'] = np.where((df['INCOME_GROUP'] < 1) | (df['INCOME_GROUP'].isna()), -1, df['INCOME_GROUP'])
    df['INCOME_GROUP'] = df['INCOME_GROUP'].round().astype(int)

    # --- WEALTH_RATING: clip to [0, 9] ---
    df['WEALTH_RATING'] = np.where(df['WEALTH_RATING'] > 9, 9.0, df['WEALTH_RATING'])
    df['WEALTH_RATING'] = np.where((df['WEALTH_RATING'] < 0) | (df['WEALTH_RATING'].isna()), -1.0, df['WEALTH_RATING'])
    df['WEALTH_RATING'] = df['WEALTH_RATING'].round().astype(int)

    # --- RECENT_CARD_RESPONSE_COUNT: clip negatives + missingness flag ---
    df['RECENT_CARD_RESPONSE_COUNT_IS_MISSING'] = np.where(df['RECENT_CARD_RESPONSE_COUNT'].isna(), 1, 0)
    df['RECENT_CARD_RESPONSE_COUNT'] = np.where(
        df['RECENT_CARD_RESPONSE_COUNT'] > 0,
        df['RECENT_CARD_RESPONSE_COUNT'].round(),
        df['RECENT_CARD_RESPONSE_COUNT']
    )
    df['RECENT_CARD_RESPONSE_COUNT'] = np.where(
        (df['RECENT_CARD_RESPONSE_COUNT'] < 0) | (df['RECENT_CARD_RESPONSE_COUNT'].isna()),
        0, df['RECENT_CARD_RESPONSE_COUNT']
    ).astype(int)

    # --- RECENT_RESPONSE_COUNT: clip negatives + missingness flag ---
    df['RECENT_RESPONSE_COUNT_IS_MISSING'] = np.where(df['RECENT_RESPONSE_COUNT'].isna(), 1, 0)
    df['RECENT_RESPONSE_COUNT'] = np.where(
        df['RECENT_RESPONSE_COUNT'] > 0,
        df['RECENT_RESPONSE_COUNT'].round(),
        df['RECENT_RESPONSE_COUNT']
    )
    df['RECENT_RESPONSE_COUNT'] = np.where(
        (df['RECENT_RESPONSE_COUNT'] < 0) | (df['RECENT_RESPONSE_COUNT'].isna()),
        0, df['RECENT_RESPONSE_COUNT']
    ).astype(int)

    # --- CARD_PROM_12: round + fillna(0) + missingness flag ---
    df['CARD_PROM_12_IS_MISSING'] = np.where(df['CARD_PROM_12'].isna(), 1, 0)
    df['CARD_PROM_12'] = np.where(df['CARD_PROM_12'].notna(), df['CARD_PROM_12'].round(), df['CARD_PROM_12'])
    df['CARD_PROM_12'] = df['CARD_PROM_12'].fillna(0).astype(int)

    # --- RECENT_STAR_STATUS: binarize ---
    df['RECENT_STAR_STATUS'] = np.where(df['RECENT_STAR_STATUS'].isna(), 1, 0)
    df['RECENT_STAR_STATUS'] = np.where(df['RECENT_STAR_STATUS'] < 0, 0, df['RECENT_STAR_STATUS'])
    df['RECENT_STAR_STATUS'] = np.where(df['RECENT_STAR_STATUS'] > 0, 1, df['RECENT_STAR_STATUS'])
    df['RECENT_STAR_STATUS'] = df['RECENT_STAR_STATUS'].astype(int)

    # --- DONOR_AGE: missingness flag only (imputation happens in pipeline) ---
    df['DONOR_AGE_IS_MISSING'] = np.where(df['DONOR_AGE'].isna(), 1, 0)

    # --- Discrete cols: clip negatives (no imputation here) ---
    discrete_cols = [
        'MONTHS_SINCE_LAST_GIFT', 'FILE_CARD_GIFT', 'MONTHS_SINCE_LAST_PROM_RESP',
        'NUMBER_PROM_12', 'LIFETIME_CARD_PROM', 'LIFETIME_GIFT_COUNT',
        'MONTHS_SINCE_FIRST_GIFT', 'LIFETIME_PROM'
    ]
    for col in discrete_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].clip(lower=0)

    # --- Proportions/percentages: clip to valid range ---
    proportions_cols = [
        'RECENT_CARD_RESPONSE_PROP', 'PCT_ATTRIBUTE1', 'PCT_ATTRIBUTE2',
        'PCT_ATTRIBUTE3', 'PCT_ATTRIBUTE4', 'PCT_OWNER_OCCUPIED', 'RECENT_RESPONSE_PROP'
    ]
    for col in proportions_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].max() <= 1.0:
            df[col] = df[col].clip(lower=0.0, upper=1.0)
        else:
            df[col] = df[col].clip(lower=0.0, upper=100.0)

    # --- Monetary cols: clip to positive ---
    monetary_cols = [
        'LIFETIME_MIN_GIFT_AMT', 'LAST_GIFT_AMT', 'LIFETIME_MAX_GIFT_AMT',
        'RECENT_AVG_CARD_GIFT_AMT', 'RECENT_AVG_GIFT_AMT', 'LIFETIME_GIFT_AMOUNT',
        'MEDIAN_HOUSEHOLD_INCOME', 'MEDIAN_HOME_VALUE', 'PER_CAPITA_INCOME'
    ]
    for col in monetary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].clip(lower=0.01)

    return df
