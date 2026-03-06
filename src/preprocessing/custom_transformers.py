"""
Custom scikit-learn transformers for the MercadoLibre fraud detection project.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


class RareGroupTransformer(BaseEstimator, TransformerMixin):
    """
    Groups infrequent categories into a single 'rare' category.
    output_suffix: appended to each column name in get_feature_names_out.
                   Set to '' to keep original names.
    """
    def __init__(self, min_freq=0.01, other_label='rare', output_suffix='_rare_grouped'):
        self.min_freq = min_freq
        self.other_label = other_label
        self.output_suffix = output_suffix

    def fit(self, X, y=None):
        """
        Learns which categories are frequent from the training data for each column.
        
        Args:
            X (pd.DataFrame): A dataframe with one or more categorical columns.
        """
        # Calculate frequency of each category for each column and store in a dictionary
        self.frequent_categories_ = {}
        for col in X.columns:
            freqs = X[col].value_counts(normalize=True)
            self.frequent_categories_[col] = freqs[freqs >= self.min_freq].index.tolist()
        return self

    def transform(self, X):
        """
        Applies the grouping to new data, replacing rare categories.
        
        Args:
            X (pd.DataFrame): A dataframe with one or more categorical columns.
        """
        X_copy = X.copy()
        for col in X.columns:
            # Get the list of frequent categories for the current column
            frequent_cats = self.frequent_categories_.get(col, [])
            # Cast to object first so a string label can be assigned to any dtype
            X_copy[col] = X_copy[col].astype(object)
            is_rare = ~X_copy[col].isin(frequent_cats)
            X_copy.loc[is_rare, col] = self.other_label
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{col}{self.output_suffix}' for col in input_features])

class FrequencyEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Performs frequency encoding, learning the mapping from the training set.
    output_suffix: appended to each column name in get_feature_names_out.
    """
    def __init__(self, output_suffix='_freq_encoded'):
        self.output_suffix = output_suffix
        self.encoding_map_ = {}

    def fit(self, X, y=None):
        """
        Learns the frequency of each category from the training data.
        
        Args:
            X (pd.DataFrame): A dataframe with a single column.
        """
        series = X.squeeze()
        self.encoding_map_ = series.value_counts(normalize=True).to_dict()
        # Calculate the mean frequency for unseen categories
        self.unseen_frequency_ = series.value_counts(normalize=True).mean()
        return self

    def transform(self, X):
        """
        Applies the frequency encoding to new data.
        """
        X_copy = X.copy()
        series = X_copy.squeeze()
        
        # Map known categories and fill unseen ones with the mean frequency
        X_copy[series.name] = series.map(self.encoding_map_).fillna(self.unseen_frequency_)
        
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{col}{self.output_suffix}' for col in input_features])

class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Calculates Weight of Evidence (WOE) for a categorical feature.
    
    Learns the WOE mapping from the training data to prevent data leakage.
    """
    def __init__(self, eps=1e-9):
        self.eps = eps
        self.woe_map_ = {}

    def fit(self, X, y):
        """
        Learns the WOE for each category from the training data.
        
        Args:
            X (pd.DataFrame): A dataframe with a single categorical column.
            y (pd.Series): The binary target variable.
        """
        df = pd.DataFrame({
            'x': X.squeeze(),
            'y': y
        })

        total_good = (df['y'] == 0).sum()
        total_bad = (df['y'] == 1).sum()

        if total_good == 0 or total_bad == 0:
            # Handle cases with only one class in the training data
            self.woe_map_ = {cat: 0 for cat in df['x'].unique()}
            self.global_woe_ = 0
            return self

        gb = df.groupby('x')['y'].agg(['count', lambda s: (s==1).sum()]).rename(columns={'<lambda_0>': 'bad'})
        gb['good'] = gb['count'] - gb['bad']

        gb['dist_good'] = gb['good'] / (total_good + self.eps)
        gb['dist_bad'] = gb['bad'] / (total_bad + self.eps)

        gb['woe'] = np.log((gb['dist_good'] + self.eps) / (gb['dist_bad'] + self.eps))
        
        self.woe_map_ = gb['woe'].to_dict()
        
        # Calculate a global WOE for unseen categories
        self.global_woe_ = np.log(((total_good / (total_good + total_bad)) + self.eps) / ((total_bad / (total_good + total_bad)) + self.eps))

        return self

    def transform(self, X):
        """
        Applies the learned WOE mapping to new data.
        """
        X_copy = X.copy()
        series = X_copy.squeeze()
        
        # Map known categories and fill unseen ones with the global WOE
        X_copy[series.name] = series.map(self.woe_map_).fillna(self.global_woe_)
        
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features

class BinningWithNaNTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that bins a numerical feature and treats NaN as a separate,
    dedicated bin.
    """
    def __init__(self, n_bins=10, strategy='quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        # The special integer value we will use to represent the NaN bin
        self.nan_bin_value = -1 

    def fit(self, X, y=None):
        """
        Fits the KBinsDiscretizer on the non-missing values to learn the bin edges.
        """
        # We only fit the binner on the non-missing values
        non_nan_values = X[X.notna()]
        
        # Initialize the binner that will be used in the transform step
        self.binner_ = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
        
        # Fit it only on the real numbers
        if not non_nan_values.empty:
            self.binner_.fit(non_nan_values)
        
        return self

    def transform(self, X):
        """
        Transforms the data, applying the learned bins to numerical values and
        assigning a special bin for NaNs.
        """
        # Create a copy to avoid changing the original dataframe
        X_transformed = pd.DataFrame(index=X.index, columns=X.columns)
        
        # Identify non-missing and missing values
        non_nan_mask = X.notna().squeeze()
        
        # Apply the fitted binner ONLY to the non-missing values
        if non_nan_mask.any():
            X_transformed.loc[non_nan_mask, :] = self.binner_.transform(X[non_nan_mask])
        
        # Assign our special value to all the missing values
        if not non_nan_mask.all():
            X_transformed.loc[~non_nan_mask, :] = self.nan_bin_value
            
        return X_transformed.astype(int)

class BinaryToNumericTransformer(BaseEstimator, TransformerMixin):
    """
    Converts a binary column with 'Y'/'N' values to 1/0.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Use map() to avoid pandas FutureWarning on implicit downcasting via replace()
        return X.apply(lambda col: col.map({'Y': 1, 'N': 0})).astype(float)

    def get_feature_names_out(self, input_features=None):
        return input_features


class BinaryNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Converts binary Y/N columns to 1/0, then imputes NaNs with the most frequent value.
    Combines BinaryToNumericTransformer + SimpleImputer(strategy='most_frequent').
    """
    def fit(self, X, y=None):
        X_numeric = X.apply(lambda col: col.map({'Y': 1, 'N': 0})).astype(float)
        self.imputer_ = SimpleImputer(strategy='most_frequent')
        self.imputer_.fit(X_numeric)
        return self

    def transform(self, X, y=None):
        X_numeric = X.apply(lambda col: col.map({'Y': 1, 'N': 0})).astype(float)
        return self.imputer_.transform(X_numeric)

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)


class CategoricalEncoderWithNaN(BaseEstimator, TransformerMixin):
    """
    OrdinalEncoder that safely handles unknown and NaN categories.
    Unknown categories at transform time are encoded as -1.
    """
    def fit(self, X, y=None):
        from sklearn.preprocessing import OrdinalEncoder
        self.encoder_ = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.encoder_.fit(X)
        return self

    def transform(self, X, y=None):
        return self.encoder_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)


class DatetimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts features from a datetime column matching the feature_engineering_config:
    weekday, hour, is_weekend, hour_sin, hour_cos.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col_name = X.columns[0]
        dt_col = pd.to_datetime(X[col_name])

        df = pd.DataFrame(index=X.index)
        df['weekday']    = dt_col.dt.dayofweek
        df['hour']       = dt_col.dt.hour
        df['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
        df['hour_sin']   = np.sin(2 * np.pi * dt_col.dt.hour / 24)
        df['hour_cos']   = np.cos(2 * np.pi * dt_col.dt.hour / 24)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(['weekday', 'hour', 'is_weekend', 'hour_sin', 'hour_cos'])


# ---------------------------------------------------------------------------
# Flag transformers
# ---------------------------------------------------------------------------

class MissingFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Adds a binary (1/0) flag for each column indicating whether the value is NaN.
    Output column names: {col}_is_missing.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.isna().astype(int)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{col}_is_missing' for col in input_features])


class ZeroFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Adds a binary (1/0) flag for each column indicating whether the value is 0.
    Output column names: {col}_is_zero.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return (X == 0).astype(int)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{col}_is_zero' for col in input_features])


class TopPercentileFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Flags values above a given percentile with 1, others 0.
    Output column names: {col}_top{pct_label}_flag.
    pct_label is inferred from percentile (e.g. 0.99 → 'top1_flag').
    """
    def __init__(self, percentile=0.99):
        self.percentile = percentile

    def fit(self, X, y=None):
        self.thresholds_ = {col: X[col].quantile(self.percentile) for col in X.columns}
        return self

    def transform(self, X, y=None):
        result = pd.DataFrame(index=X.index)
        pct_int = int(round((1 - self.percentile) * 100))
        for col in X.columns:
            result[f'{col}_top{pct_int}_flag'] = (X[col] > self.thresholds_[col]).astype(int)
        return result

    def get_feature_names_out(self, input_features=None):
        pct_int = int(round((1 - self.percentile) * 100))
        return np.array([f'{col}_top{pct_int}_flag' for col in input_features])


# ---------------------------------------------------------------------------
# Numeric transformers with config-matching names
# ---------------------------------------------------------------------------

class BinnedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes with median then applies quantile binning.
    Output column names: bin_{col}.

    n_bins: int applied to all columns, or dict {col_name: n_bins} for per-feature control.
            Columns not present in the dict fall back to default_n_bins (default 10).
    default_n_bins: fallback bin count when n_bins is a dict and a column is not listed.
    """
    def __init__(self, n_bins: int | dict = 10, strategy: str = 'quantile', default_n_bins: int = 10):
        self.n_bins = n_bins
        self.strategy = strategy
        self.default_n_bins = default_n_bins

    def fit(self, X, y=None):
        if isinstance(self.n_bins, dict):
            n_bins_arr = [self.n_bins.get(col, self.default_n_bins) for col in X.columns]
        else:
            n_bins_arr = self.n_bins
        self.imputer_ = SimpleImputer(strategy='median')
        self.binner_  = KBinsDiscretizer(
            n_bins=n_bins_arr,
            encode='ordinal',
            strategy=self.strategy,
            quantile_method='averaged_inverted_cdf',  # silences FutureWarning
        )
        X_imputed = self.imputer_.fit_transform(X)
        self.binner_.fit(X_imputed)
        return self

    def transform(self, X, y=None):
        return self.binner_.transform(self.imputer_.transform(X))

    def get_feature_names_out(self, input_features=None):
        return np.array([f'bin_{col}' for col in input_features])


class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Yeo-Johnson power transformation.
    Output column names: yeojohnson_{col}.
    """
    def fit(self, X, y=None):
        self.pt_ = PowerTransformer(method='yeo-johnson')
        self.pt_.fit(X)
        return self

    def transform(self, X, y=None):
        return self.pt_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'yeojohnson_{col}' for col in input_features])


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation (safe for zeros).
    Output column names: log_{col}.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log1p(np.abs(X))

    def get_feature_names_out(self, input_features=None):
        return np.array([f'log_{col}' for col in input_features])


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Box-Cox transformation per column.
    Box-Cox requires strictly positive values; a shift equal to
    abs(min) + 1e-6 is learned from training data and applied at
    transform time so negative/zero values are handled safely.
    Output column names: boxcox_{col}.
    """
    def fit(self, X, y=None):
        from scipy.stats import boxcox as scipy_boxcox
        self._scipy_boxcox = scipy_boxcox
        self.shifts_ = {}
        self.lambdas_ = {}
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in X.columns:
            series = X[col].astype(float)
            shift = max(0.0, -series.min() + 1e-6)
            self.shifts_[col] = shift
            _, lmbda = scipy_boxcox(series + shift)
            self.lambdas_[col] = lmbda
        return self

    def transform(self, X, y=None):
        from scipy.special import boxcox1p
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        result = pd.DataFrame(index=X.index)
        for col in X.columns:
            shifted = X[col].astype(float) + self.shifts_[col]
            lmbda = self.lambdas_[col]
            if lmbda == 0:
                result[col] = np.log(shifted)
            else:
                result[col] = (shifted ** lmbda - 1) / lmbda
        return result.values

    def get_feature_names_out(self, input_features=None):
        return np.array([f'boxcox_{col}' for col in input_features])


# ---------------------------------------------------------------------------
# Multi-column interaction transformers
# ---------------------------------------------------------------------------

class RatioTransformer(BaseEstimator, TransformerMixin):
    """
    Divides the first column by the second: col1 / col2.
    Expects exactly two columns; NaN where denominator is 0.
    Output column name: {col1}_per_{col2}.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col1, col2 = X.columns[0], X.columns[1]
        ratio = X[col1] / X[col2].replace(0, np.nan)
        return pd.DataFrame(ratio.values, index=X.index, columns=[f'{col1}_per_{col2}'])

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{input_features[0]}_per_{input_features[1]}'])


class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Multiplies two columns: col1 * col2.
    Expects exactly two columns.
    Output column name: interaction_{col1}_{col2}.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col1, col2 = X.columns[0], X.columns[1]
        product = X[col1] * X[col2]
        return pd.DataFrame(product.values, index=X.index, columns=[f'interaction_{col1}_{col2}'])

    def get_feature_names_out(self, input_features=None):
        return np.array([f'interaction_{input_features[0]}_{input_features[1]}'])


class NamedPCATransformer(BaseEstimator, TransformerMixin):
    """
    Imputes with median then applies PCA.
    Output column names: {prefix}_0, {prefix}_1, ...
    """
    def __init__(self, n_components=2, prefix='pca'):
        self.n_components = n_components
        self.prefix = prefix

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy='median')
        self.pca_     = PCA(n_components=self.n_components)
        self.pca_.fit(self.imputer_.fit_transform(X))
        return self

    def transform(self, X, y=None):
        return self.pca_.transform(self.imputer_.transform(X))

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{self.prefix}_{i+1}' for i in range(self.n_components)])
