from .technical import generate_all_technical_features
from .microstructure import generate_all_microstructure_features
from .statarb import generate_all_statarb_features

def build_full_feature_set(df):
    """Apply all feature groups to the input DataFrame."""
    df = generate_all_technical_features(df)
    df = generate_all_microstructure_features(df)
    df = generate_all_statarb_features(df)
    return df

__all__ = [
    'generate_all_technical_features',
    'generate_all_microstructure_features',
    'generate_all_statarb_features',
    'build_full_feature_set',
]
