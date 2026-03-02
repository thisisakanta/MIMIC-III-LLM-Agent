import numpy as np
from scipy.stats import skew

all_functions = [min, max, np.mean, np.std, skew, len]

functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
               (3, 10), (3, 25), (3, 50)]


def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions):
    global sub_periods
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)


def extract_features(data_raw, period, features):
    period = periods_map[period]
    functions = functions_map[features]
    return np.array([extract_features_single_episode(x, period, functions)
                     for x in data_raw])


def get_feature_names(header, period, features):
    """
    Generate feature names based on channels, sub-periods, and functions.
    
    Args:
        header: List of channel names (excluding 'Hours' column)
        period: Period string (e.g., 'all', 'first4days')
        features: Features string (e.g., 'all', 'all_but_len')
    
    Returns:
        List of feature names
    """
    global sub_periods
    period_obj = periods_map[period]
    functions = functions_map[features]
    
    # Function names for display
    function_names = []
    for fn in functions:
        if fn == min:
            function_names.append('min')
        elif fn == max:
            function_names.append('max')
        elif fn == np.mean:
            function_names.append('mean')
        elif fn == np.std:
            function_names.append('std')
        elif fn == skew:
            function_names.append('skew')
        elif fn == len:
            function_names.append('count')
        else:
            function_names.append(fn.__name__)
    
    # Sub-period names
    sub_period_names = []
    for sub_period in sub_periods:
        if sub_period[0] == 2:  # first p%
            sub_period_names.append(f'first{sub_period[1]}%')
        elif sub_period[0] == 3:  # last p%
            sub_period_names.append(f'last{sub_period[1]}%')
        else:
            sub_period_names.append(f'subperiod_{sub_period}')
    
    # Generate all feature names
    feature_names = []
    # Header includes 'Hours' at position 0, skip it
    channels = [h for h in header if h != 'Hours']
    
    for channel in channels:
        for sub_period_name in sub_period_names:
            for fn_name in function_names:
                feature_names.append(f'{channel}_{sub_period_name}_{fn_name}')
    
    return feature_names
