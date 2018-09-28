
def binary_classified_precision(y_real, y_pred):
    _total_amount = len(y_real)
    _matching_amount = 0

    for _r, _p in zip(y_real, y_pred):
        if _r == _p: _matching_amount += 1

    return _matching_amount / _total_amount
