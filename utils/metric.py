"""
This script exists to reduce code duplication across metrics.
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(
    error_message: str, solution: Union[pd.DataFrame, np.ndarray]
) -> bool:
    """Many metrics can raise more errors than can be handled manually. This function attempts
    to identify errors that can be treated as ParticipantVisibleError without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and can be shown to participants.

    We expect this filter to reject many safe messages. It's intended only to reduce the number of errors we need to manage manually.
    """
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all(
            [pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values]
        )
        solution_has_bools = any(
            [pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values]
        )
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if "true" in error_message.lower() or "false" in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    """
    Call score. If that raises an error and that already been specifically handled, just raise it.
    Otherwise make a conservative attempt to identify potential participant visible errors.
    """
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == "ParticipantVisibleError":
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == "HostVisibleError":
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    return score_result


def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """Verify that the dataframe contains valid probabilities.

    The dataframe must be limited to the target columns; do not pass in any ID columns.
    """
    if not pandas.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f"All target values in {df_name} must be numeric")

    if df.min().min() < 0:
        raise ParticipantVisibleError(
            f"All target values in {df_name} must be at least zero"
        )

    if df.max().max() > 1:
        raise ParticipantVisibleError(
            f"All target values in {df_name} must be no greater than one"
        )

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(
            f"Target values in {df_name} do not add to one within all rows"
        )


# %% [code]
# %% [code]


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    weights_column_name: Optional[str] = None,
    multioutput="uniform_average",
) -> float:
    """
    Wrapper for https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    :math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    In the particular case when ``y_true`` is constant, the :math:`R^2` score
    is not finite: it is either ``NaN`` (perfect predictions) or ``-Inf``
    (imperfect predictions). To prevent such non-finite numbers to pollute
    higher-level experiments such as a grid search cross-validation, by default
    these cases are replaced with 1.0 (perfect predictions) or 0.0 (imperfect
    predictions) respectively. You can set ``force_finite`` to ``False`` to
    prevent this fix from happening.

    Note: when the prediction residuals have zero mean, the :math:`R^2` score
    is identical to the
    `Explained Variance score <explained_variance_score>`.

    Parameters
    ----------
    solution : DataFrame of shape (n_samples,) or (n_samples, n_outputs)
    Ground truth (correct) target values.

    submission : DataFrame of shape (n_samples,) or (n_samples, n_outputs)
    Estimated target values.

    weights_column_name: optional str, the name of the sample weights column in the solution file.

    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`

    Examples
    --------

    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))

    >>> y_true = [1, 2, 3]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [1, 2, 3]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    1.0
    >>> y_true = [1, 2, 3]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [2, 2, 2]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.0
    >>> y_true = [1, 2, 3]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [3, 2, 1]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    -3.0
    >>> y_true = [-2, -2, -2]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [-2, -2, -2]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    1.0

    >>> y_true = [-2, -2, -2]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [-2, -2, -2 + 1e-8]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.0
    """
    # Skip sorting and equality checks for the row_id_column since that should already be handled
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weight = None
    if weights_column_name:
        if weights_column_name not in solution.columns:
            raise ValueError(
                f"The solution weights column {weights_column_name} is not found"
            )
        sample_weight = solution.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError("The solution weights are not numeric")

    if not (
        (len(submission.columns) == 1)
        or (len(submission.columns) == len(solution.columns))
    ):
        raise ParticipantVisibleError(
            f"Invalid number of submission columns. Found {len(submission.columns)}"
        )

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {
            x: submission[x].dtype
            for x in submission.columns
            if not pandas.api.types.is_numeric_dtype(submission[x])
        }
        raise ParticipantVisibleError(
            f"Invalid submission data types found: {bad_dtypes}"
        )

    solution = solution.values
    submission = submission.values

    score_result = safe_call_score(
        sklearn.metrics.r2_score,
        solution,
        submission,
        sample_weight=sample_weight,
        force_finite=True,
        multioutput=multioutput,
    )

    return score_result
