import pandas as pd

def group_metrics(data: pd.DataFrame, group_size: int) -> pd.DataFrame:
    """
    Group group_size rows at a time together and sum them up.

    :param group_size: Number of rows to group together
    :return: A new grouped up data frame
    """

    return data.groupby(data.index // group_size).sum()
