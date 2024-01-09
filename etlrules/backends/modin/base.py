import modin.pandas as pd
from typing import Mapping


class PandasMixin:
    def assign_do_apply(self, df: pd.DataFrame, input_column: str, output_column: str):
        return df.assign(**{output_column: self.do_apply(df, df[input_column])})

    def assign_do_apply_dict(self, df: pd.DataFrame, mapper_dict: Mapping[str, pd.Series]):
        return df.assign(**mapper_dict)
