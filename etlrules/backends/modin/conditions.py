import numpy as np

from etlrules.backends.common.conditions import (
    IfThenElseRule as IfThenElseRuleBase,
    FilterRule as FilterRuleBase
)
from .expressions import Expression


class IfThenElseRule(IfThenElseRuleBase):

    def get_condition_expression(self):
        return Expression(self.condition_expression, filename=f'{self.output_column}.py')

    def apply(self, data):
        df = self._get_input_df(data)
        df_columns = set(df.columns)
        self._validate_columns(df_columns)
        cond_series = self._condition_expression.eval(df)
        then_value = self.then_value if self.then_value is not None else df[self.then_column]
        else_value = self.else_value if self.else_value is not None else df[self.else_column]
        result = np.where(cond_series, then_value, else_value)
        df = df.assign(**{self.output_column: result})
        if df.empty and (isinstance(then_value, str) or isinstance(else_value, str)):
            df = df.astype({self.output_column: "string"})
        self._set_output_df(data, df)


class FilterRule(FilterRuleBase):

    def get_condition_expression(self):
        return Expression(self.condition_expression, filename="FilterRule.py")

    def apply(self, data):
        df = self._get_input_df(data)
        cond_series = self._condition_expression.eval(df)
        if self.discard_matching_rows:
            cond_series = ~cond_series
        self._set_output_df(data, df[cond_series].reset_index(drop=True))
        if self.named_output_discarded:
            data.set_named_output(self.named_output_discarded, df[~cond_series].reset_index(drop=True))
