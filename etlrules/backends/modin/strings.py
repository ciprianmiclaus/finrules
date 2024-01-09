import logging
from modin.pandas import NA
from numpy import nan

from etlrules.backends.common.strings import (
    StrLowerRule as StrLowerRuleBase,
    StrUpperRule as StrUpperRuleBase,
    StrCapitalizeRule as StrCapitalizeRuleBase,
    StrSplitRule as StrSplitRuleBase,
    StrSplitRejoinRule as StrSplitRejoinRuleBase,
    StrStripRule as StrStripRuleBase,
    StrPadRule as StrPadRuleBase,
    StrExtractRule as StrExtractRuleBase,
)

from .base import PandasMixin

perf_logger = logging.getLogger("etlrules.perf")


class StrLowerRule(StrLowerRuleBase, PandasMixin):
    def do_apply(self, df, col):
        return col.str.lower()


class StrUpperRule(StrUpperRuleBase, PandasMixin):
    def do_apply(self, df, col):
        return col.str.upper()


class StrCapitalizeRule(StrCapitalizeRuleBase, PandasMixin):
    def do_apply(self, df, col):
        return col.str.capitalize()


class StrSplitRule(StrSplitRuleBase, PandasMixin):
    def do_apply(self, df, col):
        return col.str.split(pat=self.separator, n=self.limit, regex=False)


class StrSplitRejoinRule(StrSplitRejoinRuleBase, PandasMixin):
    def do_apply(self, df, col):
        perf_logger.warning("StrSplitRejoinRule is not vectorized and might hurt the overall performance.")
        new_col = col.str.split(pat=self.separator, n=self.limit, regex=False)
        new_separator = self.new_separator
        if self.sort is not None:
            reverse = self.sort==self.SORT_DESCENDING
            func = lambda val: new_separator.join(sorted(val, reverse=reverse)) if val not in (nan, NA, None) else val
        else:
            func = lambda val: new_separator.join(val) if val not in (nan, NA, None) else val
        return new_col.apply(func).astype("string")


class StrStripRule(StrStripRuleBase, PandasMixin):
    def do_apply(self, df, col):
        if self.how == self.STRIP_BOTH:
            return col.str.strip(to_strip=self.characters)
        elif self.how == self.STRIP_RIGHT:
            return col.str.rstrip(to_strip=self.characters)
        return col.str.lstrip(to_strip=self.characters)


class StrPadRule(StrPadRuleBase, PandasMixin):
    def do_apply(self, df, col):
        if self.how == self.PAD_LEFT:
            return col.str.rjust(self.width, fillchar=self.fill_character)
        return col.str.ljust(self.width, fillchar=self.fill_character)


class StrExtractRule(StrExtractRuleBase, PandasMixin):
    def apply(self, data):
        df = self._get_input_df(data)
        columns, output_columns = self.validate_columns_in_out(df.columns, [self.input_column], self.output_columns, self.strict, validate_length=False)
        new_cols_dict = {}
        groups = self._compiled_expr.groups
        for idx, col in enumerate(columns):
            new_col = df[col].str.extract(self._compiled_expr, expand=True)
            if self.keep_original_value:
                # only the first new column keeps the value (in case of multiple groups)
                new_col[0].fillna(value=df[col], inplace=True)
            for group in range(groups):
                new_cols_dict[output_columns[idx * groups + group]] = new_col[group]
        df = df.assign(**new_cols_dict)
        self._set_output_df(data, df)
