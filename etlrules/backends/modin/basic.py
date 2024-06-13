import re

from etlrules.backends.common.basic import (
    DedupeRule as DedupeRuleBase,
    ExplodeValuesRule as ExplodeValuesRuleBase,
    RenameRule as RenameRuleBase,
    ReplaceRule as ReplaceRuleBase,
    SortRule as SortRuleBase,
)
from etlrules.backends.pandas.base import PandasMixin
from etlrules.backends.pandas.types import MAP_TYPES


class DedupeRule(DedupeRuleBase):
    def do_dedupe(self, df):
        keep = False if self.keep == DedupeRule.KEEP_NONE else self.keep
        return df.drop_duplicates(subset=self.columns, keep=keep, ignore_index=True)


class RenameRule(RenameRuleBase):
    def do_rename(self, df, mapper):
        return df.rename(columns=mapper)


class SortRule(SortRuleBase):
    def do_sort(self, df):
        return df.sort_values(by=self.sort_by, ascending=self.ascending, ignore_index=True)


class ReplaceRule(ReplaceRuleBase, PandasMixin):

    def _get_old_new_regex(self, old_val, new_val):
        compiled = re.compile(old_val)
        groupindex = compiled.groupindex
        if compiled.groups > 0 and not groupindex:
            groupindex = {v: v for v in range(1, compiled.groups + 1)}
        for group_name, group_idx in groupindex.items():
            new_val = new_val.replace(f"${group_name}", f"\\g<{group_name}>")
            new_val = new_val.replace(f"${{{group_name}}}", f"\\g<{group_name}>")
            if group_name != group_idx:
                new_val = new_val.replace(f"${{{group_idx}}}", f"\\{group_idx}")
                new_val = new_val.replace(f"${group_idx}", f"\\{group_idx}")
        return old_val, new_val

    def do_apply(self, df, col):
        if self.regex:
            for old_val, new_val in zip(self.values, self.new_values):
                old_val, new_val = self._get_old_new_regex(old_val, new_val)
                return col.replace(to_replace=old_val, value=new_val, regex=self.regex)
        return col.replace(to_replace=self.values, value=self.new_values, regex=self.regex)


class ExplodeValuesRule(ExplodeValuesRuleBase):

    def apply(self, data):
        df = self._get_input_df(data)
        self._validate_input_column(df)
        result = df.explode(self.input_column, ignore_index=True)
        if self.column_type:
            result = result.astype({self.input_column: MAP_TYPES[self.column_type]})
        self._set_output_df(data, result)
