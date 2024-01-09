from etlrules.backends.common.fill import (
    BackFillRule as BackFillRuleBase,
    ForwardFillRule as ForwardFillRuleBase,
)


class FillMixin:
    def do_apply(self, df):
        df_columns = [col for col in df.columns]
        if self.sort_by:
            df = df.sort_values(by=self.sort_by, ascending=self.sort_ascending, ignore_index=True)
        if self.group_by:
            res = df.groupby(self.group_by)
            res = getattr(res, self.FILL_METHOD)()
            res = res[self.columns]
        else:
            res = df[self.columns]
            res = getattr(res, self.FILL_METHOD)()
        df = df.assign(**{col: res[col] for col in self.columns})
        df = df[df_columns]
        return df


class ForwardFillRule(FillMixin, ForwardFillRuleBase):
    FILL_METHOD = "ffill"


class BackFillRule(FillMixin, BackFillRuleBase):
    FILL_METHOD = "bfill"
