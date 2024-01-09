from etlrules.backends.common.numeric import RoundRule as RoundRuleBase, AbsRule as AbsRuleBase

from .base import PandasMixin


class RoundRule(RoundRuleBase, PandasMixin):
    def do_apply(self, df, col):
        return col.round(self.scale)


class AbsRule(AbsRuleBase, PandasMixin):
    def do_apply(self, df, col):
        return col.abs()
