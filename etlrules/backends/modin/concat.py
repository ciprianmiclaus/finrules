from modin.pandas import concat

from etlrules.backends.common.concat import VConcatRule as VConcatRuleBase, HConcatRule as HConcatRuleBase


class VConcatRule(VConcatRuleBase):

    def do_concat(self, left_df, right_df):
        return concat([left_df, right_df], axis=0, ignore_index=True)


class HConcatRule(HConcatRuleBase):

    def do_concat(self, left_df, right_df):
        return concat([left_df, right_df], axis=1)
