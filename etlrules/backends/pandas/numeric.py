from etlrules.backends.common.numeric import RoundRule as RoundRuleBase, AbsRule as AbsRuleBase


class RoundRule(RoundRuleBase):
    def do_df_apply(self, df):
        input_column, output_column = self.validate_in_out_columns(df.columns, self.input_column, self.output_column, self.strict)
        res_df = df.round({input_column: self.scale})
        return df.assign(**{output_column: res_df[input_column]})


class AbsRule(AbsRuleBase):
    def do_df_apply(self, df):
        input_column, output_column = self.validate_in_out_columns(df.columns, self.input_column, self.output_column, self.strict)
        return df.assign(**{output_column: df[input_column].abs()})
