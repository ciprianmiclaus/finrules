from etlrules.backends.common.joins import (
    LeftJoinRule as LeftJoinRuleBase,
    RightJoinRule as RightJoinRuleBase,
    InnerJoinRule as InnerJoinRuleBase,
    OuterJoinRule as OuterJoinRuleBase,
)


class JoinsMixin():
    def do_apply(self, left_df, right_df):
        left_on, right_on = self._get_key_columns()
        return left_df.merge(
            right_df,
            how=self.JOIN_TYPE,
            left_on=left_on,
            right_on=right_on,
            suffixes=self.suffixes
        )


class LeftJoinRule(JoinsMixin, LeftJoinRuleBase):
    JOIN_TYPE = "left"


class InnerJoinRule(JoinsMixin, InnerJoinRuleBase):
    JOIN_TYPE = "inner"


class OuterJoinRule(JoinsMixin, OuterJoinRuleBase):
    JOIN_TYPE = "outer"


class RightJoinRule(JoinsMixin, RightJoinRuleBase):
    JOIN_TYPE = "right"
