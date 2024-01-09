import logging
from pandas import Series

from etlrules.backends.common.expressions import Expression as ExpressionBase
from etlrules.data import context


perf_logger = logging.getLogger("etlrules.perf")


class Expression(ExpressionBase):

    def eval(self, df):
        try:
            expr_series = eval(self._compiled_expr, {}, {'df': df, 'context': context})
        except (TypeError, ValueError):
            # attempt to run a slower apply
            expr = self._compiled_expr
            if df.empty:
                expr_series = Series([], dtype="string")
            else:
                perf_logger.warning("Evaluating expression '%s' is not vectorized and might hurt the overall performance.", self.expression_str)
                expr_series = df.apply(lambda df: eval(expr, {}, {'df': df, 'context': context}), axis=1)
        return expr_series
