import polars as pl

from etlrules.backends.common.aggregate import AggregateRule as AggregateRuleBase
from etlrules.backends.polars.types import MAP_TYPES


def listify(values):
    return [value for value in values if value is not None]

def stringify(values):
    return ",".join(str(elem) for elem in values if elem is not None)


class AggregateRule(AggregateRuleBase):

    AGGREGATIONS = {
        "min": "min",
        "max": "max",
        "mean": "mean",
        "count": [("fill_null", {"strategy": "zero"}), "count"],
        "countNoNA": ["is_not_null", "sum"],
        "sum": "sum",
        "first": "first",
        "last": "last",
        "list": {
            "func": listify,
            "type": None,
        },
        "csv": {
            "func": stringify,
            "type": pl.Utf8,
        },
    }

    def _get_agg(self, col, aggs, kwargs):
        if isinstance(aggs, str):
            return getattr(pl.col(col), aggs)(**kwargs)
        elif isinstance(aggs, list):
            expr = None
            for agg in aggs:
                kwargs = {}
                if isinstance(agg, tuple):
                    agg, kwargs = agg
                if expr is None:
                    expr = self._get_agg(col, agg, kwargs)
                else:
                    expr = getattr(expr, agg)(**kwargs)
            return expr
        elif isinstance(aggs, dict):
            return pl.col(col).map_elements(aggs["func"], return_dtype=aggs["type"])
        # assumes lambda
        return pl.col(col).map_elements(aggs)

    def do_aggregate(self, df, aggs):
        aggs = [self._get_agg(col, aggs, {}) for col, aggs in aggs.items()]
        result = df.group_by(self.group_by, maintain_order=True).agg(*aggs)
        if self.aggregation_types:
            result = result.select(
                pl.col(col).cast(MAP_TYPES[self.aggregation_types[col]]) if col in self.aggregation_types else pl.col(col)
                for col in result.columns
            )
        return result
