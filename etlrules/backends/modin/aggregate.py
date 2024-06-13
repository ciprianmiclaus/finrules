from modin.pandas import isnull

from etlrules.backends.common.aggregate import AggregateRule as AggregateRuleBase
from etlrules.backends.pandas.types import MAP_TYPES


class AggregateRule(AggregateRuleBase):

    AGGREGATIONS = {
        "min": "min",
        "max": "max",
        "mean": "mean",
        "count": "size",
        "countNoNA": "count",
        "sum": "sum",
        "first": "first",
        "last": "last",
        "list": lambda values: [value for value in values if not isnull(value)],
        "csv": lambda values: ",".join(str(elem) for elem in values if not isnull(elem)),
    }

    def do_aggregate(self, df, aggs):
        result = df.groupby(by=self.group_by, as_index=False, dropna=False).agg(aggs)
        if self.aggregation_types:
            result = result.astype({col: MAP_TYPES[col_type] for col, col_type in self.aggregation_types.items()})
        return result
