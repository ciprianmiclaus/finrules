import pytest
import dask.dataframe as dd
import pandas as pd
import polars as pl
import modin.pandas as md

from etlrules.backends import dask as dd_rules
from etlrules.backends import pandas as pd_rules
from etlrules.backends import polars as pl_rules
from etlrules.backends import modin as md_rules


TYPE_MAPPING = {
    "pl": {
        "string": pl.Utf8,
        "int8": pl.Int8,
        "Int8": pl.Int8,
        "int16": pl.Int16,
        "Int16": pl.Int16,
        "int32": pl.Int32,
        "Int32": pl.Int32,
        "int64": pl.Int64,
        "Int64": pl.Int64,
        "uint8": pl.UInt8,
        "UInt8": pl.UInt8,
        "uint16": pl.UInt16,
        "UInt16": pl.UInt16,
        "uint32": pl.UInt32,
        "UInt32": pl.UInt32,
        "uint64": pl.UInt64,
        "UInt64": pl.UInt64,
        "boolean": pl.Boolean,
        "object": pl.Object,
        "float64": pl.Float64,
        "Float64": pl.Float64,
        "list_strings": pl.List(pl.Utf8),
        "list_int64s": pl.List(pl.Int64),
        "datetime": pl.Datetime,
        "timedelta": pl.Duration,
    },
    "pd": {
        "boolean": "boolean",
        "datetime": "datetime64[ns]",
        "list_strings": "object",
        "list_int64s": "object",
        "timedelta": "timedelta64[ns]",
    },
    "dd": {
        "boolean": "boolean",
        "datetime": "datetime64[ns]",
        "list_strings": "object",
        "list_int64s": "object",
        "timedelta": "timedelta64[ns]",
    },
}

class BackendFixture:
    def __init__(self, name, impl_pckg, rules_pckgs):
        self.name = name
        self.impl_pckg = impl_pckg
        self.rules_pckgs = rules_pckgs

    @property
    def impl(self):
        return self.impl_pckg

    @property
    def rules(self):
        return self.rules_pckgs

    def __str__(self):
        return self.name

    def DataFrame(self, data, dtype=None, astype=None):

        def get_data_keys(ddata):
            keys = []
            if isinstance(ddata, list):
                for elem in ddata:
                    assert isinstance(elem, dict)
                    for k in elem.keys():
                        if k not in keys:
                            keys.append(k)
            elif isinstance(ddata, dict):
                keys = list(ddata.keys())
            return keys

        if self.impl_pckg in (pd, md):
            df = self.impl_pckg.DataFrame(data, dtype=TYPE_MAPPING["pd"].get(dtype, dtype))
            if astype is not None:
                df = df.astype({k: TYPE_MAPPING["pd"].get(v, v) for k, v in astype.items()})
        elif self.impl_pckg == dd:
            df = pd.DataFrame(data, dtype=TYPE_MAPPING["pd"].get(dtype, dtype))
            if astype is not None:
                df = df.astype({k: TYPE_MAPPING["pd"].get(v, v) for k, v in astype.items()})
            df = dd.from_pandas(df, npartitions=1)
        elif self.impl_pckg == pl:
            schema = None
            if dtype is not None:
                keys = get_data_keys(data)
                if keys:
                    schema = {k: TYPE_MAPPING["pl"][dtype] for k in keys}
            elif astype is not None:
                assert isinstance(astype, dict)
                keys = get_data_keys(data)
                schema = {
                    col: TYPE_MAPPING["pl"][astype[col]] if col in astype else None for col in keys
                }
            df = pl.DataFrame(data, schema)
        else:
            assert False, f"unknown impl_pckg: {self.impl_pckg}"
        return df

    def astype(self, df, astype):
        if self.impl_pckg in (pd, dd, md):
            astype = {
                col: TYPE_MAPPING["pd"][dtype] for col, dtype in astype.items()
            }
            return df.astype(astype)
        elif self.impl_pckg == pl:
            return df.with_columns(
                *[pl.col(col).cast(TYPE_MAPPING["pl"][dtype]) for col, dtype in astype.items()]
            )
        assert False, f"unknown impl_pckg: {self.impl_pckg}"

    def assign(self, df, column_name, column):
        if self.impl_pckg == pl:
            return df.with_columns(**{column_name: column})
        return df.assign(**{column_name: column})

    def rename(self, df, rename_dict):
        if self.impl_pckg in (pd, dd, md):
            return df.rename(columns=rename_dict)
        elif self.impl_pckg == pl:
            columns = list(df.columns)
            if all(isinstance(key, int) and 0 <= key < len(columns) for key in rename_dict.keys()):
                rename_dict = {
                    columns[key]: val for key, val in rename_dict.items()
                }
            return df.rename(rename_dict)
        assert False, f"unknown impl_pckg: {self.impl_pckg}"

    def hconcat(self, df1, df2):
        if self.impl_pckg in (pd, dd, md):
            return self.impl_pckg.concat([df1, df2], axis=1)
        elif self.impl_pckg == pl:
            return df1.hstack(df2, in_place=False)
        assert False, f"unknown impl_pckg: {self.impl_pckg}"

    def empty_df(self, df):
        if self.impl_pckg in (pd, pl, md):
            return df[:0]
        elif self.impl_pckg == dd:
            return dd.from_pandas(df.head(0), npartitions=1)
        assert False, f"unknown impl_pckg: {self.impl_pckg}"

    def business_day_offset(self, dt_col, offset, strict=True):
        if self.impl_pckg in (pd, md):
            from etlrules.backends.pandas.datetime import business_day_offset
            return business_day_offset(dt_col, offset, strict=strict)
        elif self.impl_pckg == dd:
            from etlrules.backends.dask.datetime import business_day_offset
            return business_day_offset(dt_col, offset, strict=strict)
        elif self.impl_pckg == pl:
            from etlrules.backends.polars.datetime import business_day_offset
            return business_day_offset(dt_col, offset, strict=strict)
        assert False, "Not implemented for backend."

    def months_offset(self, dt_col, offset, strict=True):
        if self.impl_pckg in (pd, md):
            from etlrules.backends.pandas.datetime import months_offset
            return months_offset(dt_col, offset, strict=strict)
        elif self.impl_pckg == dd:
            from etlrules.backends.dask.datetime import months_offset
            return months_offset(dt_col, offset, strict=strict)
        elif self.impl_pckg == pl:
            from etlrules.backends.polars.datetime import months_offset
            return months_offset(dt_col, offset, strict=strict)
        assert False, "Not implemented for backend."

@pytest.fixture(params=[
    ('pandas', pd, pd_rules),
    ('polars', pl, pl_rules),
    ('dask', dd, dd_rules),
    ('modin', md, md_rules),
])
def backend(request):
    return BackendFixture(*request.param)
