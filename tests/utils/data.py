
from contextlib import contextmanager
from copy import deepcopy

try:
    from pandas import DataFrame as pandas_DataFrame
    from pandas.testing import assert_frame_equal as pandas_assert_frame_equal
except:
    pandas_DataFrame = None
    pandas_assert_frame_equal = None
try:
    from polars import DataFrame as polars_DataFrame
    from polars.testing import assert_frame_equal as polars_assert_frame_equal
except:
    polars_DataFrame = None
    polars_assert_frame_equal = None
try:
    from dask.dataframe import DataFrame as dask_DataFrame
    from dask.dataframe.utils import assert_eq
    dask_assert_frame_equal = lambda a, b: assert_eq(a, b, check_divisions=False, check_index=False, sort_results=False)
except:
    dask_DataFrame = None
    dask_assert_frame_equal = None
try:
    from modin.pandas import DataFrame as modin_DataFrame
    from modin.pandas.test.utils import assert_frame_equal as modin_assert_frame_equal
except:
    modin_DataFrame = None
    modin_assert_frame_equal = None
import shutil

from etlrules.data import RuleData, context
from etlrules.runner import get_etlrules_temp_dir


def assert_frame_equal(df, df2, ignore_row_ordering=False, ignore_column_ordering=False):
    if pandas_DataFrame is not None and isinstance(df, pandas_DataFrame) and isinstance(df2, pandas_DataFrame):
        assert pandas_assert_frame_equal is not None
        if ignore_row_ordering:
            df = df.sort_values(list(df.columns))
            df2 = df2.sort_values(list(df2.columns))
        if ignore_column_ordering:
            df = df[sorted(df.columns)]
            df2 = df2[sorted(df2.columns)]
        pandas_assert_frame_equal(df, df2)
    elif dask_DataFrame is not None and isinstance(df, dask_DataFrame) and isinstance(df2, dask_DataFrame):
        assert dask_assert_frame_equal is not None
        if ignore_row_ordering:
            df = df.sort_values(list(df.columns))
            df2 = df2.sort_values(list(df2.columns))
        if ignore_column_ordering:
            df = df[sorted(df.columns)]
            df2 = df2[sorted(df2.columns)]
        df = df.compute()
        df2 = df2.compute()
        dask_assert_frame_equal(df, df2)
    elif polars_DataFrame is not None and isinstance(df, polars_DataFrame) and isinstance(df2, polars_DataFrame):
        assert polars_assert_frame_equal is not None
        if ignore_row_ordering:
            df = df.sort(df.columns)
            df2 = df2.sort(df2.columns)
        if ignore_column_ordering:
            df = df[sorted(df.columns)]
            df2 = df2[sorted(df2.columns)]
        polars_assert_frame_equal(df, df2)
    elif modin_DataFrame is not None and isinstance(df, modin_DataFrame) and isinstance(df2, modin_DataFrame):
        assert modin_assert_frame_equal is not None
        if ignore_row_ordering:
            df = df.sort_values(list(df.columns))
            df2 = df2.sort_values(list(df2.columns))
        if ignore_column_ordering:
            df = df[sorted(df.columns)]
            df2 = df2[sorted(df2.columns)]
        modin_assert_frame_equal(df, df2)
    else:
        assert False


@contextmanager
def get_test_data(main_input=None, named_inputs=None, named_output=None, strict=True):
    etlrules_tempdir, etlrules_tempdir_cleanup = get_etlrules_temp_dir()
    ctx = {
        "etlrules_tempdir": etlrules_tempdir,
        "etlrules_tempdir_cleanup": etlrules_tempdir_cleanup,
    }
    with context.set(ctx):
        data = TestRule(main_input=main_input, named_inputs=named_inputs, named_output=named_output, context=ctx, strict=strict)
        try:
            yield data
            data.validate()
        finally:
            if etlrules_tempdir_cleanup:
                shutil.rmtree(etlrules_tempdir)


class TestRule(RuleData):

    def __init__(self, main_input=None, named_inputs=None, named_output=None, context=None, strict=True):
        self.main_input_copy = deepcopy(main_input) if main_input is not None else None
        self.named_inputs_copies = {
            name: deepcopy(df) for name, df in (named_inputs or {}).items()
        }
        super().__init__(main_input=main_input, named_inputs=named_inputs, context=context, strict=strict)
        self.named_output = named_output

    def validate(self):
        if self.named_output is not None and self.main_input_copy is not None:
            assert_frame_equal(self.main_input_copy, self.get_main_output())
        for name, df in self.named_inputs_copies.items():
            if name != self.named_output:
                assert_frame_equal(df, self.get_named_output(name))
