from pandas import DataFrame
from pandas.testing import assert_frame_equal
import pytest

from etlrules.backends.pandas import TypeConversionRule
from etlrules.exceptions import MissingColumnError, UnsupportedTypeError
from tests.backends.pandas.utils.data import get_test_data


def test_type_conversion_rule_simple():
    df = DataFrame(data=[
        {'A': '1', 'B': 2},
        {'A': '2', 'B': 3},
        {'A': '3', 'B': 4},
    ])
    with get_test_data(df, named_inputs={"other": df}, named_output="result") as data:
        rule = TypeConversionRule({
            'A': 'int64',
            'B': 'str',
        }, named_output="result")
        rule.apply(data)
        expected = DataFrame(data=[
            {'A': 1, 'B': '2'},
            {'A': 2, 'B': '3'},
            {'A': 3, 'B': '4'},
        ])
        assert_frame_equal(data.get_named_output("result"), expected)


def test_type_conversion_rule_missing_column():
    df = DataFrame(data=[
        {'A': '1', 'B': 2},
        {'A': '2', 'B': 3},
        {'A': '3', 'B': 4},
    ])
    with get_test_data(df, named_inputs={"other": df}, named_output="result") as data:
        rule = TypeConversionRule({
            'A': 'int64',
            'C': 'str',
        }, named_output="result")
        with pytest.raises(MissingColumnError):
            rule.apply(data)


def test_type_conversion_rule_unsupported_type():
    df = DataFrame(data=[
        {'A': '1', 'B': 2},
        {'A': '2', 'B': 3},
        {'A': '3', 'B': 4},
    ])
    with get_test_data(df, named_inputs={"other": df}, named_output="result") as data:
        rule = TypeConversionRule({
            'A': 'int64',
            'B': 'unknown_type',
        }, named_output="result")
        with pytest.raises(UnsupportedTypeError):
            rule.apply(data)
