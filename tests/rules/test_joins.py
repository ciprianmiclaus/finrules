import pytest

from etlrules.exceptions import MissingColumnError
from tests.utils.data import assert_frame_equal, get_test_data


LEFT_DF = [
    {"A": 1, "B": "b", "C": 10, "D": "test", "E": 3},
    {"A": 2, "B": "b", "C": 10, "D": "test", "E": 4},
    {"A": 3, "B": "b", "C": 10, "D": "test", "E": 5},
    {"A": 4, "B": "b", "C": 10, "D": "test", "E": 6},
]
EMPTY_LEFT_DF = {
    "A": [], "B": [], "C": [], "D": [], "E": []
}
LEFT_DF_TYPES = {"A": "Int64", "B": "string", "C": "Int64", "D": "string", "E": "Int64"}

RIGHT_DF = [
    {"A": 1, "B": "b", "E": 3, "G": "one"},
    {"A": 2, "B": "b", "E": 4, "G": "two"},
    {"A": 5, "B": "b", "E": 7, "G": "three"},
    {"A": 6, "B": "b", "E": 8, "G": "four"},
]
EMPTY_RIGHT_DF = {
    "A": [], "B": [], "E": [], "G": []
}
RIGHT_DF_TYPES = {"A": "Int64", "B": "string", "E": "Int64", "G": "string"}


@pytest.mark.parametrize("rule_cls_str,key_columns_left,key_columns_right,suffixes,expected", [
    ["LeftJoinRule", ["A", "B"], None, (None, "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E": 4, "E_y": 4, "G": "two"},
        {"A": 3, "B": "b", "C": 10, "D": "test", "E": 5},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E": 6},
    ]],
    ["LeftJoinRule", ["A", "B"], None, ("_x", "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E_y": 4, "G": "two"},
        {"A": 3, "B": "b", "C": 10, "D": "test", "E_x": 5},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E_x": 6},
    ]],
    ["LeftJoinRule", ["A", "B"], None, ("_x", None), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E": 4, "G": "two"},
        {"A": 3, "B": "b", "C": 10, "D": "test", "E_x": 5},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E_x": 6},
    ]],
    ["InnerJoinRule", ["A", "B"], None, (None, "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E": 4, "E_y": 4, "G": "two"},
    ]],
    ["InnerJoinRule", ["A", "B"], None, ("_x", "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E_y": 4, "G": "two"},
    ]],
    ["InnerJoinRule", ["A", "B"], None, ("_x", None), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E": 4, "G": "two"},
    ]],
    ["InnerJoinRule", ["A", "B"], ["A", "B"], (None, "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E": 4, "E_y": 4, "G": "two"},
    ]],
    ["InnerJoinRule", ["A", "B"], ["A", "B"], ("_x", "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E_y": 4, "G": "two"},
    ]],
    ["InnerJoinRule", ["A", "B"], ["E", "B"], (None, "_y"), [
        {"A": 3, "B": "b", "C": 10, "D": "test", "E": 5, "A_y": 1, "E_y": 3, "G": "one"},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E": 6, "A_y": 2, "E_y": 4, "G": "two"},
    ]],
    ["RightJoinRule", ["A", "B"], None, (None, "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E": 4, "E_y": 4, "G": "two"},
        {"A": 5, "B": "b", "E_y": 7, "G": "three"},
        {"A": 6, "B": "b", "E_y": 8, "G": "four"},
    ]],
    ["RightJoinRule", ["A", "B"], None, ("_x", "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E_y": 4, "G": "two"},
        {"A": 5, "B": "b", "E_y": 7, "G": "three"},
        {"A": 6, "B": "b", "E_y": 8, "G": "four"},
    ]],
    ["RightJoinRule", ["A", "B"], None, ("_x", None), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E": 4, "G": "two"},
        {"A": 5, "B": "b", "E": 7, "G": "three"},
        {"A": 6, "B": "b", "E": 8, "G": "four"},
    ]],
    ["OuterJoinRule", ["A", "B"], None, (None, "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E": 4, "E_y": 4, "G": "two"},
        {"A": 3, "B": "b", "C": 10, "D": "test", "E": 5},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E": 6},
        {"A": 5, "B": "b", "E_y": 7, "G": "three"},
        {"A": 6, "B": "b", "E_y": 8, "G": "four"},
    ]],
    ["OuterJoinRule", ["A", "B"], None, ("_x", "_y"), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E_y": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E_y": 4, "G": "two"},
        {"A": 3, "B": "b", "C": 10, "D": "test", "E_x": 5},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E_x": 6},
        {"A": 5, "B": "b", "E_y": 7, "G": "three"},
        {"A": 6, "B": "b", "E_y": 8, "G": "four"},
    ]],
    ["OuterJoinRule", ["A", "B"], None, ("_x", None), [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E_x": 3, "E": 3, "G": "one"},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E_x": 4, "E": 4, "G": "two"},
        {"A": 3, "B": "b", "C": 10, "D": "test", "E_x": 5},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E_x": 6},
        {"A": 5, "B": "b", "E": 7, "G": "three"},
        {"A": 6, "B": "b", "E": 8, "G": "four"},
    ]],
])
def test_join_scenarios(rule_cls_str, key_columns_left, key_columns_right, suffixes, expected, backend):
    left_df = backend.DataFrame(data=LEFT_DF)
    right_df = backend.DataFrame(data=RIGHT_DF)
    expected = backend.DataFrame(data=expected)
    rule_cls = getattr(backend.rules, rule_cls_str)
    with get_test_data(left_df, named_inputs={"right": right_df}, named_output="result") as data:
        rule = rule_cls(named_input_left=None, named_input_right="right", key_columns_left=key_columns_left, 
                        key_columns_right=key_columns_right, suffixes=suffixes, named_output="result")
        rule.apply(data)
        actual = data.get_named_output("result")
        assert_frame_equal(actual, expected, ignore_column_ordering=True, ignore_row_ordering=True)


@pytest.mark.parametrize("rule_cls_str,key_columns_left,key_columns_right,suffixes,left_df,right_df,expected,expected_astype", [
    ["LeftJoinRule", ["A", "B"], None, (None, "_y"), LEFT_DF, EMPTY_RIGHT_DF, [
        {"A": 1, "B": "b", "C": 10, "D": "test", "E": 3, "E_y": None, "G": None},
        {"A": 2, "B": "b", "C": 10, "D": "test", "E": 4},
        {"A": 3, "B": "b", "C": 10, "D": "test", "E": 5},
        {"A": 4, "B": "b", "C": 10, "D": "test", "E": 6},
    ], {"C": "Int64", "D": "string", "E": "Int64", "A": "Int64", "B": "string", "E_y": "Int64", "G": "string"}],
    ["LeftJoinRule", ["A", "B"], None, (None, "_y"), EMPTY_LEFT_DF, EMPTY_RIGHT_DF, {
        "C": [], "D": [], "E": [], "A": [], "B": [], "E_y": [], "G": []
    }, {"C": "Int64", "D": "string", "E": "Int64", "A": "Int64", "B": "string", "E_y": "Int64", "G": "string"}],
    ["InnerJoinRule", ["A", "B"], None, (None, "_y"), EMPTY_LEFT_DF, EMPTY_RIGHT_DF, {
        "C": [], "D": [], "E": [], "A": [], "B": [], "E_y": [], "G": []
    }, {"C": "Int64", "D": "string", "E": "Int64", "A": "Int64", "B": "string", "E_y": "Int64", "G": "string"}],
    ["OuterJoinRule", ["A", "B"], None, (None, "_y"), EMPTY_LEFT_DF, EMPTY_RIGHT_DF, {
        "C": [], "D": [], "E": [], "A": [], "B": [], "E_y": [], "G": []
    }, {"C": "Int64", "D": "string", "E": "Int64", "A": "Int64", "B": "string", "E_y": "Int64", "G": "string"}],
    ["RightJoinRule", ["A", "B"], None, (None, "_y"), EMPTY_LEFT_DF, EMPTY_RIGHT_DF, {
        "C": [], "D": [], "E": [], "A": [], "B": [], "E_y": [], "G": []
    }, {"C": "Int64", "D": "string", "E": "Int64", "A": "Int64", "B": "string", "E_y": "Int64", "G": "string"}],
])
def test_empty_df_join_scenarios(rule_cls_str, key_columns_left, key_columns_right, suffixes, left_df, right_df, expected, expected_astype, backend):
    left_df = backend.DataFrame(data=left_df, astype=LEFT_DF_TYPES)
    right_df = backend.DataFrame(data=right_df, astype=RIGHT_DF_TYPES)
    expected = backend.DataFrame(data=expected, astype=expected_astype)
    rule_cls = getattr(backend.rules, rule_cls_str)
    with get_test_data(left_df, named_inputs={"right": right_df}, named_output="result") as data:
        rule = rule_cls(named_input_left=None, named_input_right="right", key_columns_left=key_columns_left, 
                        key_columns_right=key_columns_right, suffixes=suffixes, named_output="result")
        rule.apply(data)
        assert_frame_equal(data.get_named_output("result"), expected, ignore_column_ordering=True)


LEFT_DF_JOIN_TO_SELF_RESULT = [
    {"A": 1, "B": "b", "C_x": 10, "D_x": "test", "E_x": 3, "C_y": 10, "D_y": "test", "E_y": 3},
    {"A": 2, "B": "b", "C_x": 10, "D_x": "test", "E_x": 4, "C_y": 10, "D_y": "test", "E_y": 4},
    {"A": 3, "B": "b", "C_x": 10, "D_x": "test", "E_x": 5, "C_y": 10, "D_y": "test", "E_y": 5},
    {"A": 4, "B": "b", "C_x": 10, "D_x": "test", "E_x": 6, "C_y": 10, "D_y": "test", "E_y": 6},
]

@pytest.mark.parametrize("rule_cls_str,key_columns_left,key_columns_right,named_input_left,named_input_right,named_output,expected", [
    ["LeftJoinRule", ["A", "B"], None, "input", "input", "result", LEFT_DF_JOIN_TO_SELF_RESULT],
    ["LeftJoinRule", ["A", "B"], None, None, None, None, LEFT_DF_JOIN_TO_SELF_RESULT],
    ["RightJoinRule", ["A", "B"], None, "input", "input", "result", LEFT_DF_JOIN_TO_SELF_RESULT],
    ["RightJoinRule", ["A", "B"], None, None, None, None, LEFT_DF_JOIN_TO_SELF_RESULT],
    ["InnerJoinRule", ["A", "B"], None, "input", "input", "result", LEFT_DF_JOIN_TO_SELF_RESULT],
    ["InnerJoinRule", ["A", "B"], None, None, None, None, LEFT_DF_JOIN_TO_SELF_RESULT],
    ["OuterJoinRule", ["A", "B"], None, "input", "input", "result", LEFT_DF_JOIN_TO_SELF_RESULT],
    ["OuterJoinRule", ["A", "B"], None, None, None, None, LEFT_DF_JOIN_TO_SELF_RESULT],
])
def test_join_to_itself(rule_cls_str, key_columns_left, key_columns_right, named_input_left, named_input_right, named_output, expected, backend):
    left_df = backend.DataFrame(data=LEFT_DF)
    expected = backend.DataFrame(data=expected)
    rule_cls = getattr(backend.rules, rule_cls_str)
    with get_test_data(left_df, named_inputs={"input": left_df}, named_output=named_output) as data:
        rule = rule_cls(named_input_left=named_input_left, named_input_right=named_input_right, key_columns_left=key_columns_left, 
                        key_columns_right=key_columns_right, suffixes=["_x", "_y"], named_output=named_output)
        rule.apply(data)
        result = data.get_named_output(named_output) if named_output is not None else data.get_main_output()
        assert_frame_equal(result, expected, ignore_column_ordering=True)


def test_raises_missing_column_left(backend):
    left_df = backend.DataFrame(data=LEFT_DF)
    right_df = backend.DataFrame(data=RIGHT_DF)
    with get_test_data(named_inputs={"left": left_df, "right": right_df}) as data:
        rule = backend.rules.LeftJoinRule(named_input_left="left", named_input_right="right",
                        key_columns_left=["A", "Z"], key_columns_right=["A", "B"])
        with pytest.raises(MissingColumnError) as exc:
            rule.apply(data)
        assert str(exc.value) == "Missing columns in join in the left dataframe: {'Z'}"


def test_raises_missing_column_right(backend):
    left_df = backend.DataFrame(data=LEFT_DF)
    right_df = backend.DataFrame(data=RIGHT_DF)
    with get_test_data(named_inputs={"left": left_df, "right": right_df}) as data:
        rule = backend.rules.LeftJoinRule(named_input_left="left", named_input_right="right",
                        key_columns_left=["A", "B"], key_columns_right=["A", "Z"])
        with pytest.raises(MissingColumnError) as exc:
            rule.apply(data)
        assert str(exc.value) == "Missing columns in join in the right dataframe: {'Z'}"
