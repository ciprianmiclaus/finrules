import datetime
import os
import pytest

from etlrules.backends.common.io.files import WriteCSVFileRule
from tests.utils.data import assert_frame_equal, get_test_data


TEST_DF = [
    {"A": 1, "B": True, "C": "c1", "D": datetime.datetime(2023, 5, 23, 10, 30, 45)},
    {"A": 2, "B": False, "C": "c2", "D": datetime.datetime(2023, 5, 24, 11, 30, 45)},
    {"A": 3, "B": True, "C": "c3", "D": datetime.datetime(2023, 5, 25, 12, 30, 45)},
    {"B": False, "D": datetime.datetime(2023, 5, 26, 13, 30, 45)},
    {"A": 4, "C": "c4"},
    {}
]


@pytest.mark.parametrize("compression",
    [None] + 
    list(WriteCSVFileRule.COMPRESSIONS)
)
def test_write_read_csv_file(compression, backend):
    extension = WriteCSVFileRule.COMPRESSIONS[compression] if compression else ""
    test_df = backend.DataFrame(data=TEST_DF)
    try:
        with get_test_data(test_df, named_inputs={"input": test_df}, named_output="result") as data:
            write_rule = backend.rules.WriteCSVFileRule(file_name="tst.csv" + extension, file_dir="/tmp", compression=compression, named_input="input")
            write_rule.apply(data)
            read_rule = backend.rules.ReadCSVFileRule(file_name="tst.csv" + extension, file_dir="/tmp", named_output="result")
            read_rule.apply(data)
            result = data.get_named_output("result")
            result = backend.astype(result, {"D": "datetime"})
            assert_frame_equal(result, test_df)
    finally:
        os.remove(os.path.join("/tmp", "tst.csv" + extension))


@pytest.mark.parametrize("separator,skip_header_rows", [
    [",", None],
    ["|", None],
    [",", 0],
    [",", 3],
])
def test_write_read_csv_file_options(separator, skip_header_rows, backend):
    try:
        test_df = backend.DataFrame(data=TEST_DF)
        with get_test_data(test_df, named_inputs={"input": test_df}, named_output="result") as data:
            write_rule = backend.rules.WriteCSVFileRule(file_name="tst.csv", file_dir="/tmp", separator=separator, named_input="input")
            write_rule.apply(data)
            if skip_header_rows:
                with open("/tmp/tst.csv", "rt") as f:
                    contents = f.read()
                with open("/tmp/tst.csv", "wt") as f:
                    for x in range(skip_header_rows):
                        f.write(f"skipped line {x}\n")
                    f.write(contents)
            read_rule = backend.rules.ReadCSVFileRule(file_name="tst.csv", file_dir="/tmp", separator=separator, skip_header_rows=skip_header_rows, named_output="result")
            read_rule.apply(data)
            result = data.get_named_output("result")
            result = backend.astype(result, {"D": "datetime"})
            assert_frame_equal(result, test_df)
    finally:
        os.remove(os.path.join("/tmp", "tst.csv"))


def test_write_read_csv_file_no_header(backend):
    try:
        test_df = backend.DataFrame(data=TEST_DF)
        with get_test_data(test_df, named_inputs={"input": test_df}, named_output="result") as data:
            write_rule = backend.rules.WriteCSVFileRule(file_name="tst.csv", file_dir="/tmp", header=False, named_input="input")
            write_rule.apply(data)
            read_rule = backend.rules.ReadCSVFileRule(file_name="tst.csv", file_dir="/tmp", header=False, named_output="result")
            read_rule.apply(data)
            result = data.get_named_output("result")
            result = backend.rename(result, {0: "A", 1: "B", 2: "C", 3: "D"})
            result = backend.astype(result, {"D": "datetime"})
            assert_frame_equal(result, test_df)
    finally:
        os.remove(os.path.join("/tmp", "tst.csv"))


EXPECTED = [
    {"Id": 1, "FirstName": "Mike", "LastName": "Geoffrey", "Book": "Like a python", "Year": 2012},
    {"Id": 2, "FirstName": "Adam", "LastName": "Rolley", "Book": "How to make friends", "Year": 1998},
    {"Id": 3, "FirstName": "Michelle", "LastName": "Saville", "Book": "Scent of a pear", "Year": 2022},
    {"Id": 4, "FirstName": "Dorothy", "LastName": "Andrews", "Book": "Tell me your name", "Year": 2014},
    {"Id": 5, "FirstName": "John", "LastName": "Rawley", "Book": "Make me a lasagna", "Year": 2011},
]

def test_read_csv_file_via_http(backend):
    url = "https://raw.githubusercontent.com/ciprianmiclaus/etlrules/main/examples/csv2db/csv_sample.csv"
    with get_test_data(None, named_inputs={}, named_output="result") as data:
        read_rule = backend.rules.ReadCSVFileRule(file_name=url, header=True, named_output="result")
        read_rule.apply(data)
        actual = data.get_named_output("result")
        expected = backend.DataFrame(data=EXPECTED)
        assert_frame_equal(actual, expected)


def test_read_csv_file_via_http_regex(backend):
    url = "https://raw.githubusercontent.com/ciprianmiclaus/etlrules/main/examples/csv2db/.*.csv"
    with get_test_data(None, named_inputs={}, named_output="result") as data:
        with pytest.raises(ValueError) as exc:
            backend.rules.ReadCSVFileRule(file_name=url, regex=True, header=False, named_output="result")
        assert str(exc.value) == "Regex read not supported for URIs."
