import os, re
from typing import List, NoReturn, Optional, Sequence, Tuple, Union

from etlrules.rule import BaseRule, UnaryOpBaseRule
from etlrules.backends.common.substitution import subst_string


class BaseReadFileRule(BaseRule):
    def __init__(self, file_name: str, file_dir: Optional[str]=None, regex: bool=False, named_output: Optional[str]=None, name: Optional[str]=None, description: Optional[str]=None, strict: bool=True):
        super().__init__(named_output=named_output, name=name, description=description, strict=strict)
        self.file_name = file_name
        self.file_dir = file_dir
        self.regex = bool(regex)
        if self._is_uri() and self.regex:
            raise ValueError("Regex read not supported for URIs.")

    def _is_uri(self):
        file_name = self.file_name.lower()
        return file_name.startswith("http://") or file_name.startswith("https://")

    def has_input(self):
        return False

    def _get_full_file_paths(self):
        file_name = subst_string(self.file_name)
        file_dir = subst_string(self.file_dir or "")
        if self.regex:
            pattern = re.compile(file_name)
            for fn in os.listdir(file_dir):
                if pattern.match(fn):
                    yield os.path.join(file_dir, fn)
        else:
            if self._is_uri():
                yield file_name
            else:
                yield os.path.join(file_dir, file_name)

    def do_read(self, file_path: str):
        raise NotImplementedError("Have you imported the rules from etlrules.backends.<your_backend> and not common?")

    def do_concat(self, left_df, right_df):
        raise NotImplementedError("Have you imported the rules from etlrules.backends.<your_backend> and not common?")

    def apply(self, data):
        super().apply(data)

        result = None
        for file_path in self._get_full_file_paths():
            df = self.do_read(file_path)
            if result is None:
                result = df
            else:
                result = self.do_concat(result, df)
        self._set_output_df(data, result)


class ReadCSVFileRule(BaseReadFileRule):
    r""" Reads one or multiple csv files from a directory and persists it as a dataframe for subsequent rules to operate on.

    Basic usage::

        # reads a file data.csv and persists it as the main output of the rule
        rule = ReadCSVFileRule("data.csv", "/home/myuser/")
        rule.apply(data)

        # reads a file test_data.csv and persists it as the input_data named output
        # other rules can specify input_data as their named_input to operate on it
        rule = ReadCSVFileRule("test_data.csv", "/home/myuser/", named_output="input_data")
        rule.apply(data)

        # extracts all files starting with data followed by 4 digits and concatenate them
        # e.g. data1234.csv, data5678.csv, etc.
        rule = ReadCSVFileRule("data[0-9]{4}.csv", "/home/myuser/", regex=True, named_output="input_data")
        rule.apply(data)

    Args:
        file_name: The name of the csv file to load. The format will be inferred from the extension of the file.
            A simple text csv file will be inferred from the .csv extension. The extensions like .zip, .gz, .bz2, .xz
            will extract a single compressed csv file from the given input compressed file.
            file_name can also be a regular expression (specify regex=True in that case).
            The reader will find all the files in the file_dir directory that match the regular expression and extract
            all those csv file and concatenate them into a single dataframe.
            For example, file_name=".*\.csv", file_dir=".", regex=True will extract all the files with the .csv extension
            from the current directory.
            It can also be an URI (e.g. https://example.com/mycsv.csv)
        file_dir: The file directory where the file_name is located. When file_name is a regular expression and 
            the regex parameter is True, file_dir is the directory that is inspected for any files that match the
            regular expression. Optional.
            For files it defaults to . (ie the current directory). Ignored for URIs.
        regex: When True, the file_name is interpreted as a regular expression. Defaults to False.
        separator: The single character to be used as separator in the csv file. Defaults to , (comma).
        header: When True, the first line is interpreted as the header and the column names are extracted from it.
            When False, the first line is part of the data and the columns will have names like 0, 1, 2, etc.
            Defaults to True.
        skip_header_rows: Optional number of rows to skip at the top of the file, before the header.

        named_output (Optional[str]): Give the output of this rule a name so it can be used by another rule as a named input. Optional.
            When not set, the result of this rule will be available as the main output.
            When set to a name (string), the result will be available as that named output.
        name (Optional[str]): Give the rule a name. Optional.
            Named rules are more descriptive as to what they're trying to do/the intent.
        description (Optional[str]): Describe in detail what the rules does, how it does it. Optional.
            Together with the name, the description acts as the documentation of the rule.
        strict (bool): When set to True, the rule does a stricter valiation. Default: True

    Raises:
        IOError: raised when the file is not found.
    """

    def __init__(self, file_name: str, file_dir: Optional[str]=None, regex: bool=False, separator: str=",",
                 header: bool=True, skip_header_rows: Optional[int]=None,
                 named_output: Optional[str]=None, name: Optional[str]=None, description: Optional[str]=None, strict: bool=True):
        super().__init__(file_name=file_name, file_dir=file_dir, regex=regex, named_output=named_output, name=name, description=description, strict=strict)
        self.separator = separator
        self.header = header
        self.skip_header_rows = skip_header_rows


class ReadParquetFileRule(BaseReadFileRule):
    r""" Reads one or multiple parquet files from a directory and persists it as a dataframe for subsequent rules to operate on.

    Basic usage::

        # reads a file data.csv and persists it as the main output of the rule
        rule = ReadParquetFileRule("data.parquet", "/home/myuser/")
        rule.apply(data)

        # reads a file test_data.parquet and persists it as the input_data named output
        # other rules can specify input_data as their named_input to operate on it
        rule = ReadParquetFileRule("test_data.parquet", "/home/myuser/", named_output="input_data")
        rule.apply(data)

        # reads all the files with the .parquet extension from the home dir of myuser and
        # concatenates them into a single dataframe
        rule = ReadParquetFileRule(".*\.parquet", "/home/myuser/", named_output="input_data")
        rule.apply(data)

        # reads only the A,B,C columns from the file data.csv file
        rule = ReadParquetFileRule("data.parquet", "/home/myuser/", columns=["A", "B", "C"])
        rule.apply(data)

        # reads only those rows where column A is greater than 10 and column B is True
        rule = ReadParquetFileRule("data.parquet", "/home/myuser/", filters=[["A", ">=", 10], ["B", "==", True]])
        rule.apply(data)

    Args:
        file_name: The name of the parquet file to load. The format will be inferred from the extension of the file.
            file_name can also be a regular expression (specify regex=True in that case).
            The reader will find all the files in the file_dir directory that match the regular expression and extract
            all those parquet file and concatenate them into a single dataframe.
            For example, file_name=".*\.parquet", file_dir=".", regex=True will extract all the files with the .parquet extension
            from the current directory.
        file_dir: The file directory where the file_name is located. When file_name is a regular expression and 
            the regex parameter is True, file_dir is the directory that is inspected for any files that match the
            regular expression.
            Defaults to . (ie the current directory).
        regex: When True, the file_name is interpreted as a regular expression. Defaults to False.
        columns: A subset of the columns in the parquet file to load.
        filters: A list of filters to apply to filter the rows returned. Rows which do not match the filter conditions
            will be removed from scanned data.
            When passed as a List[List[Tuple]], the conditions in the inner lists are AND-end together with the top level
            condition OR-ed together. Eg: ((cond1 AND cond2...) OR (cond3 AND cond4...)...)
            When passed as a List[Tuple], the conditions are AND-ed together. E.g.: cond1 AND cond2 AND cond3...
            Each condition is specified as a tuple of 3 elements: (column, operation, value).
            Column is the name of a column in the input dataframe.
            Operation is one of: "==", "=", ">", ">=", "<", "<=", "!=", "in", "not in".
            Value is a scalar value, int, float, string, etc. When the operation is in or not in, the value must be a list, tuple or set of values.

        named_output (Optional[str]): Give the output of this rule a name so it can be used by another rule as a named input. Optional.
            When not set, the result of this rule will be available as the main output.
            When set to a name (string), the result will be available as that named output.
        name (Optional[str]): Give the rule a name. Optional.
            Named rules are more descriptive as to what they're trying to do/the intent.
        description (Optional[str]): Describe in detail what the rules does, how it does it. Optional.
            Together with the name, the description acts as the documentation of the rule.
        strict (bool): When set to True, the rule does a stricter valiation. Default: True

    Raises:
        IOError: raised when the file is not found.
        ValueError: raised if filters are specified but the format is incorrect.
        MissingColumnError: raised if a column is specified in columns or filters but it doesn't exist in the input dataframe.
    
    Note:
        The parquet file can be compressed in which case the compression will be inferred from the file.
        The following compression algorithms are supported: "snappy", "gzip", "brotli", "lz4", "zstd".
    """

    SUPPORTED_FILTERS_OPS = {"==", "=", ">", ">=", "<", "<=", "!=", "in", "not in"}

    def __init__(self, file_name: str, file_dir: str=".", columns: Optional[Sequence[str]]=None, filters:Optional[Union[List[Tuple], List[List[Tuple]]]]=None, regex: bool=False, named_output: Optional[str]=None, name: Optional[str]=None, description: Optional[str]=None, strict: bool=True):
        super().__init__(
            file_name=file_name, file_dir=file_dir, regex=regex, named_output=named_output,
            name=name, description=description, strict=strict)
        self.columns = columns
        self.filters = self._get_filters(filters) if filters is not None else None

    def _raise_filters_invalid(self, error: str) -> NoReturn:
        raise ValueError(f"Invalid filters. It must be a List[Tuple] or List[List[Tuple]] with each Tuple being (column, op, value): {error}")

    def _validate_tuple(self, tpl):
        if len(tpl) != 3 or not isinstance(tpl[0], str) or not isinstance(tpl[1], str):
            self._raise_filters_invalid(f"Third level expected a list/tuple (cond, op, value), got: {tpl}.")
        op = tpl[1]
        if op not in self.SUPPORTED_FILTERS_OPS:
            self._raise_filters_invalid(f"Invalid operator {op} in {tpl}. Must be one of: {self.SUPPORTED_FILTERS_OPS}.")
        value = tpl[2]
        if op in ("in", "not in"):
            if not isinstance(value, (list, tuple, set)):
                self._raise_filters_invalid(f"Invalid value type for {value} for {op} operand in {tpl}. Must be list/tuple/set.")
            else:
                value = list(value)
        return (tpl[0], op, value)

    def _get_filters(self, filters):
        lst = []
        if isinstance(filters, (list, tuple)):
            if not filters:
                return None
            for filter2 in filters:
                if isinstance(filter2, (list, tuple)) and filter2:
                    if len(filter2) == 3 and isinstance(filter2[0], str):
                        # List[Tuple] form
                        tpl = self._validate_tuple(filter2)
                        lst.append(tpl)
                    else:
                        lst2 = []
                        for filter3 in filter2:
                            if isinstance(filter3, (list, tuple)) and filter3:
                                tpl = self._validate_tuple(filter3)
                                lst2.append(tpl)
                            else:
                                self._raise_filters_invalid(f"Third level expected a list/tuple, got: {filter3}.")
                        lst.append(lst2)
                else:
                    self._raise_filters_invalid(f"Second level expected a list/tuple, got: {filter2}.")
        else:
           self._raise_filters_invalid(f"Top level expected a list/tuple, got: {filters}")
        return lst


class BaseWriteFileRule(UnaryOpBaseRule):

    EXCLUDE_FROM_SERIALIZE = ("named_output", )

    def __init__(self, file_name, file_dir=".", named_input=None, name=None, description=None, strict=True):
        super().__init__(named_input=named_input, named_output=None, name=name, description=description, strict=strict)
        self.file_name = file_name
        self.file_dir = file_dir

    def has_output(self):
        return False

    def do_write(self, file_name: str, file_dir: str, df) -> None:
        raise NotImplementedError("Have you imported the rules from etlrules.backends.<your_backend> and not common?")

    def apply(self, data):
        super().apply(data)
        df = self._get_input_df(data)
        self.do_write(subst_string(self.file_name), subst_string(self.file_dir), df)


class WriteCSVFileRule(BaseWriteFileRule):
    """ Writes an existing dataframe to a csv file (optionally compressed) on disk.

    The rule is a final rule, which means it produces no additional outputs, it takes any of the existing outputs and writes it to disk.

    Basic usage::

        # writes a file data.csv and persists the main output of the previous rule to it
        rule = WriteCSVFileRule("data.csv", "/home/myuser/")
        rule.apply(data)

        # writes a file test_data.csv and persists the dataframe named input_data into it
        rule = WriteCSVFileRule("test_data.csv", "/home/myuser/", named_input="input_data")
        rule.apply(data)

    Args:
        file_name: The name of the csv file to write to disk. It will be written in the directory
            specified by the file_dir parameter.
        file_dir: The file directory where the file_name should be written.
            Defaults to . (ie the current directory).
        separator: The single character to separate values in the csv file. Defaults to , (comma).
        header: When True, the first line will contain the columns separated by the separator.
            When False, the columns will not be written and the first line contains data.
            Defaults to True.
        compression: Compress the csv file using a supported compression algorithms. Optional.
            When the compression is specified, the file_name must end with the extension associate with that
            compression format. The following options are supported:
            zip - file_name must end with .zip (e.g. output.csv.zip), will produced a zipped csv file
            gzip - file_name must end with .gz (e.g. output.csv.gz), will produced a gzipped csv file
            bz2 - file_name must end with .bz2 (e.g. output.csv.bz2), will produced a bzipped csv 
            xz - file_name must end with .xz (e.g. output.csv.xz), will produced a xz-compressed csv file

        named_input (Optional[str]): Select by name the dataframe to write from the input data.
            Optional. When not specified, the main output of the previous rule will be written.
        name (Optional[str]): Give the rule a name. Optional.
            Named rules are more descriptive as to what they're trying to do/the intent.
        description (Optional[str]): Describe in detail what the rules does, how it does it. Optional.
            Together with the name, the description acts as the documentation of the rule.
        strict (bool): When set to True, the rule does a stricter valiation. Default: True.
    """

    COMPRESSIONS = {
        'zip': '.zip',
        'gzip': '.gz',
        'bz2': '.bz2',
        'xz': '.xz',
    }

    def __init__(self, file_name: str, file_dir: str=".", separator: str=",", header: bool=True, compression: Optional[str]=None, named_input: Optional[str]=None, name: Optional[str]=None, description: Optional[str]=None, strict: bool=True):
        super().__init__(
            file_name=file_name, file_dir=file_dir, named_input=named_input, 
            name=name, description=description, strict=strict)
        self.separator = separator
        self.header = header
        assert compression is None or compression in self.COMPRESSIONS.keys(), f"Unsupported compression '{compression}'. It must be one of: {self.COMPRESSIONS.keys()}."
        if compression:
            assert file_name.endswith(self.COMPRESSIONS[compression]), f"The file name {file_name} must have the extension {self.COMPRESSIONS[compression]} when the compression is set to {compression}."
        self.compression = compression


class WriteParquetFileRule(BaseWriteFileRule):
    """ Writes an existing dataframe to a parquet file on disk.

    The rule is a final rule, which means it produces no additional outputs, it takes any of the existing outputs and writes it to disk.

    Basic usage::

        # writes a file data.parquet and persists the main output of the previous rule to it
        rule = WriteParquetFileRule("data.parquet", "/home/myuser/")
        rule.apply(data)

        # writes a file test_data.parquet and persists the dataframe named input_data into it
        rule = WriteParquetFileRule("test_data.parquet", "/home/myuser/", named_input="input_data")
        rule.apply(data)

    Args:
        file_name: The name of the parquet file to write to disk. It will be written in the directory
            specified by the file_dir parameter.
        file_dir: The file directory where the file_name should be written.
            Defaults to . (ie the current directory).
        compression: Compress the parquet file using a supported compression algorithms. Optional.
            The following compression algorithms are supported: "snappy", "gzip", "brotli", "lz4", "zstd".

        named_input (Optional[str]): Select by name the dataframe to write from the input data.
            Optional. When not specified, the main output of the previous rule will be written.
        name (Optional[str]): Give the rule a name. Optional.
            Named rules are more descriptive as to what they're trying to do/the intent.
        description (Optional[str]): Describe in detail what the rules does, how it does it. Optional.
            Together with the name, the description acts as the documentation of the rule.
        strict (bool): When set to True, the rule does a stricter valiation. Default: True.
    """

    COMPRESSIONS = ("snappy", "gzip", "brotli", "lz4", "zstd")

    def __init__(self, file_name: str, file_dir: str=".", compression: Optional[str]=None, named_input: Optional[str]=None, name: Optional[str]=None, description: Optional[str]=None, strict: bool=True):
        super().__init__(
            file_name=file_name, file_dir=file_dir, named_input=named_input, 
            name=name, description=description, strict=strict)
        assert compression is None or compression in self.COMPRESSIONS, f"Unsupported compression '{compression}'. It must be one of: {self.COMPRESSIONS}."
        self.compression = compression

