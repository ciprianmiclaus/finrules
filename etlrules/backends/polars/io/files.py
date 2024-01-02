import os
import polars as pl

from etlrules.exceptions import MissingColumnError

from etlrules.backends.common.io.files import (
    ReadCSVFileRule as ReadCSVFileRuleBase,
    ReadParquetFileRule as ReadParquetFileRuleBase,
    WriteCSVFileRule as WriteCSVFileRuleBase,
    WriteParquetFileRule as WriteParquetFileRuleBase,
)


class ReadCSVFileRule(ReadCSVFileRuleBase):

    # TODO: support inferring of compressions

    def do_read(self, file_path: str) -> pl.DataFrame:
        return pl.read_csv(
            file_path, separator=self.separator, has_header=self.header
        )


class ReadParquetFileRule(ReadParquetFileRuleBase):
    def do_read(self, file_path: str) -> pl.DataFrame:
        from pyarrow.lib import ArrowInvalid
        try:
            return pl.read_parquet(
                file_path, use_pyarrow=True, columns=self.columns,
                pyarrow_options={
                    "filters": self.filters
                }
            )
        except ArrowInvalid as exc:
            raise MissingColumnError(str(exc))


class WriteCSVFileRule(WriteCSVFileRuleBase):

    # TODO: support compressions

    def do_write(self, file_name: str, file_dir: str, df: pl.DataFrame) -> None:
        df.write_csv(
            os.path.join(file_dir, file_name),
            separator=self.separator,
            include_header=self.header,
        )


class WriteParquetFileRule(WriteParquetFileRuleBase):

    def do_write(self, file_name: str, file_dir: str, df: pl.DataFrame) -> None:
        df.write_parquet(
            os.path.join(file_dir, file_name),
            use_pyarrow=True,
            compression=self.compression or "uncompressed",
        )
