import os
import dask.dataframe as dd

from etlrules.exceptions import MissingColumnError

from etlrules.backends.common.io.files import (
    ReadCSVFileRule as ReadCSVFileRuleBase,
    ReadParquetFileRule as ReadParquetFileRuleBase,
    WriteCSVFileRule as WriteCSVFileRuleBase,
    WriteParquetFileRule as WriteParquetFileRuleBase,
)


class ReadCSVFileRule(ReadCSVFileRuleBase):
    def do_read(self, file_path: str) -> dd.DataFrame:
        return dd.read_csv(
            file_path, blocksize=None, sep=self.separator, header='infer' if self.header else None,
            skiprows=self.skip_header_rows,
            index_col=False
        )


def parquet_file_name_split(file_name: str) -> tuple[str, str]:
    fn, ext = os.path.splitext(file_name)
    return fn, ext or "parquet"


class ReadParquetFileRule(ReadParquetFileRuleBase):

    def do_read(self, file_path: str) -> dd.DataFrame:
        from pyarrow.lib import ArrowInvalid
        file_dir, file_name = os.path.split(file_path)
        fn, ext = parquet_file_name_split(file_name)
        try:
            return dd.read_parquet(
                os.path.join(file_dir, f"{fn}*.{ext}"), engine="pyarrow", columns=self.columns, filters=self.filters
            )
        except ArrowInvalid as exc:
            raise MissingColumnError(str(exc))
        except ValueError as exc:
            if "The following columns were not found in the dataset" in str(exc):
                raise MissingColumnError(str(exc))
            raise


class WriteCSVFileRule(WriteCSVFileRuleBase):

    def do_write(self, file_name: str, file_dir: str,  df: dd.DataFrame) -> None:
        df.to_csv(
            os.path.join(file_dir, file_name),
            single_file=True,
            sep=self.separator,
            header=self.header,
            compression=self.compression,
            index=False,
        )


class WriteParquetFileRule(WriteParquetFileRuleBase):

    def do_write(self, file_name: str, file_dir: str, df: dd.DataFrame) -> None:
        fn, ext = parquet_file_name_split(file_name)
        df.to_parquet(
            path=file_dir,
            engine="pyarrow",
            write_metadata_file=False,
            name_function=lambda idx: f"{fn}_part_{idx}.{ext}",
            compression=self.compression,
            write_index=False
        )

