import os
import polars as pl
import zipfile

from etlrules.exceptions import MissingColumnError

from etlrules.backends.common.io.files import (
    ReadCSVFileRule as ReadCSVFileRuleBase,
    ReadParquetFileRule as ReadParquetFileRuleBase,
    WriteCSVFileRule as WriteCSVFileRuleBase,
    WriteParquetFileRule as WriteParquetFileRuleBase,
)


COMPRESSION_MAP = {
    'zip': zipfile.ZIP_DEFLATED,
    'gzip': zipfile.ZIP_DEFLATED,
    'bz2': zipfile.ZIP_BZIP2,
    'xz': zipfile.ZIP_LZMA,
}

COMPRESSION_EXT = {
    '.zip': 'zip',
    '.gz': 'gzip',
    '.bz2': 'bz2',
    '.xz': 'xz',
}

class ReadCSVFileRule(ReadCSVFileRuleBase):

    def do_read(self, file_path: str) -> pl.DataFrame:
        _, ext = os.path.splitext(file_path)
        compression = COMPRESSION_EXT.get(ext)
        if compression is not None:
            with zipfile.ZipFile(file_path, 'r', compression=COMPRESSION_MAP[compression]) as zarch:
                arch_files = zarch.namelist()
                if len(arch_files) != 1:
                    raise RuntimeError(f"One a single csv file can be read from an archive. {file_path} has {len(arch_files)} files.")
                with zarch.open(arch_files[0]) as zf:
                    return pl.read_csv(
                        zf, separator=self.separator, has_header=self.header,
                        skip_rows=self.skip_header_rows or 0
                    )
        return pl.read_csv(
            file_path, separator=self.separator, has_header=self.header,
            skip_rows=self.skip_header_rows or 0
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

    def do_write(self, file_name: str, file_dir: str, df: pl.DataFrame) -> None:
        file_path = os.path.join(file_dir, file_name)
        if self.compression is not None:
            fname, _ = os.path.splitext(file_name)
            fname_csv = fname + ".csv"
            with zipfile.ZipFile(file_path, 'w', compression=COMPRESSION_MAP[self.compression]) as zarch:
                with zarch.open(fname_csv, "w") as zf:
                    df.write_csv(zf,
                        separator=self.separator,
                        include_header=self.header,
                    )
        else:
            df.write_csv(file_path,
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
