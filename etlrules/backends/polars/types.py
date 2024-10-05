import polars as pl

from etlrules.backends.common.types import TypeConversionRule as TypeConversionRuleBase

from .base import PolarsMixin

MAP_TYPES = {
    'int8': pl.Int8,
    'int16': pl.Int16,
    'int32': pl.Int32,
    'int64': pl.Int64,
    'uint8': pl.UInt8,
    'uint16': pl.UInt16,
    'uint32': pl.UInt32,
    'uint64': pl.UInt64,
    'float32': pl.Float32,
    'float64': pl.Float64,
    'string': pl.Utf8,
    'boolean': pl.Boolean,
}


class TypeConversionRule(TypeConversionRuleBase, PolarsMixin):

    def do_type_conversion(self, df, col, dtype):
        if self.strict:
            try:
                col.cast(MAP_TYPES[dtype], strict=self.strict)
            except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as exc:
                raise ValueError(str(exc))
        return col.cast(MAP_TYPES[dtype], strict=self.strict)
