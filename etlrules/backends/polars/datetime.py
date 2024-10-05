import datetime
import locale
import polars as pl

from .base import PolarsMixin
from etlrules.exceptions import ColumnAlreadyExistsError, MissingColumnError

from etlrules.backends.common.datetime import (
    DateTimeLocalNowRule as DateTimeLocalNowRuleBase,
    DateTimeUTCNowRule as DateTimeUTCNowRuleBase,
    DateTimeToStrFormatRule as DateTimeToStrFormatRuleBase,
    DateTimeRoundRule as DateTimeRoundRuleBase,
    DateTimeRoundDownRule as DateTimeRoundDownRuleBase,
    DateTimeRoundUpRule as DateTimeRoundUpRuleBase,
    DateTimeExtractComponentRule as DateTimeExtractComponentRuleBase,
    DateTimeAddRule as DateTimeAddRuleBase,
    DateTimeSubstractRule as DateTimeSubstractRuleBase,
    DateTimeDiffRule as DateTimeDiffRuleBase,
)


ROUND_TRUNC_UNITS_MAPPED = {
    "day": "1d",
    "hour": "1h",
    "minute": "1m",
    "second": "1s",
    "millisecond": "1ms",
    "microsecond": "1us",
}


class DateTimeRoundRule(PolarsMixin, DateTimeRoundRuleBase):
    def do_apply(self, df, series):
        offset = "-1ns" if self.unit == "microsecond" else "-1us"
        return series.dt.offset_by(offset).dt.round(
            every=ROUND_TRUNC_UNITS_MAPPED[self.unit]
        )


class DateTimeRoundDownRule(PolarsMixin, DateTimeRoundDownRuleBase):
    def do_apply(self, df, series):
        return series.dt.truncate(
            every=ROUND_TRUNC_UNITS_MAPPED[self.unit]
        )


class DateTimeRoundUpRule(PolarsMixin, DateTimeRoundUpRuleBase):

    def do_apply(self, df, series):
        return series.dt.offset_by(ROUND_TRUNC_UNITS_MAPPED[self.unit]).dt.offset_by("-1us").dt.truncate(
            every=ROUND_TRUNC_UNITS_MAPPED[self.unit]
        )


class DateTimeExtractComponentRule(PolarsMixin, DateTimeExtractComponentRuleBase):

    COMPONENTS = {
        "year": "year",
        "month": "month",
        "day": "day",
        "hour": "hour",
        "minute": "minute",
        "second": "second",
        "microsecond": "microsecond",
        "weekday": "weekday",
        "day_name": "day_name",
        "month_name": "month_name",
    }

    def _cannot_set_locale(self, loc):
        return loc and not self._set_locale(loc)

    def _set_locale(self, loc):
        if loc:
            loc = loc.split(".")
            if len(loc) != 2:
                return False
            current_locale = None
            try:
                current_locale = locale.getlocale(locale.LC_TIME)
                locale.setlocale(locale.LC_TIME, loc)
                locale.setlocale(locale.LC_TIME, current_locale)
            except locale.Error:
                return False
        return True

    def do_apply(self, df, col):
        component = self.COMPONENTS[self.component]
        if component in ("day_name", "month_name"):
            current_locale = None
            formatting = "%B" if component == "month_name" else "%A"
            try:
                if self._locale:
                    current_locale = locale.getlocale(locale.LC_TIME)
                    locale.setlocale(locale.LC_TIME, self._locale.split("."))
                res = col.dt.strftime(formatting)
            except locale.Error:
                raise ValueError(f"Unsupported locale: {self._locale}")
            finally:
                if current_locale:
                    locale.setlocale(locale.LC_TIME, current_locale)
        else:
            res = getattr(col.dt, component)().cast(pl.Int64)
            if component == "weekday":
                res = res - 1
        return res


# date arithmetic
DT_ARITHMETIC_UNITS = {
    "years": "years",
    "months": "months",
    "weeks": "weeks",
    "days": "days",
    "weekdays": None,
    "hours": "hours",
    "minutes": "minutes",
    "seconds": "seconds",
    "milliseconds": "milliseconds",
    "microseconds": "microseconds",
}

OFFSETS = {
    "years": "y",
    "months": "mo",
    "weeks": "w",
    "days": "d",
    "weekdays": "bd",
    "hours": "h",
    "minutes": "m",
    "seconds": "s",
    "milliseconds": "ms",
    "microseconds": "us",
}

DT_TIMEDELTA_UNITS = set(["weeks", "days", "hours", "minutes", "seconds", "milliseconds", "microseconds"])


def is_scalar(offset):
    return isinstance(offset, (int, float))

FOLL_MONDAY_ADJ_WEEKEND_OFFSETS = {0: None, 1: None, 2: None, 3: None, 4: None, 5: 2, 6: 1}
PREV_FRIDAY_ADJ_WEEKEND_OFFSETS = {0: None, 1: None, 2: None, 3: None, 4: None, 5: -1, 6: -2}
def dt_adjust_weekends(dt_col, offset, strict=True):
    weekdays = dt_col.dt.weekday().cast(pl.Int64) - 1
    if not is_scalar(offset):
        offset = offset.cast(pl.Int64).fill_null(0)
        # -1 -> 0 | 0, 1 -> 3
        offset = ((offset // offset.replace({0: 1}, default=offset).abs() + 1) // 2) * 3
        dt_col_weekend_offset = weekdays.replace(FOLL_MONDAY_ADJ_WEEKEND_OFFSETS, default=weekdays) - offset
    else:
        dt_col_weekend_offset = weekdays.replace(FOLL_MONDAY_ADJ_WEEKEND_OFFSETS if offset < 0 else PREV_FRIDAY_ADJ_WEEKEND_OFFSETS, default=weekdays)
    dt_col_weekend_offset = dt_col_weekend_offset.fill_null(0)
    return dt_col + pl.duration(days=dt_col_weekend_offset)


def business_day_offset(dt_col, offset, strict=True):
    if not is_scalar(offset):
        offset = offset.cast(pl.Int64).fill_null(0)
    else:
        offset = offset or 0
    dt_col2 = dt_adjust_weekends(dt_col, offset, strict=strict)
    col = (dt_col2.dt.weekday().cast(pl.Int64) - 1).fill_null(0)
    col2 =  col + offset
    col3 = (
        (col2 // 5) * 7 + col2.mod(5)
    ) - col
    return dt_col2 + pl.duration(days=col3)


def months_offset(dt_col, offset, strict=True):
    if not is_scalar(offset):
        offset = offset.fill_null(0)
    else:
        offset = offset or 0
    offset = offset + dt_col.dt.month() - 1
    year = dt_col.dt.year() + (offset // 12)

    df = year.to_frame(name="year")
    df = df.with_columns(
        month=(
            offset % 12
        ) + 1,
        day=dt_col.dt.day(),
        hour=dt_col.dt.hour(),
        minute=dt_col.dt.minute(),
        second=dt_col.dt.second(),
        microsecond=dt_col.dt.microsecond(),
    )
    df = df.with_columns(
        max=df["month"].replace({
            1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }, default=df["month"]),
    )
    df = df.with_columns(
        day=pl.when(df["day"] <= df["max"]).then(df["day"]).otherwise(df["max"])
    )
    df = df.with_columns(
        day=pl.when((df["year"] % 4 != 0) & (df["month"] == 2) & (df["day"] == 29)).then(28).otherwise(df["day"])
    )

    df = df[["year", "month", "day", "hour", "minute", "second", "microsecond"]]
    return to_datetime(df)


def to_datetime(df):
    return pl.concat_str(
        df["year"].cast(pl.Utf8),
        pl.lit("-"),
        df["month"].cast(pl.Utf8).str.pad_start(2, '0'),
        pl.lit("-"),
        df["day"].cast(pl.Utf8).str.pad_start(2, '0'),
        pl.lit(" "),
        df["hour"].cast(pl.Utf8).str.pad_start(2, '0'),
        pl.lit(":"),
        df["minute"].cast(pl.Utf8).str.pad_start(2, '0'),
        pl.lit(":"),
        df["second"].cast(pl.Utf8).str.pad_start(2, '0'),
        pl.lit("."),
        df["microsecond"].cast(pl.Utf8).str.pad_start(6, '0'),
    ).str.to_datetime(format="%Y-%m-%d %H:%M:%S%.6f", time_unit="us", ambiguous="latest")


def years_offset(dt_col, offset, strict=True):
    if not is_scalar(offset):
        offset = offset.fill_null(0)
    else:
        offset = offset or 0

    year = dt_col.dt.year() + offset

    df = year.to_frame(name="year")
    df = df.with_columns(
        month=dt_col.dt.month(),
        day=dt_col.dt.day(),
        hour=dt_col.dt.hour(),
        minute=dt_col.dt.minute(),
        second=dt_col.dt.second(),
        microsecond=dt_col.dt.microsecond(),
    )
    df = df.with_columns(
        day=pl.when((df["year"] % 4 != 0) & (df["month"] == 2) & (df["day"] == 29)).then(28).otherwise(df["day"])
    )
    df = df[["year", "month", "day", "hour", "minute", "second", "microsecond"]]
    return to_datetime(df)


def add_sub_col(df, col, unit_value, unit, sign, input_column, strict=True):
    if isinstance(unit_value, str):
        # unit_value is a column
        if unit_value not in df.columns:
            raise MissingColumnError(f"Column {unit_value} in unit_value does not exist in the input dataframe.")
        col2 = df[unit_value]
        if col2.dtype == pl.Datetime:
            if sign != -1:  # only supported for substracting a datetime from another datetime
                raise ValueError(f"Cannot add column {unit_value} of type datetime to another datetime column.")
            return col - col2
        elif col2.dtype == pl.Duration:
            pass  # do nothing for timedelta
        else:
            # another type - will be interpreted as an offset/timedelta
            if unit not in DT_ARITHMETIC_UNITS.keys():
                raise ValueError(f"Unsupported unit: '{unit}'. It must be one of {DT_ARITHMETIC_UNITS.keys()}")
            if unit in DT_TIMEDELTA_UNITS:
                col2 = pl.duration(**{DT_ARITHMETIC_UNITS[unit]: col2})
            else:
                if unit == "weekdays":
                    return business_day_offset(col, sign * col2, strict=strict)
                elif unit == "weeks":
                    return col + pl.duration(weeks=col2.fillna(0) * sign)
                elif unit == "months":
                    return months_offset(col, sign * col2, strict=strict)
                elif unit == "years":
                    return years_offset(col, sign * col2, strict=strict)
                else:
                    assert False, f"Unexpected unit {unit}"
        if sign == -1:
            return col - col2
        return col + col2
    if unit not in DT_ARITHMETIC_UNITS.keys():
        raise ValueError(f"Unsupported unit: '{unit}'. It must be one of {DT_ARITHMETIC_UNITS.keys()}")
    if unit == "weekdays":
        return business_day_offset(col, sign * unit_value, strict=strict)
    return col.dt.offset_by(f"{sign * unit_value}{OFFSETS[unit]}")


class DateTimeAddRule(PolarsMixin, DateTimeAddRuleBase):

    def do_apply(self, df, col):
        return add_sub_col(df, col, self.unit_value, self.unit, 1, self.input_column, strict=self.strict)


class DateTimeSubstractRule(PolarsMixin, DateTimeSubstractRuleBase):

    def do_apply(self, df, col):
        return add_sub_col(df, col, self.unit_value, self.unit, -1, self.input_column, strict=self.strict)


class DateTimeDiffRule(PolarsMixin, DateTimeDiffRuleBase):

    COMPONENTS = {
        "days": ("total_days", None),
        "hours": ("total_hours", 24),
        "minutes": ("total_minutes", 60),
        "seconds": ("total_seconds", 60),
        "microseconds": ("total_microseconds", 1000),
        "total_seconds": ("total_seconds", None),
    }

    def do_apply(self, df, col):
        if self.input_column2 not in df.columns:
            raise MissingColumnError(f"Column {self.input_column2} in input_column2 does not exist in the input dataframe.")
        res = add_sub_col(df, col, self.input_column2, self.unit, -1, self.input_column, strict=self.strict)
        if res.dtype == pl.Duration and self.unit:
            method, mod = self.COMPONENTS[self.unit]
            res = getattr(res.dt, method)()
            if mod is not None:
                res = res % mod
        return res


class DateTimeUTCNowRule(DateTimeUTCNowRuleBase):

    def apply(self, data):
        df = self._get_input_df(data)
        if self.strict and self.output_column in df.columns:
            raise ColumnAlreadyExistsError(f"{self.output_column} already exists in the input dataframe.")
        df = df.with_columns(
            pl.lit(datetime.datetime.utcnow()).alias(self.output_column)
        )
        self._set_output_df(data, df)


class DateTimeLocalNowRule(DateTimeLocalNowRuleBase):

    def apply(self, data):
        df = self._get_input_df(data)
        if self.strict and self.output_column in df.columns:
            raise ColumnAlreadyExistsError(f"{self.output_column} already exists in the input dataframe.")
        df = df.with_columns(
            pl.lit(datetime.datetime.now()).alias(self.output_column)
        )
        self._set_output_df(data, df)


class DateTimeToStrFormatRule(PolarsMixin, DateTimeToStrFormatRuleBase):

    def do_apply(self, df, col):
        return col.dt.strftime(self.format)
