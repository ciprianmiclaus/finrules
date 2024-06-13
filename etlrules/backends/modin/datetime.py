import datetime
import locale
import logging
import numpy as np
from modin.pandas import to_timedelta, isnull, to_datetime
from modin.pandas import DateOffset
# TODO: FIX this
from pandas.api.types import is_scalar, is_timedelta64_dtype, is_datetime64_any_dtype

from .base import PandasMixin
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

perf_logger = logging.getLogger("etlrules.perf")


ROUND_TRUNC_UNITS_MAPPED = {
    "day": "D",
    "hour": "H",
    "minute": "T",
    "second": "S",
    "millisecond": "L",
    "microsecond": "U",
    "nanosecond": "N",
}


class DateTimeRoundRule(PandasMixin, DateTimeRoundRuleBase):
    def do_apply(self, df, series):
        return series.dt.round(
            freq=ROUND_TRUNC_UNITS_MAPPED[self.unit],
            ambiguous='infer',
            nonexistent='shift_forward'
        )


class DateTimeRoundDownRule(PandasMixin, DateTimeRoundDownRuleBase):
    def do_apply(self, df, series):
        return series.dt.floor(
            freq=ROUND_TRUNC_UNITS_MAPPED[self.unit],
            ambiguous='infer',
            nonexistent='shift_forward'
        )


class DateTimeRoundUpRule(PandasMixin, DateTimeRoundUpRuleBase):
    def do_apply(self, df, series):
        return series.dt.ceil(
            freq=ROUND_TRUNC_UNITS_MAPPED[self.unit],
            ambiguous='infer',
            nonexistent='shift_forward'
        )


class DateTimeExtractComponentRule(PandasMixin, DateTimeExtractComponentRuleBase):

    COMPONENTS = {
        "year": "year",
        "month": "month",
        "day": "day",
        "hour": "hour",
        "minute": "minute",
        "second": "second",
        "microsecond": "microsecond",
        "nanosecond": "nanosecond",
        "weekday": "weekday",
        "day_name": "day_name",
        "month_name": "month_name",
    }

    def do_apply(self, df, col):
        component = self.COMPONENTS[self.component]
        res = getattr(col.dt, component)
        if component in ("day_name", "month_name"):
            try:
                res = res(locale=self._locale)
            except locale.Error:
                raise ValueError(f"Unsupported locale: {self._locale}")
        if component in ("day_name", "month_name"):
            res = res.astype('string')
        else:
            res = res.astype('Int64')
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
    "nanoseconds": "nanoseconds",
}

DT_TIMEDELTA_UNITS = set(["days", "hours", "minutes", "seconds", "milliseconds", "microseconds", "nanoseconds"])

FOLL_MONDAY_ADJ_WEEKEND_OFFSETS = {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: 2, 6: 1}
PREV_FRIDAY_ADJ_WEEKEND_OFFSETS = {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: -1, 6: -2}
def dt_adjust_weekends(dt_col, offset, strict=True):
    weekdays = dt_col.dt.weekday
    if not is_scalar(offset):
        offset = offset.fillna(0)
        # -1 -> 0 | 0, 1 -> 3
        offset = ((offset // offset.replace({0: 1}).abs() + 1) // 2) * 3
        dt_col_weekend_offset = weekdays.replace(FOLL_MONDAY_ADJ_WEEKEND_OFFSETS) - offset
    else:
        dt_col_weekend_offset = weekdays.replace(FOLL_MONDAY_ADJ_WEEKEND_OFFSETS if offset < 0 else PREV_FRIDAY_ADJ_WEEKEND_OFFSETS)
    dt_col_weekend_offset = dt_col_weekend_offset.fillna(0)
    return dt_col + to_timedelta(dt_col_weekend_offset, unit="D", errors="raise" if strict else "coerce")


def business_day_offset(dt_col, offset, strict=True):
    if not is_scalar(offset):
        offset = offset.fillna(0)
    else:
        offset = offset or 0
    dt_col2 = dt_adjust_weekends(dt_col, offset, strict=strict)
    col = dt_col2.dt.weekday.fillna(0)
    col2 =  col + offset
    col3 = (
        (col2 // 5) * 7 + (col2 % 5)
    ) - col
    col4 = to_timedelta(col3, unit="D", errors="raise" if strict else "coerce")
    return dt_col2 + col4


def months_offset(dt_col, offset, strict=True):
    if not is_scalar(offset):
        offset = offset.fillna(0)
    else:
        offset = offset or 0
    offset = offset + dt_col.dt.month - 1
    year = dt_col.dt.year + (offset // 12)

    df = year.to_frame(name="year")
    df["month"] = (offset % 12) + 1
    df["day"] = dt_col.dt.day
    df["hour"] = dt_col.dt.hour
    df["minute"] = dt_col.dt.minute
    df["second"] = dt_col.dt.second
    df["microsecond"] = dt_col.dt.microsecond

    df["max"] = df["month"].replace({
        1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    })
    df["day"] = df["day"].where(df["day"] <= df["max"], df["max"])
    df["day"] = df["day"].mask((df["year"] % 4 != 0) & (df["month"] == 2) & (df["day"] == 29), 28)

    df = df[["year", "month", "day", "hour", "minute", "second", "microsecond"]]
    df = to_datetime(df, errors="raise" if strict else "coerce")
    return df


def years_offset(dt_col, offset, strict=True):
    if not is_scalar(offset):
        offset = offset.fillna(0)
    else:
        offset = offset or 0

    year = dt_col.dt.year + offset

    df = year.to_frame(name="year")
    df["month"] = dt_col.dt.month
    df["day"] = dt_col.dt.day
    df["hour"] = dt_col.dt.hour
    df["minute"] = dt_col.dt.minute
    df["second"] = dt_col.dt.second
    df["microsecond"] = dt_col.dt.microsecond

    df["day"] = df["day"].mask((df["year"] % 4 != 0) & (df["month"] == 2) & (df["day"] == 29), 28)

    df = df[["year", "month", "day", "hour", "minute", "second", "microsecond"]]
    df = to_datetime(df, errors="raise" if strict else "coerce")
    return df


def add_sub_col(df, col, unit_value, unit, sign, strict=True):
    if isinstance(unit_value, str):
        # unit_value is a column
        if unit_value not in df.columns:
            raise MissingColumnError(f"Column {unit_value} in unit_value does not exist in the input dataframe.")
        col2 = df[unit_value]
        if is_datetime64_any_dtype(col2):
            if sign != -1:  # only supported for substracting a datetime from another datetime
                raise ValueError(f"Cannot add column {unit_value} of type datetime to another datetime column.")
            return col - col2
        elif is_timedelta64_dtype(col2):
            pass  # do nothing for timedelta
        else:
            # another type - will be interpreted as an offset/timedelta
            if unit not in DT_ARITHMETIC_UNITS.keys():
                raise ValueError(f"Unsupported unit: '{unit}'. It must be one of {DT_ARITHMETIC_UNITS.keys()}")
            if unit in DT_TIMEDELTA_UNITS:
                col2 = to_timedelta(col2, unit=DT_ARITHMETIC_UNITS[unit], errors="coerce")
            else:
                if unit == "weekdays":
                    return business_day_offset(col, sign * col2, strict=strict)
                elif unit == "weeks":
                    return to_datetime(
                        col + to_timedelta(col2.fillna(0) * sign * 7, "D", errors="raise" if strict else "coerce"),
                        errors="raise" if strict else "coerce"
                    )
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
    return col + DateOffset(**{DT_ARITHMETIC_UNITS[unit]: sign * unit_value})


class DateTimeAddRule(PandasMixin, DateTimeAddRuleBase):

    def do_apply(self, df, col):
        return add_sub_col(df, col, self.unit_value, self.unit, 1, strict=self.strict)


class DateTimeSubstractRule(PandasMixin, DateTimeSubstractRuleBase):

    def do_apply(self, df, col):
        return add_sub_col(df, col, self.unit_value, self.unit, -1, strict=self.strict)


class DateTimeDiffRule(PandasMixin, DateTimeDiffRuleBase):

    COMPONENTS = {
        "days": "days",
        "hours": "hours",
        "minutes": "minutes",
        "seconds": "seconds",
        "microseconds": "microseconds",
        "nanoseconds": "nanoseconds",
        "total_seconds": None,
    }

    def do_apply(self, df, col):
        if self.input_column2 not in df.columns:
            raise MissingColumnError(f"Column {self.input_column2} in input_column2 does not exist in the input dataframe.")
        res = add_sub_col(df, col, self.input_column2, self.unit, -1, strict=self.strict)
        if is_timedelta64_dtype(res) and self.unit:
            if self.unit == "total_seconds":
                res = res.dt.total_seconds()
            else:
                res = res.dt.components[self.COMPONENTS[self.unit]]
            res = res.astype("Int64")
        return res


class DateTimeUTCNowRule(DateTimeUTCNowRuleBase):

    def apply(self, data):
        df = self._get_input_df(data)
        if self.strict and self.output_column in df.columns:
            raise ColumnAlreadyExistsError(f"{self.output_column} already exists in the input dataframe.")
        df = df.assign(**{self.output_column: datetime.datetime.utcnow()})
        self._set_output_df(data, df)


class DateTimeLocalNowRule(DateTimeLocalNowRuleBase):

    def apply(self, data):
        df = self._get_input_df(data)
        if self.strict and self.output_column in df.columns:
            raise ColumnAlreadyExistsError(f"{self.output_column} already exists in the input dataframe.")
        df = df.assign(**{self.output_column: datetime.datetime.now()})
        self._set_output_df(data, df)


class DateTimeToStrFormatRule(PandasMixin, DateTimeToStrFormatRuleBase):

    def do_apply(self, df, col):
        return col.dt.strftime(self.format).astype('string')
