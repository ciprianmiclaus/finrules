import polars as pl

from etlrules.backends.common.io.db import (
    ReadSQLQueryRule as ReadSQLQueryRuleBase,
    WriteSQLTableRule as WriteSQLTableRuleBase,
)
from etlrules.backends.polars.types import MAP_TYPES
from etlrules.exceptions import SQLError


class ReadSQLQueryRule(ReadSQLQueryRuleBase):
    def _do_apply(self, connection):
        if self.column_types is not None:
            column_types = {col: MAP_TYPES[col_type] for col, col_type in self.column_types.items()}
        else:
            column_types = None
        return pl.read_database(
            self._get_sql_query(),
            connection,
            schema_overrides=column_types,
        )


class WriteSQLTableRule(WriteSQLTableRuleBase):
    def apply(self, data):
        super().apply(data)
        df = self._get_input_df(data)
        import sqlalchemy as sa
        try:
            df.write_database(
                self._get_sql_table(),
                self._get_sql_engine(),
                if_table_exists=self.if_exists
            )
        except sa.exc.SQLAlchemyError as exc:
            raise SQLError(str(exc))
