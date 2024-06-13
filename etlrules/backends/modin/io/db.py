import modin.pandas as pd

from etlrules.backends.common.io.db import (
    ReadSQLQueryRule as ReadSQLQueryRuleBase,
    WriteSQLTableRule as WriteSQLTableRuleBase,
    SQLAlchemyEngines,
)
from etlrules.backends.pandas.types import MAP_TYPES
from etlrules.exceptions import SQLError


class ReadSQLQueryRule(ReadSQLQueryRuleBase):
    def _do_apply(self, connection):
        if self.column_types is not None:
            column_types = {col: MAP_TYPES[col_type] for col, col_type in self.column_types.items()}
        else:
            column_types = None
        return pd.read_sql_query(
            self._get_sql_query(),
            connection,
            dtype=column_types,
        ).convert_dtypes()


class WriteSQLTableRule(WriteSQLTableRuleBase):

    METHOD = 'multi'

    def _do_apply(self, connection, df):
        df.to_sql(
            self._get_sql_table(),
            connection,
            if_exists=self.if_exists,
            index=False,
            method=self.METHOD
        )

    def apply(self, data):
        super().apply(data)
        df = self._get_input_df(data)
        engine = SQLAlchemyEngines.get_engine(self._get_sql_engine())
        import sqlalchemy as sa
        with engine.connect() as connection:
            try:
                self._do_apply(connection, df)
            except sa.exc.SQLAlchemyError as exc:
                raise SQLError(str(exc))
            connection.commit()
