# History

## 0.3.2 (2024-01-08)

* Add option to skip rows from the top of a csv file for the csv read rule
* Add support for reading/writing compressed csv files for the polars backend
* Support reading csv files via http and https
* Read csv files in one block in dask

## 0.3.1 (2023-12-22)

* Remove polars-business dependency and implement vectorized datetime operations for weekdays offsets
* Add tests to run the examples to make sure they don't get broken by future changes
* Optimize pandas business_offset and date_offset using values from a different columns to be vectorized operations
* Add a perf logger which will log a warning when using operations which are not vectorized

## 0.3.0 (2023-12-11)

* Add support for dask backend
* Ability to deserialize custom rules (ie not part of the etlrules package) to support users implementing their own rules

## 0.2.3 (2023-11-28)

* Fix to apply substitution in the WriteSQLTableRule sql_engine parameter
* Apply substitution in the Read/Write rules for csv and parquet files for the file_name and file_dir parameters
* Add a cli runner which allows users to run a yml file and parameterize with cli args the plan context
* Add the csv2db plan/yml example
* Add the db2csv plan/yml example
* Remove poetry

## 0.2.2 (2023-11-26)

* Support environment variables substitution in the sql_engine string for SQL rules
* Add support for the Boolean type
* Introduce the concept of a plan context, consisting of a key-value mapping of string to int/float/str/bool values
  which will act as the args into the plan. They can be used in expressions when adding new columns, ifthenelse and
  filter rules.
* Add env and context substitution feature to sql queries

## 0.2.1 (2023-11-20)

* Add support for unsigned int types
* Add AddRowNumbersRule, a rule to add row numbers
* Add ExplodeValuesRule which explode lists of values into individual scalar values as additional rows
* Add support for reading/writing to/from DBs via sqlalchemy under etlrules.backends.common.io.db

## 0.2.0 (2023-11-14)

* Add support for polars as a backend

## 0.1.1 (2023-10-29)

* Fix bug in RuleEngine validate affecting pipeline mode
* Export the main classes from etlrules __init__
* More typing annotations
* More documentation

## 0.1.0 (2023-10-28)

* First release on PyPI.
* Rule engine supporting running pipeline mode and graph mode plans
* Support for an initial set of etl rules, pandas backend only
