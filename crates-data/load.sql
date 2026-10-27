-- Load all crates.io dump CSVs into DuckDB tables.
-- run from inside the extracted dump directory:  cd crates-data/<dump> && duckdb crates.duckdb < ../load.sql

-- sample_size=-1 -> scan the whole file for type inference (avoids mis-typed
-- columns on the large files). union_by_name is harmless here but robust.

CREATE OR REPLACE TABLE categories           AS FROM read_csv('data/categories.csv',           sample_size=-1);
CREATE OR REPLACE TABLE crate_downloads      AS FROM read_csv('data/crate_downloads.csv',      sample_size=-1);
CREATE OR REPLACE TABLE crate_owners         AS FROM read_csv('data/crate_owners.csv',         sample_size=-1);
CREATE OR REPLACE TABLE crates               AS FROM read_csv('data/crates.csv',               sample_size=-1, max_line_size=20000000);
CREATE OR REPLACE TABLE crates_categories    AS FROM read_csv('data/crates_categories.csv',    sample_size=-1);
CREATE OR REPLACE TABLE crates_keywords      AS FROM read_csv('data/crates_keywords.csv',      sample_size=-1);
CREATE OR REPLACE TABLE default_versions     AS FROM read_csv('data/default_versions.csv',     sample_size=-1);
CREATE OR REPLACE TABLE deleted_crates       AS FROM read_csv('data/deleted_crates.csv',       sample_size=-1);
CREATE OR REPLACE TABLE dependencies         AS FROM read_csv('data/dependencies.csv',         sample_size=-1, max_line_size=20000000);
CREATE OR REPLACE TABLE keywords             AS FROM read_csv('data/keywords.csv',             sample_size=-1);
CREATE OR REPLACE TABLE metadata             AS FROM read_csv('data/metadata.csv',             sample_size=-1);
CREATE OR REPLACE TABLE oauth_github         AS FROM read_csv('data/oauth_github.csv',         sample_size=-1);
CREATE OR REPLACE TABLE reserved_crate_names AS FROM read_csv('data/reserved_crate_names.csv', sample_size=-1);
CREATE OR REPLACE TABLE reserved_usernames   AS FROM read_csv('data/reserved_usernames.csv',   sample_size=-1);
CREATE OR REPLACE TABLE teams                AS FROM read_csv('data/teams.csv',                sample_size=-1);
CREATE OR REPLACE TABLE users                AS FROM read_csv('data/users.csv',                sample_size=-1);
CREATE OR REPLACE TABLE version_downloads    AS FROM read_csv('data/version_downloads.csv',    sample_size=-1);
CREATE OR REPLACE TABLE versions             AS FROM read_csv('data/versions.csv',             sample_size=-1, max_line_size=20000000);

-- Sanity check: row counts per table.
SELECT 'crates' AS tbl, count(*) FROM crates
UNION ALL SELECT 'versions',          count(*) FROM versions
UNION ALL SELECT 'dependencies',      count(*) FROM dependencies
UNION ALL SELECT 'version_downloads', count(*) FROM version_downloads
UNION ALL SELECT 'crate_downloads',   count(*) FROM crate_downloads
UNION ALL SELECT 'crate_owners',      count(*) FROM crate_owners
UNION ALL SELECT 'users',             count(*) FROM users;
