-- Run by the Postgres container the first time its volume is created.
-- Tests use this separate database so they never touch your development data.
CREATE DATABASE app_test;
