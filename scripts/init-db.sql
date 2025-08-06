-- Initialize databases
-- Note: Main database 'kang_db' is created automatically via POSTGRES_DB environment variable

-- Create test database
CREATE DATABASE test_kang_db;
GRANT ALL PRIVILEGES ON DATABASE test_kang_db TO postgres;

-- Ensure proper encoding and extensions
\c kang_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector extension for RAG

\c test_kang_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector extension for RAG
