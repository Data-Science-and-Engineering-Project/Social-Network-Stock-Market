"""Step 1: Create the 13FGNN database."""

import os
import psycopg2
from dotenv import load_dotenv
from ETL.logger.logger import ETLLogger

load_dotenv()


def run():
    """Create the 13FGNN database if it doesn't exist."""
    logger = ETLLogger(name="Step1_CreateDatabase", console_output=True)
    logger.info("=" * 80)
    logger.info("STEP 1: CREATE DATABASE 13FGNN")
    logger.info("=" * 80)

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", 5432))
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    conn = None
    try:
        # Connect to postgres maintenance database to create new database
        conn = psycopg2.connect(
            host=host,
            port=port,
            database="postgres",
            user=user,
            password=password,
            connect_timeout=5,
        )
        conn.autocommit = True
        cur = conn.cursor()

        try:
            cur.execute('CREATE DATABASE "13FGNN"')
            logger.info('Database "13FGNN" created successfully')
        except psycopg2.errors.DuplicateDatabase:
            logger.info('Database "13FGNN" already exists, skipping creation')

        cur.close()
        logger.info("Step 1 completed")

    except psycopg2.Error as e:
        logger.error(f"Database creation failed: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()
        logger.close()
