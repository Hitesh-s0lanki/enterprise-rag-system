"""Explicitly initialize the local auth database."""

import argparse

from dotenv import load_dotenv

from .database import initialize_database


def main():
    parser = argparse.ArgumentParser(description="Initialize SQLite auth storage and one-time demo seed")
    parser.add_argument("command", choices=["init"])
    parser.add_argument("--db-path", help="Override AUTH_DB_PATH for this initialization")
    args = parser.parse_args()
    load_dotenv(override=False)
    print(f"Auth database ready: {initialize_database(args.db_path)}")


if __name__ == "__main__":
    main()
