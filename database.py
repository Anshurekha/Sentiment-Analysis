from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
from databases import Database

# Database URL (using SQLite for simplicity)
DATABASE_URL = "sqlite:///./test.db"  # You can change this to your preferred database

# Create a new database connection
database = Database(DATABASE_URL)
metadata = MetaData()

# SQLAlchemy engine and session configuration
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
