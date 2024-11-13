from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite+aiosqlite:///./chat_app.db"  # Path to your SQLite database

# Database engine setup
engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# SessionLocal to interact with the database
SessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for models
Base = declarative_base()
