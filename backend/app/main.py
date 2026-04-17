from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.database import create_tables
from app.routes import auth
from app.routes import aa
from app.routes import penny
from app.routes import goals

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    from app.core.db_config import init_penny_tables
    init_penny_tables()
    print(f"✅ FinSight API started | env={settings.APP_ENV}")
    yield
    print("FinSight API shutting down")


app = FastAPI(
    title="FinSight API",
    version="1.0.0",
    description="Backend for FinSight — RBI AA powered finance aggregator",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(aa.router,   prefix="/api")
app.include_router(penny.router, prefix="/api")
app.include_router(goals.router, prefix="/api")


@app.get("/")
async def root():
    return {"service": "FinSight API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}