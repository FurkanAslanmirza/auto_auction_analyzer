import os
from pathlib import Path

# Basis-Verzeichnisse
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.environ.get("AAA_DATA_DIR", BASE_DIR / "data")
CACHE_DIR = os.environ.get("AAA_CACHE_DIR", BASE_DIR / "cache")
MODEL_DIR = os.environ.get("AAA_MODEL_DIR", BASE_DIR / "models")
OUTPUT_DIR = os.environ.get("AAA_OUTPUT_DIR", BASE_DIR / "output")

# Erstellen Sie benötigte Verzeichnisse
for directory in [DATA_DIR, CACHE_DIR, MODEL_DIR, OUTPUT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Datenbank-Konfiguration
DB_TYPE = os.environ.get("AAA_DB_TYPE", "sqlite")
DB_NAME = os.environ.get("AAA_DB_NAME", "auto_auction.db")
DB_HOST = os.environ.get("AAA_DB_HOST", "localhost")
DB_PORT = os.environ.get("AAA_DB_PORT", "5432")
DB_USER = os.environ.get("AAA_DB_USER", "postgres")
DB_PASS = os.environ.get("AAA_DB_PASS", "postgres")

# Verbindungsstring erstellen
if DB_TYPE == "sqlite":
    DB_CONNECTION = f"sqlite:///{DATA_DIR}/{DB_NAME}"
else:
    DB_CONNECTION = f"{DB_TYPE}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Mobile.de API-Konfiguration
MOBILE_API_KEY = os.environ.get("AAA_MOBILE_API_KEY", "")
MOBILE_API_SECRET = os.environ.get("AAA_MOBILE_API_SECRET", "")

# KI-Konfiguration
AI_MODEL = os.environ.get("AAA_AI_MODEL", "deepseek-r1:32b")
AI_FALLBACK_MODEL = os.environ.get("AAA_AI_FALLBACK_MODEL", "llama3:8b")

# Logging-Konfiguration
LOG_LEVEL = os.environ.get("AAA_LOG_LEVEL", "INFO")
LOG_FILE = os.environ.get("AAA_LOG_FILE", str(OUTPUT_DIR / "auto_auction.log"))

# Profitabilitäts-Parameter
MIN_PROFIT_MARGIN = float(os.environ.get("AAA_MIN_PROFIT_MARGIN", "15.0"))
MIN_PROFIT_AMOUNT = float(os.environ.get("AAA_MIN_PROFIT_AMOUNT", "2000.0"))