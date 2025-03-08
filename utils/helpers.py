# auto_auction_analyzer/utils/helpers.py
import os
import logging
import time
import re
import json
import pandas as pd
from pathlib import Path
import subprocess
import platform

logger = logging.getLogger(__name__)

class DataValidator:
    """Hilfsfunktionen zur Validierung von Daten"""

    @staticmethod
    def validate_vehicle_data(df):
        """
        Überprüft, ob die Fahrzeugdaten vollständig und gültig sind.

        Args:
            df (pd.DataFrame): DataFrame mit Fahrzeugdaten

        Returns:
            tuple: (bool, list) - Ist gültig, Liste der Probleme
        """
        problems = []

        # Prüfe auf leeren DataFrame
        if df is None or df.empty:
            problems.append("Keine Fahrzeugdaten vorhanden.")
            return False, problems

        # Erforderliche Spalten
        required_columns = ['marke', 'modell', 'baujahr', 'kilometerstand', 'auktionspreis']

        # Prüfe, ob alle erforderlichen Spalten vorhanden sind
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            problems.append(f"Fehlende Spalten: {', '.join(missing_columns)}")

        # Prüfe auf fehlende Werte in erforderlichen Spalten
        if not missing_columns:
            for col in required_columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    problems.append(f"{missing_count} fehlende Werte in Spalte '{col}'")

        # Prüfe auf ungültige Werte
        if 'baujahr' in df.columns:
            invalid_years = df[~df['baujahr'].between(1900, 2030) & df['baujahr'].notna()].shape[0]
            if invalid_years > 0:
                problems.append(f"{invalid_years} Fahrzeuge mit ungültigem Baujahr")

        if 'kilometerstand' in df.columns:
            invalid_km = df[~df['kilometerstand'].between(0, 1000000) & df['kilometerstand'].notna()].shape[0]
            if invalid_km > 0:
                problems.append(f"{invalid_km} Fahrzeuge mit ungültigem Kilometerstand")

        if 'auktionspreis' in df.columns:
            invalid_price = df[~df['auktionspreis'].between(0, 10000000) & df['auktionspreis'].notna()].shape[0]
            if invalid_price > 0:
                problems.append(f"{invalid_price} Fahrzeuge mit ungültigem Auktionspreis")

        # Gültig, wenn keine Probleme gefunden wurden
        is_valid = len(problems) == 0

        return is_valid, problems

    @staticmethod
    def validate_market_data(df):
        """
        Überprüft, ob die Marktdaten vollständig und gültig sind.

        Args:
            df (pd.DataFrame): DataFrame mit Marktdaten

        Returns:
            tuple: (bool, list) - Ist gültig, Liste der Probleme
        """
        problems = []

        # Prüfe auf leeren DataFrame
        if df is None or df.empty:
            problems.append("Keine Marktdaten vorhanden.")
            return False, problems

        # Erforderliche Spalten
        required_columns = ['title', 'marktpreis', 'vehicle_id']

        # Prüfe, ob alle erforderlichen Spalten vorhanden sind
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            problems.append(f"Fehlende Spalten: {', '.join(missing_columns)}")

        # Prüfe auf fehlende Werte in erforderlichen Spalten
        if not missing_columns:
            for col in required_columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    problems.append(f"{missing_count} fehlende Werte in Spalte '{col}'")

        # Prüfe auf ungültige Preise
        if 'marktpreis' in df.columns:
            invalid_prices = df[~df['marktpreis'].between(0, 10000000) & df['marktpreis'].notna()].shape[0]
            if invalid_prices > 0:
                problems.append(f"{invalid_prices} Angebote mit ungültigem Marktpreis")

        # Gültig, wenn keine Probleme gefunden wurden
        is_valid = len(problems) == 0

        return is_valid, problems


class SystemChecker:
    """Hilfsfunktionen zur Überprüfung des Systems"""

    @staticmethod
    def check_python_version():
        """
        Überprüft, ob die benötigte Python-Version installiert ist.

        Returns:
            bool: True, wenn Python 3.10 oder höher installiert ist, sonst False
        """
        import sys
        major, minor, *_ = sys.version_info
        required_major, required_minor = 3, 10

        if major > required_major or (major == required_major and minor >= required_minor):
            logger.info(f"Python-Version {major}.{minor} ist kompatibel.")
            return True
        else:
            logger.warning(f"Python-Version {major}.{minor} ist nicht kompatibel. Benötigt wird mindestens Python {required_major}.{required_minor}.")
            return False

    @staticmethod
    def check_dependencies():
        """
        Überprüft, ob alle benötigten Abhängigkeiten installiert sind.

        Returns:
            tuple: (bool, list) - Alle Abhängigkeiten installiert, Liste fehlender Abhängigkeiten
        """
        dependencies = [
            "pdfplumber",
            "pymupdf",
            "selenium",
            "pandas",
            "matplotlib",
            "streamlit",
            "requests",
            "beautifulsoup4",
            "webdriver_manager",
        ]

        missing = []

        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            logger.warning(f"Fehlende Abhängigkeiten: {', '.join(missing)}")
            return False, missing
        else:
            logger.info("Alle benötigten Abhängigkeiten sind installiert.")
            return True, []

    @staticmethod
    def check_ollama():
        """
        Überprüft, ob Ollama installiert ist und verfügbar ist.

        Returns:
            bool: True, wenn Ollama installiert ist, sonst False
        """
        try:
            result = subprocess.run(
                ["ollama", "version"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Ollama installiert: {version}")
                return True
            else:
                logger.warning("Ollama ist nicht installiert oder nicht im Pfad.")
                return False

        except Exception as e:
            logger.warning(f"Fehler bei der Überprüfung von Ollama: {str(e)}")
            return False

    @staticmethod
    def check_hardware_requirements():
        """
        Überprüft die Hardware-Anforderungen für DeepSeek-R1.

        Returns:
            dict: Informationen zu den Systemanforderungen
        """
        requirements = {
            'ram_gb': 0,
            'ram_sufficient': False,
            'required_ram_gb': 32,
            'gpu_info': [],
            'gpu_sufficient': False,
            'required_gpu_vram_gb': 16,
            'overall_sufficient': False
        }

        try:
            import psutil

            # RAM überprüfen
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            requirements['ram_gb'] = ram_gb
            requirements['ram_sufficient'] = ram_gb >= requirements['required_ram_gb']

            # GPU überprüfen
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    for gpu in gpus:
                        gpu_info = {
                            'name': gpu.name,
                            'memory_total_gb': round(gpu.memoryTotal / 1024, 2),
                            'sufficient': gpu.memoryTotal / 1024 >= requirements['required_gpu_vram_gb']
                        }
                        requirements['gpu_info'].append(gpu_info)

                    requirements['gpu_sufficient'] = any(info['sufficient'] for info in requirements['gpu_info'])

            except (ImportError, Exception) as e:
                logger.warning(f"Konnte GPU-Informationen nicht abrufen: {str(e)}")
                requirements['gpu_info'] = [{'name': 'Keine GPU erkannt', 'memory_total_gb': 0, 'sufficient': False}]
                requirements['gpu_sufficient'] = False

            # Gesamtbewertung
            requirements['overall_sufficient'] = requirements['ram_sufficient'] and requirements['gpu_sufficient']

        except Exception as e:
            logger.warning(f"Fehler bei der Überprüfung der Hardware-Anforderungen: {str(e)}")

        return requirements


class FileHelper:
    """Hilfsfunktionen für Dateien"""

    @staticmethod
    def validate_pdf(file_path):
        """
        Überprüft, ob eine Datei eine gültige PDF-Datei ist.

        Args:
            file_path (str): Pfad zur PDF-Datei

        Returns:
            bool: True, wenn es sich um eine gültige PDF-Datei handelt, sonst False
        """
        try:
            import pdfplumber

            # Versuche, die PDF-Datei zu öffnen
            with pdfplumber.open(file_path) as pdf:
                # Prüfe, ob mindestens eine Seite vorhanden ist
                if len(pdf.pages) > 0:
                    return True
                else:
                    logger.warning(f"PDF-Datei {file_path} enthält keine Seiten.")
                    return False

        except Exception as e:
            logger.warning(f"Ungültige PDF-Datei {file_path}: {str(e)}")
            return False

    @staticmethod
    def get_pdf_files(directory):
        """
        Gibt eine Liste aller PDF-Dateien in einem Verzeichnis zurück.

        Args:
            directory (str): Pfad zum Verzeichnis

        Returns:
            list: Liste der Pfade zu PDF-Dateien
        """
        pdf_files = []

        try:
            directory_path = Path(directory)

            if not directory_path.exists():
                logger.warning(f"Verzeichnis {directory} existiert nicht.")
                return pdf_files

            for file_path in directory_path.glob("*.pdf"):
                if FileHelper.validate_pdf(file_path):
                    pdf_files.append(file_path)

            logger.info(f"{len(pdf_files)} gültige PDF-Dateien in {directory} gefunden.")

        except Exception as e:
            logger.error(f"Fehler beim Suchen nach PDF-Dateien: {str(e)}")

        return pdf_files

    @staticmethod
    def save_as_json(data, file_path):
        """
        Speichert Daten als JSON-Datei.

        Args:
            data: Zu speichernde Daten
            file_path (str): Pfad zur Ausgabedatei

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Daten erfolgreich als JSON gespeichert: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern als JSON: {str(e)}")
            return False

    @staticmethod
    def load_from_json(file_path):
        """
        Lädt Daten aus einer JSON-Datei.

        Args:
            file_path (str): Pfad zur JSON-Datei

        Returns:
            dict: Geladene Daten oder None bei Fehler
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Daten erfolgreich aus JSON geladen: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Fehler beim Laden aus JSON: {str(e)}")
            return None


class LegalHelper:
    """Hilfsfunktionen für rechtliche Aspekte"""

    @staticmethod
    def check_robots_txt(domain="mobile.de"):
        """
        Überprüft die robots.txt einer Domain.

        Args:
            domain (str): Domain-Name

        Returns:
            dict: Informationen zu den Regeln der robots.txt
        """
        import requests
        from urllib.parse import urlparse

        try:
            url = f"https://{domain}/robots.txt"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                content = response.text

                # Einfache Analyse der robots.txt
                user_agents = re.findall(r"User-agent: (\*|[a-zA-Z0-9_-]+)", content)
                disallows = re.findall(r"Disallow: ([^\r\n]+)", content)
                allows = re.findall(r"Allow: ([^\r\n]+)", content)
                crawl_delay = re.search(r"Crawl-delay: (\d+)", content)

                result = {
                    'user_agents': user_agents,
                    'disallows': disallows,
                    'allows': allows,
                    'crawl_delay': int(crawl_delay.group(1)) if crawl_delay else None,
                    'content': content
                }

                logger.info(f"robots.txt für {domain} erfolgreich abgerufen.")
                return result
            else:
                logger.warning(f"Konnte robots.txt für {domain} nicht abrufen: Status {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Fehler beim Abrufen der robots.txt für {domain}: {str(e)}")
            return None

    @staticmethod
    def get_legal_disclaimer():
        """
        Gibt rechtliche Hinweise zurück.

        Returns:
            str: Rechtliche Hinweise
        """
        return """
        Rechtliche Hinweise zum Web-Scraping:
        
        1. Das Web-Scraping von öffentlich zugänglichen Daten kann in Deutschland grundsätzlich legal sein,
           unterliegt jedoch rechtlichen Einschränkungen.
        
        2. Diese Software beachtet die robots.txt der gescrapten Webseiten und verwendet angemessene
           Wartezeiten zwischen Anfragen, um die Server nicht zu überlasten.
        
        3. Die gesammelten Daten dürfen nur für private, nicht-kommerzielle Zwecke verwendet werden.
        
        4. Das Scraping von persönlichen Daten ist untersagt und wird von dieser Software nicht unterstützt.
        
        5. Der Nutzer dieser Software ist für die rechtmäßige Verwendung verantwortlich und muss
           die Nutzungsbedingungen der gescrapten Webseiten einhalten.
        
        DSGVO-Hinweise:
        
        1. Diese Software sammelt und verarbeitet Fahrzeugdaten, die in der Regel keine personenbezogenen
           Daten nach der DSGVO darstellen.
        
        2. Alle Daten werden ausschließlich lokal auf dem Gerät des Nutzers gespeichert und verarbeitet.
        
        3. Es findet keine Übermittlung von Daten an Dritte statt.
        
        4. Die Integration des KI-Modells DeepSeek-R1 erfolgt lokal, ohne dass Daten an externe Server
           übermittelt werden.
        
        Haftungsausschluss:
        
        Der Autor dieser Software übernimmt keine Haftung für eventuelle rechtliche Konsequenzen,
        die aus der Nutzung dieser Software entstehen können. Die Nutzer sind selbst für die rechtmäßige
        Verwendung verantwortlich.
        """

def create_project_structure():
    """
    Erstellt die Projektstruktur für die Auto-Auktions-Analyse-Anwendung.

    Returns:
        bool: True bei Erfolg, False bei Fehler
    """
    try:
        # Verzeichnisstruktur
        directories = [
            "auto_auction_analyzer",
            "auto_auction_analyzer/pdf_extractor",
            "auto_auction_analyzer/scraper",
            "auto_auction_analyzer/data_analysis",
            "auto_auction_analyzer/ai_integration",
            "auto_auction_analyzer/dashboard",
            "auto_auction_analyzer/utils",
            "pdf_files",
            "output"
        ]

        # Erstelle Verzeichnisse
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Erstelle __init__.py Dateien
        for directory in directories:
            if directory.startswith("auto_auction_analyzer"):
                init_file = os.path.join(directory, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, 'w') as f:
                        f.write("# Automatisch generierte __init__.py Datei\n")

        # Erstelle setup.py
        setup_py = """
from setuptools import setup, find_packages

setup(
    name="auto_auction_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pdfplumber",
        "pymupdf",
        "selenium",
        "pandas",
        "matplotlib",
        "streamlit",
        "requests",
        "beautifulsoup4",
        "webdriver_manager",
        "seaborn",
        "numpy",
        "plotly",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Tool for extracting and analyzing vehicle auction data",
)
"""

        with open("setup.py", 'w') as f:
            f.write(setup_py)

        # Erstelle README.md
        readme_md = """
# Auto Auction Analyzer

Eine Anwendung zur Extraktion von Fahrzeugdaten aus Auktions-PDFs, Vergleich mit mobilde.de-Marktpreisen
und KI-basierter Entscheidungsfindung für den Weiterverkauf.

## Installation

```bash
# Clone das Repository
git clone https://github.com/yourusername/auto_auction_analyzer.git
cd auto_auction_analyzer

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate  # Unter Linux/macOS
# oder
venv\\Scripts\\activate  # Unter Windows

# Abhängigkeiten installieren
pip install -e .
```

## Verwendung

### Dashboard-Modus

```bash
python -m auto_auction_analyzer.main --dashboard
```

### Analyse-Modus

```bash
python -m auto_auction_analyzer.main --analyze --pdf_dir=/pfad/zu/pdfs --output_dir=/pfad/für/ergebnisse
```

Weitere Optionen finden Sie mit:

```bash
python -m auto_auction_analyzer.main --help
```

## Rechtliche Hinweise

Bitte beachten Sie die rechtlichen Hinweise in der Dokumentation zu Web-Scraping und DSGVO.
"""

        with open("README.md", 'w') as f:
            f.write(readme_md)

        return True

    except Exception as e:
        logger.error(f"Fehler beim Erstellen der Projektstruktur: {str(e)}")
        return False

# Beispielverwendung
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Überprüfe Systemanforderungen
    system_ok = SystemChecker.check_python_version()
    deps_ok, missing_deps = SystemChecker.check_dependencies()

    if not system_ok:
        print("Warnung: Python-Version ist nicht kompatibel.")

    if not deps_ok:
        print(f"Warnung: Fehlende Abhängigkeiten: {', '.join(missing_deps)}")
        print("Installieren Sie die fehlenden Abhängigkeiten mit:")
        print(f"pip install {' '.join(missing_deps)}")

    # Überprüfe robots.txt
    robots_info = LegalHelper.check_robots_txt()
    if robots_info:
        print("\nInformationen zur robots.txt von mobile.de:")
        if robots_info.get('crawl_delay'):
            print(f"Crawl-Delay: {robots_info['crawl_delay']} Sekunden")

        print(f"Anzahl Disallow-Regeln: {len(robots_info['disallows'])}")
        print(f"Anzahl Allow-Regeln: {len(robots_info['allows'])}")

    # Erstelle Projektstruktur
    if create_project_structure():
        print("\nProjektstruktur erfolgreich erstellt.")
    else:
        print("\nFehler beim Erstellen der Projektstruktur.")

    # Zeige rechtliche Hinweise
    print("\nRechtliche Hinweise:")
    print(LegalHelper.get_legal_disclaimer())