# auto_auction_analyzer/pdf_extractor/debug_tool.py
import pdfplumber
import logging
import sys
import os
from pathlib import Path
from extractor import VehicleDataExtractor

# Konfiguriere ausführliches Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def analyze_pdf_structure(pdf_path):
    """
    Analysiert die Struktur einer PDF-Datei und gibt detaillierte Informationen aus.
    """
    logger.info(f"Analysiere PDF-Struktur: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"PDF enthält {page_count} Seiten")

            # Analysiere jede Seite
            for i, page in enumerate(pdf.pages):
                logger.info(f"Analysiere Seite {i+1}/{page_count}")

                # Text extrahieren
                text = page.extract_text()
                logger.info(f"Extrahierter Text (erste 500 Zeichen): {text[:500]}")

                # Tabellen suchen
                tables = page.extract_tables()
                logger.info(f"Gefundene Tabellen auf Seite {i+1}: {len(tables)}")

                # Erste paar Tabellen analysieren
                for j, table in enumerate(tables[:2]):  # Begrenze auf die ersten 2 Tabellen
                    if table:
                        rows = len(table)
                        cols = len(table[0]) if rows > 0 else 0
                        logger.info(f"Tabelle {j+1}: {rows} Zeilen x {cols} Spalten")

                        # Header-Zeile ausgeben
                        if rows > 0:
                            logger.info(f"Header-Zeile: {table[0]}")

                        # Erste Datenzeile ausgeben
                        if rows > 1:
                            logger.info(f"Erste Datenzeile: {table[1]}")
    except Exception as e:
        logger.error(f"Fehler bei der PDF-Analyse: {str(e)}")

def test_extraction(pdf_path):
    """
    Testet die Extraktion von Fahrzeugdaten aus einer PDF-Datei und
    gibt detaillierte Informationen zum Ergebnis aus.
    """
    logger.info(f"Teste Extraktion für: {pdf_path}")

    extractor = VehicleDataExtractor()
    data = extractor.extract_from_pdf(pdf_path)

    logger.info("Extrahierte Daten:")
    for key, value in data.items():
        logger.info(f"  {key}: {value}")

    return data

if __name__ == "__main__":
    # Prüfe, ob Pfad als Argument übergeben wurde
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Ansonsten frage nach dem Pfad
        pdf_path = input("Bitte geben Sie den Pfad zur PDF-Datei ein: ")

    # Prüfe, ob die Datei existiert
    if not os.path.exists(pdf_path):
        logger.error(f"Die Datei {pdf_path} existiert nicht.")
        sys.exit(1)

    # Analysiere die PDF-Struktur
    analyze_pdf_structure(pdf_path)

    # Teste die Extraktion
    extracted_data = test_extraction(pdf_path)

    # Speichere die extrahierten Daten in einer CSV-Datei
    import pandas as pd
    df = pd.DataFrame([extracted_data])
    output_path = "extrahierte_fahrzeugdaten_debug.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Extrahierte Daten gespeichert in: {output_path}")