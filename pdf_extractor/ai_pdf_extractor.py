# auto_auction_analyzer/pdf_extractor/ai_pdf_extractor.py
import os
import io
import json
import re
import time
import logging
import tempfile
import pandas as pd
import pdfplumber
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIPdfExtractor:
    """
    KI-basierter PDF-Extraktor, der DeepSeek über eine direkte Prozessverbindung
    zur effizienten Extraktion von Fahrzeugdaten aus PDF-Auktionskatalogen verwendet.
    """

    def __init__(self, model_name="deepseek-r1:14b", show_live_output=True):
        """
        Initialisiert den KI-basierten PDF-Extraktor.

        Args:
            model_name (str): Name des zu verwendenden DeepSeek-Modells
            show_live_output (bool): Ob die KI-Ausgabe live angezeigt werden soll
        """
        self.model_name = model_name
        self.show_live_output = show_live_output

        # Importiere den regelbasierten Extraktor für Fallback
        from pdf_extractor.auction_catalog_extractor import AuctionCatalogExtractor
        self.rule_based_extractor = AuctionCatalogExtractor()

        logger.info(f"KI-PDF-Extraktor initialisiert mit Modell {model_name}, Live-Ausgabe: {show_live_output}")

    def run_ollama_live(self, prompt: str, max_tokens: int = None) -> str:
        """
        Führt das Ollama-Modell über eine direkte Prozessverbindung aus und gibt die Antwort zurück.

        Args:
            prompt: Der Prompt für das Ollama-Modell
            max_tokens: Optional, maximale Anzahl an Tokens in der Antwort

        Returns:
            Die Antwort des Ollama-Modells
        """
        cmd = ["ollama", "run", self.model_name]
        if max_tokens:
            cmd.extend(["--max-tokens", str(max_tokens)])

        logger.info(f"Starte Ollama-Prozess mit Modell {self.model_name}")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )

        process.stdin.write(prompt + "\n")
        process.stdin.close()

        logger.info("Prompt an Ollama gesendet, warte auf Antwort...")

        output = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            # Live-Ausgabe anzeigen, wenn aktiviert
            if self.show_live_output:
                sys.stdout.write(line)
                sys.stdout.flush()
            output.append(line)

        response = ''.join(output)
        logger.info(f"Ollama-Antwort erhalten: {len(response)} Zeichen")
        return response

    def extract_code_from_response(self, response: str) -> str:
        """
        Extrahiert JSON aus einer Ollama-Antwort.

        Args:
            response: Die vollständige Antwort des Ollama-Modells

        Returns:
            Der extrahierte JSON-String
        """
        # Entferne ANSI-Escape-Sequenzen
        cleaned = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', response)

        # Entferne <think>-Tags, falls vorhanden
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

        # Extrahiere JSON aus einem Markdown-Codeblock
        json_pattern = re.compile(r'```(?:json)?\n(.*?)```', re.DOTALL | re.IGNORECASE)
        matches = json_pattern.findall(cleaned)

        if matches:
            return matches[0].strip()

        # Wenn kein Markdown-Block gefunden wurde, suche nach JSON-Array
        json_array_pattern = re.compile(r'\[\s*{.*}\s*\]', re.DOTALL)
        array_matches = json_array_pattern.findall(cleaned)

        if array_matches:
            return array_matches[0].strip()

        # Falls kein JSON gefunden wurde, gib die gesamte bereinigte Antwort zurück
        logger.warning("Kein JSON in der Antwort gefunden. Verwende bereinigte Antwort.")
        return cleaned.strip()

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extrahiert Fahrzeugdaten aus einem Auktionskatalog-PDF mit DeepSeek-KI.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            List[Dict[str, Any]]: Liste von extrahierten Fahrzeugdaten
        """
        logger.info(f"Extrahiere Daten aus: {pdf_path} mit DeepSeek-KI")

        try:
            # 1. Text aus PDF extrahieren
            text_and_tables = self._extract_full_pdf_content(pdf_path)

            # 2. KI-Extraktion durchführen
            vehicles = self._extract_vehicles_with_ollama(text_and_tables)

            if not vehicles:
                logger.warning("KI-Extraktion konnte keine Fahrzeuge erkennen. Versuche regelbasierten Fallback.")
                return self.rule_based_extractor.extract_from_pdf(pdf_path)

            # 3. Nachbearbeitung der extrahierten Daten
            vehicles = self._post_process_vehicles(vehicles)

            logger.info(f"KI-Extraktion erfolgreich: {len(vehicles)} Fahrzeuge gefunden")
            return vehicles

        except Exception as e:
            logger.error(f"Fehler bei der KI-Extraktion: {str(e)}")
            logger.info("Verwende regelbasierten Extraktor als Fallback")

            try:
                vehicles = self.rule_based_extractor.extract_from_pdf(pdf_path)
                logger.info(f"Regelbasierte Extraktion erfolgreich: {len(vehicles)} Fahrzeuge gefunden")
                return vehicles
            except Exception as fallback_e:
                logger.error(f"Auch Fallback-Extraktion fehlgeschlagen: {str(fallback_e)}")
                return []

    def _extract_full_pdf_content(self, pdf_path: str) -> str:
        """
        Extrahiert den vollständigen Text und die Tabellen aus einer PDF-Datei.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            str: Extrahierter Text und Tabellen
        """
        full_content = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDF geöffnet: {len(pdf.pages)} Seiten")

                for i, page in enumerate(pdf.pages):
                    page_number = i + 1
                    full_content.append(f"=== SEITE {page_number} ===")

                    # Extrahiere Text
                    text = page.extract_text() or ""
                    if text:
                        full_content.append(f"--- TEXT ---\n{text}")

                    # Extrahiere Tabellen
                    tables = page.extract_tables()
                    if tables:
                        full_content.append(f"--- TABELLEN ---")
                        for j, table in enumerate(tables):
                            table_text = []
                            for row in table:
                                # Ersetze None durch leere Strings und verbinde mit Pipe
                                row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                                table_text.append(row_text)

                            full_content.append(f"Tabelle {j+1}:\n" + "\n".join(table_text))

            result = "\n\n".join(full_content)
            logger.info(f"Vollständiger PDF-Inhalt extrahiert: {len(result)} Zeichen")
            return result

        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des PDF-Inhalts: {str(e)}")
            raise

    def _extract_vehicles_with_ollama(self, pdf_content: str) -> List[Dict[str, Any]]:
        """
        Extrahiert Fahrzeugdaten mit Ollama aus dem PDF-Inhalt.

        Args:
            pdf_content (str): Der extrahierte PDF-Inhalt

        Returns:
            List[Dict[str, Any]]: Liste der extrahierten Fahrzeuge
        """
        prompt = self._create_extraction_prompt(pdf_content)

        # Führe Ollama aus
        response = self.run_ollama_live(prompt)

        # Extrahiere JSON aus der Antwort
        json_content = self.extract_code_from_response(response)

        try:
            # Versuche, das JSON zu parsen
            data = json.loads(json_content)

            # Stelle sicher, dass es sich um eine Liste handelt
            if not isinstance(data, list):
                if isinstance(data, dict):
                    # Einzelnes Fahrzeug als Liste zurückgeben
                    data = [data]
                else:
                    raise ValueError("Antwort ist weder ein Fahrzeug-Objekt noch eine Liste von Fahrzeugen.")

            logger.info(f"Erfolgreich {len(data)} Fahrzeuge aus JSON extrahiert")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Fehler beim Parsen des JSON: {str(e)}")
            logger.debug(f"JSON-Inhalt: {json_content[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei der Verarbeitung der Ollama-Antwort: {str(e)}")
            return []

    def _create_extraction_prompt(self, pdf_content: str) -> str:
        """
        Erstellt den Prompt für die Fahrzeugdatenextraktion.

        Args:
            pdf_content (str): Der zu analysierende PDF-Inhalt

        Returns:
            str: Der Prompt für Ollama
        """
        return f"""Du bist ein spezialisierter Datenextraktionsassistent für Fahrzeugdaten aus Auktionskatalogen.

Analysiere den folgenden Katalog-Inhalt und extrahiere alle Fahrzeuge mit diesen Informationen:
- Nummer/ID des Fahrzeugs
- Marke und Modell
- PS und kW-Werte
- Baujahr/Erstzulassung (EZ)
- Kilometerstand
- Auktionspreis/Ausruf
- MwSt-Status (Netto, §25a)
- Ausstattungsdetails (optional)

Hier ist der Inhalt des Auktionskatalogs:

{pdf_content}

Gib deine Antwort als JSON-Array im folgenden Format zurück:
```json
[
  {{
    "nummer": "1",
    "marke": "MercedesBenz",
    "modell": "Sprinter Kasten 317 CDI RWD 9G-TRONIC",
    "leistung": 170,
    "leistung_kw": 125,
    "baujahr": 2023,
    "kilometerstand": 44000,
    "auktionspreis": 30000,
    "mwst": "Netto",
    "ausstattung": "Klima, Sitzheizung, DPF, ZV mit FB"
  }},
  ... weitere Fahrzeuge ...
]
```

Bei Datumsangaben wie "Nov 23" oder "Mär 19" interpretiere dies als Monat und Jahr der Erstzulassung.
Wandle Kilometerangaben und Preise in reine Zahlen ohne Einheiten um.
Analysiere besonders Tabellen genau, da sie oft die strukturierten Fahrzeugdaten enthalten.
Sei präzise und vollständig in der Datenextraktion, aber gib nur tatsächlich vorhandene Fahrzeuge zurück.
"""

    def _post_process_vehicles(self, vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Nachbearbeitung der extrahierten Fahrzeuge.

        Args:
            vehicles (List[Dict[str, Any]]): Liste extrahierter Fahrzeuge

        Returns:
            List[Dict[str, Any]]: Nachbearbeitete Fahrzeugliste
        """
        processed_vehicles = []

        # Marken-Mappings für die Normalisierung
        marken_mapping = {
            'Mercedes': 'Mercedes-Benz',
            'MercedesBenz': 'Mercedes-Benz',
            'VW': 'Volkswagen',
            'Citroen': 'Citroën'
        }

        for vehicle in vehicles:
            # Marke normalisieren
            if 'marke' in vehicle and vehicle['marke']:
                marke = str(vehicle['marke']).strip()
                vehicle['marke'] = marken_mapping.get(marke, marke)

            # Mercedes-Benz spezifische Behandlung
            if vehicle.get('marke') == 'Mercedes-Benz' and vehicle.get('modell', '').startswith('Benz'):
                vehicle['modell'] = vehicle['modell'].replace('Benz', '').strip()

            # Numerische Felder konvertieren
            for field in ['baujahr', 'kilometerstand', 'auktionspreis', 'leistung', 'leistung_kw']:
                if field in vehicle and vehicle[field]:
                    try:
                        # Konvertiere Strings zu numerischen Werten, falls notwendig
                        if isinstance(vehicle[field], str):
                            # Entferne Tausendertrennzeichen und konvertiere Kommas
                            value = str(vehicle[field]).replace('.', '')
                            value = value.replace(',', '.')
                            # Extrahiere nur Zahlen und Dezimalpunkte
                            value = ''.join(c for c in value if c.isdigit() or c == '.')

                            if value:  # Nur konvertieren, wenn nicht leer
                                # Zu Float und dann zu Int, wenn möglich
                                vehicle[field] = float(value)
                                if vehicle[field] == int(vehicle[field]):
                                    vehicle[field] = int(vehicle[field])
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"Konnte Feld {field} nicht konvertieren: {str(e)}")

            # Baujahr als 4-stellige Zahl formatieren
            if 'baujahr' in vehicle and vehicle['baujahr']:
                try:
                    year = vehicle['baujahr']
                    if isinstance(year, str):
                        # Extrahiere Jahreszahlen aus Formaten wie "Mär 19" oder "Nov 23"
                        month_match = any(month in year for month in ['Jan', 'Feb', 'Mär', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])
                        if month_match:
                            # Extrahiere nur die Zahlen
                            year_digits = ''.join(c for c in year if c.isdigit())
                            if len(year_digits) == 2:
                                year = year_digits

                        if len(year) == 2:
                            year_int = int(year)
                            if 0 <= year_int <= 99:
                                if year_int < 50:  # Annahme: 00-49 -> 2000-2049
                                    vehicle['baujahr'] = 2000 + year_int
                                else:  # Annahme: 50-99 -> 1950-1999
                                    vehicle['baujahr'] = 1900 + year_int
                        elif len(str(year)) == 4 and str(year).isdigit():
                            vehicle['baujahr'] = int(year)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Konnte Baujahr nicht konvertieren: {str(e)}")

            processed_vehicles.append(vehicle)

        return processed_vehicles

    def process_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Verarbeitet alle PDFs in einem Verzeichnis.

        Args:
            directory_path (str): Pfad zum Verzeichnis mit PDF-Dateien

        Returns:
            pd.DataFrame: DataFrame mit extrahierten Fahrzeugdaten
        """
        results = []
        directory = Path(directory_path)

        for pdf_file in directory.glob("*.pdf"):
            vehicles = self.extract_from_pdf(str(pdf_file))
            # Füge den Dateinamen hinzu
            for vehicle in vehicles:
                vehicle['dateiname'] = pdf_file.name
            results.extend(vehicles)

        if results:
            df = pd.DataFrame(results)
            logger.info(f"Erfolgreich {len(results)} Fahrzeuge aus PDFs in {directory_path} extrahiert.")
            return df
        else:
            logger.warning(f"Keine Fahrzeuge aus PDFs in {directory_path} extrahiert.")
            return pd.DataFrame()

    def process_auction_pdfs(self, uploaded_files) -> pd.DataFrame:
        """
        Verarbeitet hochgeladene PDF-Dateien und extrahiert Fahrzeugdaten.

        Args:
            uploaded_files: Liste von hochgeladenen Streamlit-Dateiobjekten

        Returns:
            pd.DataFrame: DataFrame mit extrahierten Fahrzeugdaten
        """
        all_vehicles = []

        for uploaded_file in uploaded_files:
            # Kopiere die Datei temporär
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            try:
                # Schreibe Datei auf die Festplatte
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extrahiere Fahrzeuge aus der Datei
                vehicles = self.extract_from_pdf(temp_path)

                # Füge Dateinamen hinzu
                for vehicle in vehicles:
                    vehicle['dateiname'] = uploaded_file.name

                all_vehicles.extend(vehicles)

            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten von {uploaded_file.name}: {str(e)}")

            finally:
                # Lösche temporäre Dateien
                try:
                    import gc
                    gc.collect()  # Windows-spezifisch: Handles freigeben
                    time.sleep(0.5)
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Konnte temporären Ordner nicht löschen: {str(e)}")

        # Erstelle DataFrame aus allen extrahierten Fahrzeugen
        if all_vehicles:
            return pd.DataFrame(all_vehicles)
        else:
            return pd.DataFrame()

# Test-Funktion zur Überprüfung der Extraktion
def test_extraction(pdf_path):
    """
    Testfunktion zur Überprüfung der Extraktion.

    Args:
        pdf_path (str): Pfad zur PDF-Datei
    """
    extractor = AIPdfExtractor()
    vehicles = extractor.extract_from_pdf(pdf_path)

    print(f"Extraktion abgeschlossen. {len(vehicles)} Fahrzeuge gefunden.")

    for i, vehicle in enumerate(vehicles):
        print(f"\nFahrzeug {i+1}:")
        for key, value in vehicle.items():
            print(f"  {key}: {value}")

# Wenn direkt ausgeführt, starte interaktive Extraktion
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        test_extraction(pdf_path)
    else:
        print("Bitte geben Sie den Pfad zur PDF-Datei als Argument an.")