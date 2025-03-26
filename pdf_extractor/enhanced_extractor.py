# auto_auction_analyzer/pdf_extractor/enhanced_extractor.py
import pdfplumber
import re
import pandas as pd
import numpy as np
import logging
import time
import json
import os
import tempfile
import io
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPdfExtractor:
    """
    Verbesserte Klasse zur Extraktion von Fahrzeugdaten aus Auktionskatalogen.
    Kombiniert regelbasierte und KI-gestützte Methoden mit robuster Fehlerbehandlung.
    """

    def __init__(self, templates_dir=None, ollama_model="deepseek-r1:14b"):
        """
        Initialisiert den verbesserten PDF-Extraktor.

        Args:
            templates_dir (str, optional): Verzeichnis mit Extraktions-Templates
            ollama_model (str): Name des zu verwendenden Ollama-Modells
        """
        self.templates_dir = templates_dir or os.path.join(os.path.dirname(__file__), "templates")
        self.templates = self._load_templates()
        self.ollama_model = ollama_model

        # Importiere den regelbasierten Extraktor für Fallback
        try:
            from auction_catalog_extractor import AuctionCatalogExtractor
            self.rule_based_extractor = AuctionCatalogExtractor()
        except ImportError:
            logger.warning("Regelbasierter Extraktor nicht verfügbar. Wird erstellt.")
            # Erstelle eine Basisimplementierung
            from pdf_extractor.extractor import VehicleDataExtractor
            self.rule_based_extractor = VehicleDataExtractor()

    def _load_templates(self) -> dict:
        """
        Lädt alle Extraktions-Templates aus dem Templates-Verzeichnis.

        Returns:
            dict: Dictionary mit geladenen Templates
        """
        templates = {}

        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates-Verzeichnis {self.templates_dir} existiert nicht.")
            return templates

        for file_path in Path(self.templates_dir).glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    template = json.load(f)
                    template_name = template.get('name')
                    if template_name:
                        templates[template_name] = template
                        logger.debug(f"Template geladen: {template_name}")
            except Exception as e:
                logger.error(f"Fehler beim Laden des Templates {file_path}: {str(e)}")

        logger.info(f"{len(templates)} Extraktions-Templates geladen.")
        return templates

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extrahiert Fahrzeugdaten aus einem PDF mit einer Kombination aus
        regelbasierter Extraktion und KI-Unterstützung.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            List[Dict[str, Any]]: Liste von extrahierten Fahrzeugdaten
        """
        logger.info(f"Extrahiere Daten aus: {pdf_path}")

        try:
            # Schritt 1: PDF validieren
            if not self._validate_pdf(pdf_path):
                logger.error(f"Ungültige PDF-Datei: {pdf_path}")
                return []

            # Schritt 2: Versuche zunächst die regelbasierte Extraktion
            rule_based_results = self._extract_rule_based(pdf_path)

            # Wenn die regelbasierte Extraktion erfolgreich war, verwende diese
            if rule_based_results and len(rule_based_results) > 0:
                # Validiere die Ergebnisse
                valid_results = [r for r in rule_based_results if self._validate_vehicle_data(r)]
                logger.info(f"Regelbasierte Extraktion lieferte {len(valid_results)} valide Fahrzeuge.")

                # Wenn mehr als 70% der Ergebnisse valide sind, betrachte es als Erfolg
                if len(valid_results) > 0 and len(valid_results) >= 0.7 * len(rule_based_results):
                    return valid_results

            # Schritt 3: Wenn die regelbasierte Extraktion nicht funktioniert hat, versuche die KI-basierte Extraktion
            logger.info("Verwende KI-basierte Extraktion als Fallback.")
            pdf_content = self._extract_pdf_content_enhanced(pdf_path)

            # Erste KI-Extraktion mit Text und Tabellen
            ai_results = self._extract_with_ollama(pdf_content)

            # Validiere die Ergebnisse
            valid_ai_results = [r for r in ai_results if self._validate_vehicle_data(r)]
            logger.info(f"KI-basierte Extraktion lieferte {len(valid_ai_results)} valide Fahrzeuge.")

            if valid_ai_results:
                return valid_ai_results

            # Schritt 4: Wenn die KI-Extraktion keine validen Ergebnisse liefert,
            # versuche es mit einer robusteren OCR-basierten Methode
            if not valid_ai_results:
                logger.info("Versuche OCR-basierte Extraktion...")
                ocr_content = self._extract_with_ocr(pdf_path)
                return self._extract_with_ollama(ocr_content, more_detailed=True)

            return []

        except Exception as e:
            logger.error(f"Fehler bei der Extraktion: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _validate_pdf(self, pdf_path: str) -> bool:
        """
        Validiert, ob eine PDF-Datei geöffnet und verarbeitet werden kann.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            bool: True wenn die PDF gültig ist, sonst False
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                if page_count == 0:
                    logger.warning(f"PDF hat keine Seiten: {pdf_path}")
                    return False

                # Prüfe, ob zumindest die erste Seite Text enthält
                first_page_text = pdf.pages[0].extract_text()
                if not first_page_text or len(first_page_text) < 10:
                    logger.warning(f"Erste PDF-Seite enthält keinen Text: {pdf_path}")
                    # Dies könnte immer noch ein gültiges PDF sein (z.B. mit Bildern),
                    # daher Rückgabe True

            return True
        except Exception as e:
            logger.error(f"Fehler beim Öffnen der PDF: {str(e)}")
            return False

    def _extract_rule_based(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Führt eine regelbasierte Extraktion durch.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            List[Dict[str, Any]]: Liste von extrahierten Fahrzeugdaten
        """
        try:
            # Verwende den regelbasierten Extraktor
            vehicles = self.rule_based_extractor.extract_from_pdf(pdf_path)

            # Konvertiere in ein einheitliches Format
            standardized_vehicles = []
            for vehicle in vehicles:
                # Konvertiere Typen und standardisiere Schlüssel
                standardized = self._standardize_vehicle_data(vehicle)
                standardized_vehicles.append(standardized)

            return standardized_vehicles

        except Exception as e:
            logger.error(f"Fehler bei der regelbasierten Extraktion: {str(e)}")
            return []

    def _extract_pdf_content_enhanced(self, pdf_path: str) -> str:
        """
        Extrahiert den Text- und Tabelleninhalt aus einem PDF-Dokument
        mit erweiterter Genauigkeit.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            str: Extrahierter PDF-Inhalt
        """
        full_content = []

        try:
            # Methode 1: Verwende pdfplumber für Text und Tabellen
            with pdfplumber.open(pdf_path) as pdf:
                logger.debug(f"PDF geöffnet mit pdfplumber: {len(pdf.pages)} Seiten")

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
                            if not table:
                                continue

                            table_text = []
                            for row in table:
                                if not row:
                                    continue
                                # Ersetze None durch leere Strings und verbinde mit Pipe
                                row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                                table_text.append(row_text)

                            if table_text:
                                full_content.append(f"Tabelle {j+1}:\n" + "\n".join(table_text))

            # Methode 2: Verwende PyMuPDF (fitz) als Backup für zusätzliche Strukturinformationen
            try:
                with fitz.open(pdf_path) as doc:
                    full_content.append("\n=== ZUSÄTZLICHE STRUKTURDATEN ===\n")

                    for i, page in enumerate(doc):
                        page_number = i + 1

                        # Extrahiere Text mit Erhaltung von Layout-Informationen
                        text = page.get_text("dict")

                        # Extrahiere Blocks, die oft besser strukturiert sind
                        if "blocks" in text:
                            block_texts = []
                            for b, block in enumerate(text["blocks"]):
                                if "lines" in block:
                                    lines_text = []
                                    for line in block["lines"]:
                                        if "spans" in line:
                                            spans_text = []
                                            for span in line["spans"]:
                                                if "text" in span and span["text"].strip():
                                                    spans_text.append(span["text"])
                                            if spans_text:
                                                lines_text.append(" ".join(spans_text))
                                    if lines_text:
                                        block_texts.append("\n".join(lines_text))

                            if block_texts:
                                full_content.append(f"--- STRUKTURIERTER TEXT SEITE {page_number} ---\n")
                                full_content.append("\n\n".join(block_texts))
            except Exception as e:
                logger.warning(f"PyMuPDF Extraktion fehlgeschlagen: {str(e)}")

            result = "\n\n".join(full_content)
            logger.info(f"Vollständiger PDF-Inhalt extrahiert: {len(result)} Zeichen")
            return result

        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des PDF-Inhalts: {str(e)}")
            return f"Fehler bei der Extraktion: {str(e)}"

    def _extract_with_ocr(self, pdf_path: str) -> str:
        """
        Extrahiert Text aus einem PDF mittels OCR für bessere Erkennung.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            str: Mit OCR extrahierter Text
        """
        try:
            # Prüfe, ob pytesseract vorhanden ist
            import pytesseract
            from pdf2image import convert_from_path

            content = ["=== OCR-EXTRAHIERTER TEXT ==="]

            # Konvertiere PDF-Seiten zu Bildern
            images = convert_from_path(pdf_path)

            for i, image in enumerate(images):
                page_number = i + 1

                # OCR auf dem Bild ausführen
                text = pytesseract.image_to_string(image, lang='deu+eng')
                if text:
                    content.append(f"=== SEITE {page_number} OCR ===\n{text}")

                # Tabellenextraktion mittels OCR
                tables = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

                # Versuche, Tabellen zu rekonstruieren (vereinfachte Version)
                if tables and 'text' in tables:
                    table_content = []
                    current_line = []
                    current_line_num = -1

                    for i in range(len(tables['text'])):
                        if tables['text'][i].strip():
                            if tables['line_num'][i] != current_line_num:
                                if current_line:
                                    table_content.append(" | ".join(current_line))
                                current_line = []
                                current_line_num = tables['line_num'][i]
                            current_line.append(tables['text'][i])

                    if current_line:
                        table_content.append(" | ".join(current_line))

                    if table_content:
                        content.append(f"=== TABELLEN SEITE {page_number} OCR ===\n" + "\n".join(table_content))

            return "\n\n".join(content)

        except ImportError:
            logger.warning("pytesseract oder pdf2image nicht installiert. OCR wird übersprungen.")
            return "OCR nicht verfügbar. Benötigte Pakete: pytesseract, pdf2image."
        except Exception as e:
            logger.error(f"Fehler bei der OCR-Extraktion: {str(e)}")
            return f"OCR-Fehler: {str(e)}"

    def run_ollama_live(self, prompt: str, max_tokens: int = None) -> str:
        """
        Führt das Ollama-Modell über eine direkte Prozessverbindung aus und gibt die Antwort zurück.

        Args:
            prompt: Der Prompt für das Ollama-Modell
            max_tokens: Optional, maximale Anzahl an Tokens in der Antwort

        Returns:
            Die Antwort des Ollama-Modells
        """
        import subprocess
        import sys

        cmd = ["ollama", "run", self.ollama_model]
        if max_tokens:
            cmd.extend(["--max-tokens", str(max_tokens)])

        logger.info(f"Starte Ollama-Prozess mit Modell {self.ollama_model}")

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
            output.append(line)

        response = ''.join(output)
        logger.info(f"Ollama-Antwort erhalten: {len(response)} Zeichen")
        return response

    def _extract_with_ollama(self, pdf_content: str, more_detailed: bool = False) -> List[Dict[str, Any]]:
        """
        Extrahiert Fahrzeugdaten aus dem PDF-Inhalt mit Hilfe von Ollama.

        Args:
            pdf_content (str): Der PDF-Inhalt als Text
            more_detailed (bool): Ob detailliertere Anweisungen gegeben werden sollen

        Returns:
            List[Dict[str, Any]]: Liste von extrahierten Fahrzeugdaten
        """
        # Erstelle den KI-Prompt basierend auf dem PDF-Inhalt
        prompt = self._create_extraction_prompt(pdf_content, more_detailed)

        # Führe Ollama aus
        response = self.run_ollama_live(prompt)

        # Extrahiere JSON aus der Antwort
        json_content = self._extract_code_from_response(response)

        try:
            # Parsen des JSON
            data = json.loads(json_content)

            # Stelle sicher, dass es sich um eine Liste handelt
            if not isinstance(data, list):
                if isinstance(data, dict):
                    # Einzelnes Fahrzeug als Liste zurückgeben
                    data = [data]
                else:
                    logger.error("Antwort ist weder ein Fahrzeug-Objekt noch eine Liste von Fahrzeugen.")
                    return []

            # Standardisiere und konvertiere Typen
            standardized_vehicles = []
            for vehicle in data:
                standardized = self._standardize_vehicle_data(vehicle)
                if self._validate_vehicle_data(standardized):
                    standardized_vehicles.append(standardized)
                else:
                    logger.warning(f"Ungültiges Fahrzeug übersprungen: {vehicle}")

            logger.info(f"Erfolgreich {len(standardized_vehicles)} Fahrzeuge aus JSON extrahiert")
            return standardized_vehicles

        except json.JSONDecodeError as e:
            logger.error(f"Fehler beim Parsen des JSON: {str(e)}")
            logger.debug(f"JSON-Inhalt: {json_content[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei der Verarbeitung der Ollama-Antwort: {str(e)}")
            return []

    def _extract_code_from_response(self, response: str) -> str:
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

        # Falls kein JSON gefunden wurde, suche nach JSON-Objekt
        json_object_pattern = re.compile(r'{\s*"[^"]+"\s*:.*}', re.DOTALL)
        object_matches = json_object_pattern.findall(cleaned)

        if object_matches:
            return object_matches[0].strip()

        # Falls kein JSON gefunden wurde, gib die gesamte bereinigte Antwort zurück
        logger.warning("Kein JSON in der Antwort gefunden. Verwende bereinigte Antwort.")
        return cleaned.strip()

    def _create_extraction_prompt(self, pdf_content: str, more_detailed: bool = False) -> str:
        """
        Erstellt einen optimierten Prompt für die Fahrzeugdatenextraktion.

        Args:
            pdf_content (str): Der zu analysierende PDF-Inhalt
            more_detailed (bool): Ob detailliertere Anweisungen gegeben werden sollen

        Returns:
            str: Der Prompt für Ollama
        """
        # Basis-Prompt
        base_prompt = """Du bist ein spezialisierter Datenextraktionsassistent für Fahrzeugauktionskataloge. Deine Aufgabe ist es, alle Fahrzeugeinträge mit ihren Detailinformationen zu identifizieren und zu extrahieren.

Analysiere den folgenden Auktionskatalog-Inhalt und extrahiere alle Fahrzeuge mit diesen Informationen:
- Nummer/ID des Fahrzeugs
- Marke (z.B. BMW, Mercedes-Benz, Audi)
- Modell (z.B. 320d, Sprinter, A4)
- PS und kW-Werte
- Baujahr/Erstzulassung (EZ)
- Kilometerstand
- Auktionspreis/Ausruf
- MwSt-Status (Netto, §25a)
- Ausstattungsdetails (optional)
- Fahrzeugidentifikationsnummer/Fahrgestellnummer (falls vorhanden)

Beachte folgende Punkte bei der Extraktion:
1. Mercedes-Benz sollte immer als "Mercedes-Benz" extrahiert werden, nicht nur als "Mercedes"
2. Bei Datumsangaben wie "Nov 23" oder "Mär 19" handelt es sich um Monat und Jahr der Erstzulassung
3. Konvertiere das Baujahr immer in ein vierstelliges Jahr (z.B. "23" → 2023, "19" → 2019)
4. Analysiere besonders Tabellen genau, da sie oft die strukturierten Fahrzeugdaten enthalten
5. Extrahiere nur tatsächlich vorhandene Fahrzeuge und erfinde keine Daten

Formatiere deine Antwort als JSON-Array mit diesen Feldern:
- nummer: Losnummer oder ID des Fahrzeugs
- marke: Herstellername (z.B. "BMW", "Mercedes-Benz")
- modell: Modellbezeichnung
- leistung: PS-Wert als Zahl
- leistung_kw: Leistung in kW als Zahl
- baujahr: 4-stelliges Jahr
- kilometerstand: Kilometerstand als Zahl ohne Einheit
- auktionspreis: Preis als Zahl ohne Währungssymbol oder Tausendertrennzeichen
- mwst: MwSt-Status ("Netto", "Brutto", "§25a")
- kraftstoff: Kraftstoffart (falls angegeben)
- ausstattung: Liste der Ausstattungsmerkmale (falls angegeben)
- fahrgestellnummer: Fahrzeugidentifikationsnummer (falls angegeben)"""

        # Detailliertere Anweisungen für schwierigere Dokumente
        if more_detailed:
            base_prompt += """

WICHTIGE ZUSATZANWEISUNGEN:
- Erkenne auch unklare oder unvollständige Daten und extrahiere so viel wie möglich
- Verwende bei Mercedes-Fahrzeugen immer "Mercedes-Benz" als Markenname
- Achte auf Muster wie "EZ MM/YY" für Erstzulassung und konvertiere in vierstelliges Jahr
- Kilometerstände haben oft Formate wie "110.000 km" - extrahiere nur die Zahl
- Preise können als "15.500,- Euro", "15.500 €", usw. dargestellt sein - extrahiere nur die Zahlenwerte
- Beachte, dass ein Fahrzeugeintrag mehrere Zeilen umfassen kann
- Suche nach Mustern wie "Los-Nr. X" oder fortlaufende Nummerierungen am Zeilenbeginn
- Wenn eine Fahrgestellnummer im Format "WBA...", "WDD...", "VIN:..." gefunden wird, extrahiere sie

Bei der OCR-Texterkennung können Zeichen falsch erkannt worden sein. Versuche, häufige Fehler zu korrigieren:
- "0" und "O" werden oft verwechselt
- "I", "l" und "1" können verwechselt werden
- Leerzeichen können fehlen oder zusätzlich eingefügt sein"""

        # Anweisung zur JSON-Formatierung
        json_format = """

Gib deine Antwort als JSON-Array im folgenden Format zurück:
```json
[
  {
    "nummer": "1",
    "marke": "Mercedes-Benz",
    "modell": "Sprinter 317 CDI",
    "leistung": 170,
    "leistung_kw": 125,
    "baujahr": 2023,
    "kilometerstand": 44000,
    "auktionspreis": 30000,
    "mwst": "Netto",
    "kraftstoff": "Diesel",
    "ausstattung": "Klima, Sitzheizung, DPF, ZV mit FB",
    "fahrgestellnummer": "WDB1234567890"
  },
  ... weitere Fahrzeuge ...
]
```

Sei präzise und vollständig in der Datenextraktion. Wenn ein Feld nicht gefunden werden kann, lasse es weg oder verwende null. Achte darauf, dass numerische Felder als Zahlen formatiert sind, nicht als Strings.

Hier ist der Inhalt des Auktionskatalogs:"""

        # Finaler Prompt
        final_prompt = base_prompt + json_format + "\n\n" + pdf_content

        # Prüfe, ob der Prompt zu lang ist und kürze ihn bei Bedarf
        if len(final_prompt) > 32000:  # Ungefähres Limit für Ollama
            logger.warning(f"Prompt zu lang ({len(final_prompt)} Zeichen), kürze auf 32000 Zeichen")
            content_limit = 32000 - len(base_prompt) - len(json_format) - 100
            truncated_content = pdf_content[:content_limit] + "\n\n[INHALT GEKÜRZT WEGEN LÄNGENBESCHRÄNKUNG]"
            final_prompt = base_prompt + json_format + "\n\n" + truncated_content

        return final_prompt

    def _standardize_vehicle_data(self, vehicle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardisiert Fahrzeugdaten und konvertiert Werte in die richtigen Typen.

        Args:
            vehicle (dict): Rohe Fahrzeugdaten

        Returns:
            dict: Standardisierte Fahrzeugdaten
        """
        standardized = {}

        # Marken-Mappings für die Normalisierung
        brand_mapping = {
            'mercedes': 'Mercedes-Benz',
            'mercedesbenz': 'Mercedes-Benz',
            'vw': 'Volkswagen',
            'bmw': 'BMW',
            'audi': 'Audi',
            'opel': 'Opel',
            'ford': 'Ford',
            'porsche': 'Porsche',
            'citroen': 'Citroën',
            'seat': 'SEAT',
            'skoda': 'Škoda'
        }

        # Kopiere und standardisiere jedes Feld
        for key, value in vehicle.items():
            # Schlüssel normalisieren
            normalized_key = key.lower().strip()

            # Spezifische Feldverarbeitung
            if normalized_key in ['marke', 'hersteller', 'brand', 'make']:
                if value:
                    brand = str(value).lower().strip().replace(' ', '')
                    standardized['marke'] = brand_mapping.get(brand, str(value).strip())

            elif normalized_key in ['modell', 'model', 'typ', 'type']:
                if value:
                    # Mercedes-Benz spezifische Behandlung
                    model_value = str(value).strip()
                    if model_value.startswith('Benz') and standardized.get('marke') == 'Mercedes-Benz':
                        model_value = model_value.replace('Benz', '').strip()
                    standardized['modell'] = model_value

            elif normalized_key in ['baujahr', 'year', 'ez', 'erstzulassung', 'year_of_manufacture']:
                try:
                    # Extrahiere nur Zahlen
                    if value is None:
                        continue

                    year_str = str(value).strip()

                    # Entferne Text und behalte nur Zahlen
                    digits = ''.join(filter(str.isdigit, year_str))

                    if len(digits) == 2:
                        # 2-stelliges Jahr konvertieren
                        year_val = int(digits)
                        if year_val < 50:  # Annahme: 00-49 -> 2000-2049
                            standardized['baujahr'] = 2000 + year_val
                        else:  # Annahme: 50-99 -> 1950-1999
                            standardized['baujahr'] = 1900 + year_val
                    elif len(digits) == 4:
                        # 4-stelliges Jahr direkt übernehmen
                        standardized['baujahr'] = int(digits)
                    else:
                        # Versuche, das Jahr aus einem Datum zu extrahieren
                        date_match = re.search(r'(\d{1,2})\/(\d{2,4})', year_str)
                        if date_match:
                            year_part = date_match.group(2)
                            if len(year_part) == 2:
                                year_val = int(year_part)
                                if year_val < 50:
                                    standardized['baujahr'] = 2000 + year_val
                                else:
                                    standardized['baujahr'] = 1900 + year_val
                            else:
                                standardized['baujahr'] = int(year_part)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Fehler bei der Konvertierung des Baujahrs '{value}': {str(e)}")

            elif normalized_key in ['kilometerstand', 'km', 'mileage', 'laufleistung']:
                try:
                    if value is None:
                        continue

                    # Extrahiere nur Zahlen
                    km_str = str(value).strip()
                    digits = ''.join(filter(str.isdigit, km_str))

                    if digits:
                        standardized['kilometerstand'] = int(digits)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Fehler bei der Konvertierung des Kilometerstands '{value}': {str(e)}")

            elif normalized_key in ['auktionspreis', 'price', 'preis', 'ausruf']:
                try:
                    if value is None:
                        continue

                    # Konvertiere Preisangaben
                    price_str = str(value).strip()

                    # Entferne Währungssymbole und Tausendertrennzeichen
                    cleaned = re.sub(r'[^\d,.]', '', price_str)

                    # Ersetze Kommas durch Punkte für Dezimalstellen
                    if ',' in cleaned and '.' in cleaned:
                        # Format: 1.234,56
                        cleaned = cleaned.replace('.', '').replace(',', '.')
                    elif ',' in cleaned:
                        # Format: 1234,56
                        cleaned = cleaned.replace(',', '.')

                    if cleaned:
                        price_val = float(cleaned)
                        standardized['auktionspreis'] = price_val

                        # Wenn es sich um einen ganzzahligen Wert handelt, als int speichern
                        if price_val == int(price_val):
                            standardized['auktionspreis'] = int(price_val)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Fehler bei der Konvertierung des Preises '{value}': {str(e)}")

            elif normalized_key in ['leistung', 'ps', 'hp', 'horsepower']:
                try:
                    if value is not None:
                        # Extrahiere nur Zahlen
                        ps_str = str(value).strip()
                        digits = ''.join(filter(str.isdigit, ps_str))

                        if digits:
                            standardized['leistung'] = int(digits)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Fehler bei der Konvertierung der Leistung (PS) '{value}': {str(e)}")

            elif normalized_key in ['leistung_kw', 'kw']:
                try:
                    if value is not None:
                        # Extrahiere nur Zahlen
                        kw_str = str(value).strip()
                        digits = ''.join(filter(str.isdigit, kw_str))

                        if digits:
                            standardized['leistung_kw'] = int(digits)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Fehler bei der Konvertierung der Leistung (kW) '{value}': {str(e)}")

            elif normalized_key in ['kraftstoff', 'fuel', 'fuel_type']:
                if value:
                    standardized['kraftstoff'] = str(value).strip()

            elif normalized_key in ['ausstattung', 'equipment', 'features', 'options']:
                if value:
                    if isinstance(value, list):
                        standardized['ausstattung'] = ', '.join(str(item).strip() for item in value)
                    else:
                        standardized['ausstattung'] = str(value).strip()

            elif normalized_key in ['fahrgestellnummer', 'vin', 'chassis_number', 'vehicle_identification_number']:
                if value:
                    standardized['fahrgestellnummer'] = str(value).strip()

            elif normalized_key in ['mwst', 'vat', 'tax', 'steuer']:
                if value:
                    mwst_value = str(value).strip().lower()

                    # Standardisiere MwSt-Statuswerte
                    if 'netto' in mwst_value or 'net' in mwst_value:
                        standardized['mwst'] = 'Netto'
                    elif 'brutto' in mwst_value or 'gross' in mwst_value:
                        standardized['mwst'] = 'Brutto'
                    elif '25a' in mwst_value or '25 a' in mwst_value:
                        standardized['mwst'] = '§25a'
                    else:
                        standardized['mwst'] = str(value).strip()

            elif normalized_key in ['nummer', 'id', 'lot', 'lot_number', 'losnummer']:
                if value:
                    standardized['nummer'] = str(value).strip()

            elif not any(normalized_key in valid_keys for valid_keys in [
                ['marke', 'hersteller', 'brand', 'make'],
                ['modell', 'model', 'typ', 'type'],
                ['baujahr', 'year', 'ez', 'erstzulassung', 'year_of_manufacture'],
                ['kilometerstand', 'km', 'mileage', 'laufleistung'],
                ['auktionspreis', 'price', 'preis', 'ausruf'],
                ['leistung', 'ps', 'hp', 'horsepower'],
                ['leistung_kw', 'kw'],
                ['kraftstoff', 'fuel', 'fuel_type'],
                ['ausstattung', 'equipment', 'features', 'options'],
                ['fahrgestellnummer', 'vin', 'chassis_number', 'vehicle_identification_number'],
                ['mwst', 'vat', 'tax', 'steuer'],
                ['nummer', 'id', 'lot', 'lot_number', 'losnummer']
            ]):
                # Unbekannte Schlüssel unverändert übernehmen
                standardized[normalized_key] = value

        # Berechne fehlende Werte, wenn möglich
        if 'leistung' in standardized and 'leistung_kw' not in standardized:
            try:
                # PS zu kW: PS * 0.735
                standardized['leistung_kw'] = int(round(standardized['leistung'] * 0.735))
            except:
                pass

        elif 'leistung_kw' in standardized and 'leistung' not in standardized:
            try:
                # kW zu PS: kW / 0.735
                standardized['leistung'] = int(round(standardized['leistung_kw'] / 0.735))
            except:
                pass

        return standardized

    def _validate_vehicle_data(self, vehicle: Dict[str, Any]) -> bool:
        """
        Validiert, ob die Fahrzeugdaten ausreichend und plausibel sind.

        Args:
            vehicle (dict): Zu validierende Fahrzeugdaten

        Returns:
            bool: True wenn die Daten gültig sind, sonst False
        """
        # Minimale erforderliche Felder
        required_fields = ['marke', 'modell']

        # Mindestens ein zusätzliches Feld für die Identifikation sollte vorhanden sein
        identification_fields = ['baujahr', 'kilometerstand', 'auktionspreis', 'leistung', 'leistung_kw']

        # Prüfe, ob alle erforderlichen Felder vorhanden sind
        if not all(field in vehicle and vehicle[field] for field in required_fields):
            return False

        # Prüfe, ob mindestens ein Identifikationsfeld vorhanden ist
        if not any(field in vehicle and vehicle[field] for field in identification_fields):
            return False

        # Plausibilitätsprüfungen
        if 'baujahr' in vehicle:
            # Baujahr sollte zwischen 1900 und aktuellem Jahr + 1 liegen
            import datetime
            current_year = datetime.datetime.now().year
            if not (1900 <= vehicle['baujahr'] <= current_year + 1):
                logger.warning(f"Unplausibles Baujahr: {vehicle['baujahr']}")
                return False

        if 'kilometerstand' in vehicle:
            # Kilometerstand sollte positiv sein und unter einem sehr hohen Wert liegen
            if not (0 <= vehicle['kilometerstand'] <= 1000000):
                logger.warning(f"Unplausibler Kilometerstand: {vehicle['kilometerstand']}")
                return False

        if 'auktionspreis' in vehicle:
            # Preis sollte positiv sein und in einem sinnvollen Bereich liegen
            if not (0 <= vehicle['auktionspreis'] <= 10000000):
                logger.warning(f"Unplausibler Auktionspreis: {vehicle['auktionspreis']}")
                return False

        return True

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

        if not directory.exists():
            logger.error(f"Verzeichnis {directory_path} existiert nicht.")
            return pd.DataFrame()

        for pdf_file in directory.glob("*.pdf"):
            logger.info(f"Verarbeite {pdf_file}")
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
            uploaded_files: Liste von hochgeladenen Dateiobjekten (z.B. von Streamlit)

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