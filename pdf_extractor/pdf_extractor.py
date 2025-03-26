# auto_auction_analyzer/pdf_extractor/pdf_extractor.py
import pdfplumber
import re
import pandas as pd
import sys
import numpy as np
import logging
import time
import json
import os
import tempfile
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
import traceback
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PdfExtractor:
    """
    Optimierte Klasse zur Extraktion von Fahrzeugdaten aus Auktionskatalogen.
    Kombiniert regelbasierte und KI-gestützte Methoden mit robuster Fehlerbehandlung.
    """

    def __init__(self, templates_dir=None, ollama_model="deepseek-r1:14b"):
        """
        Initialisiert den PDF-Extraktor.

        Args:
            templates_dir (str, optional): Verzeichnis mit Extraktions-Templates
            ollama_model (str): Name des zu verwendenden Ollama-Modells
        """
        self.templates_dir = templates_dir or os.path.join(os.path.dirname(__file__), "templates")
        self.templates = self._load_templates()
        self.ollama_model = ollama_model
        logger.info(f"PDF-Extraktor initialisiert mit Ollama-Modell: {ollama_model}")

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
        Extrahiert Fahrzeugdaten aus einem PDF mit KI-basierter Extraktion.

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

            # Schritt 2: Extrahiere den PDF-Inhalt mit erweiterter Genauigkeit
            logger.info("Verwende KI-basierte Extraktion")
            pdf_content = self._extract_pdf_content_enhanced(pdf_path)

            # Schritt 3: KI-Extraktion mit dem extrahierten Inhalt
            ai_results = self._extract_with_ollama(pdf_content)

            # Validiere die Ergebnisse
            valid_ai_results = [r for r in ai_results if self._validate_vehicle_data(r)]
            logger.info(f"KI-basierte Extraktion lieferte {len(valid_ai_results)} valide Fahrzeuge.")

            return valid_ai_results

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
                    logger.warning(f"Erste PDF-Seite enthält möglicherweise keinen Text: {pdf_path}")
                    # Es könnte immer noch ein gültiges PDF sein, prüfe auf Bilder/Tabellen
                    first_page_tables = pdf.pages[0].extract_tables()
                    if not first_page_tables:
                        logger.warning(f"Keine Tabellen auf der ersten Seite gefunden: {pdf_path}")

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
            vehicles = []
            with pdfplumber.open(pdf_path) as pdf:
                # Extrahiere Text aus allen Seiten
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

                # Normale Zeilenumbrüche
                all_text = re.sub(r'\n+', '\n', all_text)
                lines = all_text.split('\n')

                current_vehicle = {}
                # Mehrere Muster für verschiedene Formate
                patterns = [
                    # Muster 1: Startet mit Nummer, enthält Marke, Modell, PS, kW
                    r'^\s*(\d+)\s+([A-Za-z\-]+)\s+(.+?)\s+(\d+)\s+(\d+)\s+([A-Za-z]{3})\s+(\d{2})\s+(\d+)',

                    # Muster 2: Mercedes-Benz Format
                    r'Mercedes-?\s*Benz\s+([A-Za-z0-9\s\-\.]+)\s+(\d+)\s+(\d+)\s+([A-Za-z]{3})\s+(\d{2})',

                    # Muster 3: Vereinfachtes Format mit Marke und Modell
                    r'([A-Za-z\-]+)\s+([A-Za-z0-9\s\-\.]+)\s+(\d+)\s+(\d+)\s+([A-Za-z]{3})\s+(\d{2})'
                ]

                # Preismuster
                price_pattern = r'(\d+[\.,]\d+)\s*(?:€|EUR|Netto)'

                for i, line in enumerate(lines):
                    # Überprüfe jedes Muster
                    for pattern_idx, pattern in enumerate(patterns):
                        match = re.search(pattern, line)
                        if match:
                            # Wenn wir ein vorheriges Fahrzeug haben, füge es zur Liste hinzu
                            if current_vehicle and 'marke' in current_vehicle:
                                vehicles.append(current_vehicle)
                                current_vehicle = {}

                            # Extrahiere Daten basierend auf dem Muster
                            if pattern_idx == 0:  # Muster 1
                                current_vehicle['nummer'] = match.group(1)
                                current_vehicle['marke'] = match.group(2)
                                current_vehicle['modell'] = match.group(3).strip()
                                current_vehicle['leistung'] = int(match.group(4))
                                current_vehicle['leistung_kw'] = int(match.group(5))
                                # Konvertiere Monat/Jahr in Baujahr
                                jahr = match.group(7)
                                current_vehicle['baujahr'] = 2000 + int(jahr) if int(jahr) < 50 else 1900 + int(jahr)
                                current_vehicle['kilometerstand'] = int(match.group(8))

                            elif pattern_idx == 1:  # Muster 2 - Mercedes-Benz
                                current_vehicle['marke'] = "Mercedes-Benz"
                                current_vehicle['modell'] = match.group(1).strip()
                                current_vehicle['leistung'] = int(match.group(2))
                                current_vehicle['leistung_kw'] = int(match.group(3))
                                # Konvertiere Monat/Jahr in Baujahr
                                jahr = match.group(5)
                                current_vehicle['baujahr'] = 2000 + int(jahr) if int(jahr) < 50 else 1900 + int(jahr)

                            elif pattern_idx == 2:  # Muster 3
                                current_vehicle['marke'] = match.group(1)
                                current_vehicle['modell'] = match.group(2).strip()
                                current_vehicle['leistung'] = int(match.group(3))
                                current_vehicle['leistung_kw'] = int(match.group(4))
                                # Konvertiere Monat/Jahr in Baujahr
                                jahr = match.group(6)
                                current_vehicle['baujahr'] = 2000 + int(jahr) if int(jahr) < 50 else 1900 + int(jahr)

                            # Suche nach Preis in dieser oder nächsten Zeile
                            price_match = re.search(price_pattern, line)
                            if price_match:
                                price = price_match.group(1).replace('.', '').replace(',', '.')
                                current_vehicle['auktionspreis'] = float(price)
                            elif i < len(lines) - 1:
                                price_match = re.search(price_pattern, lines[i+1])
                                if price_match:
                                    price = price_match.group(1).replace('.', '').replace(',', '.')
                                    current_vehicle['auktionspreis'] = float(price)

                            break  # Wenn ein Muster passt, keine weiteren prüfen

                # Das letzte Fahrzeug auch hinzufügen
                if current_vehicle and 'marke' in current_vehicle:
                    vehicles.append(current_vehicle)

                # Standardisiere alle extrahierten Fahrzeuge
                standardized_vehicles = []
                for vehicle in vehicles:
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

            # Methode 2: Versuche PyMuPDF als Backup für zusätzliche Strukturinformationen
            try:
                import fitz  # PyMuPDF
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
            except ImportError:
                logger.info("PyMuPDF nicht verfügbar, überspringe strukturierte Textextraktion")
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
        Versucht zuerst pytesseract zu verwenden, mit Fallback-Optionen.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            str: Mit OCR extrahierter Text
        """
        content = ["=== OCR-EXTRAHIERTER TEXT ==="]

        # Versuche mit pytesseract
        try:
            import pytesseract
            from pdf2image import convert_from_path

            # Konvertiere PDF-Seiten zu Bildern
            images = convert_from_path(pdf_path)

            for i, image in enumerate(images):
                page_number = i + 1
                # OCR auf dem Bild ausführen
                text = pytesseract.image_to_string(image, lang='deu+eng')
                if text:
                    content.append(f"=== SEITE {page_number} OCR ===\n{text}")

            return "\n\n".join(content)

        except ImportError:
            logger.warning("pytesseract nicht installiert, versuche Alternative...")

            # Fallback: Verwende Text-Extraktion mit alternativen Methoden
            try:
                import subprocess
                # Versuche pdftotext (Teil von poppler-utils)
                result = subprocess.run(
                    ["pdftotext", "-layout", pdf_path, "-"],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0 and result.stdout:
                    content.append("=== PDFTOTEXT EXTRAKTION ===")
                    content.append(result.stdout)
                    return "\n\n".join(content)
            except Exception:
                pass

            # Letzte Option: Nachricht zurückgeben, dass OCR nicht verfügbar ist
            return "OCR nicht verfügbar. Installieren Sie 'pytesseract' und 'pdf2image' für bessere Ergebnisse."

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
        cmd = ["ollama", "run", self.ollama_model]
        if max_tokens:
            cmd.extend(["--max-tokens", str(max_tokens)])

        logger.info(f"Starte Ollama-Prozess mit Modell {self.ollama_model}")

        try:
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
                sys.stdout.write(line)
                sys.stdout.flush()
                output.append(line)

            response = ''.join(output)
            logger.info(f"Ollama-Antwort erhalten: {len(response)} Zeichen")
            return response

        except FileNotFoundError:
            logger.error("Ollama wurde nicht gefunden. Bitte stellen Sie sicher, dass Ollama installiert ist.")
            return "Fehler: Ollama ist nicht installiert oder nicht im Pfad."
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung von Ollama: {str(e)}")
            return f"Fehler bei der Ausführung von Ollama: {str(e)}"

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
        Extrahiert JSON aus einer Ollama-Antwort mit verbesserter Robustheit.

        Args:
            response: Die vollständige Antwort des Ollama-Modells

        Returns:
            Der extrahierte JSON-String
        """
        # Entferne ANSI-Escape-Sequenzen
        cleaned = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', response)

        # Entferne <think>-Tags, falls vorhanden
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

        # Methode 1: Extrahiere JSON aus einem Markdown-Codeblock
        json_pattern = re.compile(r'```(?:json)?\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
        matches = json_pattern.findall(cleaned)

        if matches:
            for match in matches:
                try:
                    # Versuche, ob es gültiges JSON ist
                    json.loads(match.strip())
                    return match.strip()
                except json.JSONDecodeError:
                    continue  # Wenn nicht, probiere den nächsten Match

        # Methode 2: Suche nach JSON-Array mit weniger strengen Kriterien
        try:
            # Suche nach Text zwischen eckigen Klammern [...]
            array_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
            if array_match:
                potential_json = array_match.group(0)
                json.loads(potential_json)  # Prüfe ob es gültiges JSON ist
                return potential_json
        except:
            pass

        # Methode 3: Noch robusterer Ansatz - suche nach allen { } Blöcken
        try:
            # Finde alle Teile, die wie JSON-Objekte aussehen
            all_objects = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', cleaned, re.DOTALL)
            for obj_text in all_objects:
                # Wenn es ein einzelnes Objekt ist, packe es in ein Array
                try:
                    obj = json.loads(obj_text)
                    if isinstance(obj, dict):
                        return f"[{obj_text}]"
                except:
                    pass
        except:
            pass

        # Fallback: Manuelles Generieren von JSON aus der Antwort
        try:
            # Suche nach möglichen Fahrzeugdaten in der Antwort
            logger.info("Versuche Fahrzeugdaten manuell aus der Antwort zu extrahieren")

            # Suche nach Marke und Modell
            marke_match = re.search(r'Marke:\s*([A-Za-z\-]+)', cleaned)
            modell_match = re.search(r'Modell:\s*([A-Za-z0-9\s\-\.]+)', cleaned)

            if marke_match and modell_match:
                # Erstelle ein einfaches Fahrzeug-Objekt
                vehicle = {
                    "marke": marke_match.group(1).strip(),
                    "modell": modell_match.group(1).strip()
                }

                # Suche nach weiteren Daten
                baujahr_match = re.search(r'Baujahr:\s*(\d{4})', cleaned)
                if baujahr_match:
                    vehicle["baujahr"] = int(baujahr_match.group(1))

                km_match = re.search(r'Kilometerstand:\s*(\d+)', cleaned)
                if km_match:
                    vehicle["kilometerstand"] = int(km_match.group(1))

                preis_match = re.search(r'Preis:\s*(\d+)', cleaned)
                if preis_match:
                    vehicle["auktionspreis"] = int(preis_match.group(1))

                # Zu JSON konvertieren
                return json.dumps([vehicle])
        except Exception as e:
            logger.warning(f"Fehler beim manuellen Extrahieren von Fahrzeugdaten: {str(e)}")

        # Wenn kein JSON gefunden wurde, logge die Antwort für Debugging
        logger.warning("Kein JSON in der Antwort gefunden. Hier ist ein Auszug der Antwort:")
        logger.warning(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)

        # Notfallmaßnahme: Versuche, einen sinnvollen Standardwert zurückzugeben
        return '[]'  # Leeres Array als Fallback

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
5. Extrahiere nur tatsächlich vorhandene Fahrzeuge und erfinde keine Daten"""

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

"""

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

        # NEU: Korrektur, wenn PS und kW vertauscht sind
        if 'leistung' in standardized and 'leistung_kw' in standardized:
            # Wenn kW größer als PS ist, sind die Werte wahrscheinlich vertauscht
            if standardized['leistung_kw'] > standardized['leistung']:
                # Werte tauschen
                temp = standardized['leistung']
                standardized['leistung'] = standardized['leistung_kw']
                standardized['leistung_kw'] = temp

                # Jetzt die Werte nochmal auf Basis ihrer Relation korrigieren
                standardized['leistung_kw'] = int(round(standardized['leistung'] * 0.735))

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