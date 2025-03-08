# auto_auction_analyzer/pdf_extractor/auction_catalog_extractor.py
import pdfplumber
import re
import pandas as pd
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AuctionCatalogExtractor:
    """Extraktor speziell für Auktionskataloge im Format der Autobid/Auktion&Markt AG"""

    def __init__(self):
        # Reguläre Ausdrücke für typische Fahrzeugdatenformate in Auktionskatalogen
        self.patterns = {
            # Typische Muster für Auktionskataloge
            'fahrzeug_zeile': r'^\s*(\d+)\s+([A-Za-z\-]+)\s+(.+?)\s+(\d+)\s+(\d+)\s+(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\s+(\d{2})\s+(\d+)',
            'mercedes_benz': r'Mercedes[-\s]Benz',
            'preis': r'(\d+[\.,]\d+)\s*(?:€|EUR|Netto)',
            'ausstattung': r'VS|1\. Halter|2\. Halter|3\. Halter|Import|Ex-Mietwagen'
        }

        # Markenmapping (für häufige Abkürzungen und Normalisierungen)
        self.marken_mapping = {
            'Mercedes': 'Mercedes-Benz',
            'Mercedes-Benz': 'Mercedes-Benz',
            'VW': 'Volkswagen',
            'Citroën': 'Citroën',
            'Citroen': 'Citroën'
        }

    def extract_from_pdf(self, pdf_path):
        """
        Extrahiert Fahrzeugdaten aus einem Auktionskatalog-PDF.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            list: Liste von extrahierten Fahrzeugdaten
        """
        logger.info(f"Extrahiere Daten aus: {pdf_path}")
        all_vehicles = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Verarbeite jede Seite des PDFs
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"Verarbeite Seite {page_num+1} von {len(pdf.pages)}")

                    # Extrahiere Text von der Seite
                    text = page.extract_text()

                    if text:
                        # Extrahiere Fahrzeuge aus der Seite
                        vehicles = self._extract_vehicles_from_page_text(text)
                        all_vehicles.extend(vehicles)

                logger.info(f"Insgesamt {len(all_vehicles)} Fahrzeuge extrahiert")

                # Wenn keine Fahrzeuge gefunden wurden, versuche alternative Extraktionsmethode
                if not all_vehicles:
                    logger.info("Versuche alternative Extraktionsmethode...")
                    all_vehicles = self._extract_with_line_by_line(pdf_path)

                return all_vehicles

        except Exception as e:
            logger.error(f"Fehler bei der Extraktion aus {pdf_path}: {str(e)}")
            # Versuche alternative Methode bei Fehlern
            try:
                return self._extract_with_line_by_line(pdf_path)
            except Exception as e2:
                logger.error(f"Auch alternative Methode fehlgeschlagen: {str(e2)}")
                return []

    def _extract_vehicles_from_page_text(self, text):
        """
        Extrahiert Fahrzeugdaten aus dem Text einer Seite.

        Args:
            text (str): Text einer PDF-Seite

        Returns:
            list: Liste mit extrahierten Fahrzeugdaten
        """
        vehicles = []
        lines = text.split('\n')

        # Muster für Auktionskatalog-Einträge
        vehicle_pattern = r'^\s*(\d+)\s+([A-Za-z\-]+)\s+(.+?)\s+(\d+)\s+(\d+)\s+(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\s+(\d{2})\s+(\d+)'
        marke_modell_pattern = r'^\s*(\d+)\s+([A-Za-z\-]+)\s+(.+)'

        current_vehicle = None

        for line_idx, line in enumerate(lines):
            # Versuche, einen kompletten Fahrzeugeintrag zu finden
            match = re.search(vehicle_pattern, line)

            if match:
                # Wenn wir einen vollständigen Eintrag gefunden haben
                nr, marke, modell, ps, kw, monat, jahr, km = match.groups()

                # Bereinige Markenname
                marke = self._normalize_marke(marke)

                # Bereite Fahrzeugdaten vor
                vehicle = {
                    'marke': marke,
                    'modell': modell.strip(),
                    'leistung': ps,
                    'leistung_kw': kw,
                    'baujahr': f"20{jahr}" if int(jahr) < 50 else f"19{jahr}",
                    'kilometerstand': km
                }

                # Suche nach Preis in dieser Zeile oder nahen Zeilen
                price_found = False
                for i in range(max(0, line_idx - 1), min(len(lines), line_idx + 3)):
                    price_match = re.search(r'(\d+[\.,]\d+)\s*(?:€|EUR|Netto)', lines[i])
                    if price_match:
                        vehicle['auktionspreis'] = price_match.group(1).replace('.', '').replace(',', '.')
                        price_found = True
                        break

                if not price_found:
                    vehicle['auktionspreis'] = None

                vehicles.append(vehicle)
                continue

            # Wenn kein vollständiger Eintrag gefunden wurde, suche nach Marke und Modell
            marke_modell_match = re.search(marke_modell_pattern, line)
            if marke_modell_match and len(marke_modell_match.groups()) >= 3:
                nr, marke, modell = marke_modell_match.groups()

                # Suche in der nächsten Zeile nach weiteren Daten
                if line_idx + 1 < len(lines):
                    next_line = lines[line_idx + 1]
                    ps_match = re.search(r'(\d+)\s+(\d+)', next_line)
                    date_match = re.search(r'(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\s+(\d{2})', next_line)
                    km_match = re.search(r'(\d+)\s+(?:km|Tkm|abgelesen)', next_line)

                    if ps_match and date_match:
                        ps, kw = ps_match.groups()
                        monat, jahr = date_match.groups()
                        km = km_match.group(1) if km_match else None

                        # Bereinige Markenname
                        marke = self._normalize_marke(marke)

                        # Bereite Fahrzeugdaten vor
                        vehicle = {
                            'marke': marke,
                            'modell': modell.strip(),
                            'leistung': ps,
                            'leistung_kw': kw,
                            'baujahr': f"20{jahr}" if int(jahr) < 50 else f"19{jahr}",
                            'kilometerstand': km
                        }

                        # Suche nach Preis
                        for i in range(max(0, line_idx - 1), min(len(lines), line_idx + 4)):
                            price_match = re.search(r'(\d+[\.,]\d+)\s*(?:€|EUR|Netto)', lines[i])
                            if price_match:
                                vehicle['auktionspreis'] = price_match.group(1).replace('.', '').replace(',', '.')
                                break
                            else:
                                vehicle['auktionspreis'] = None

                        vehicles.append(vehicle)

        # Besondere Behandlung für Mercedes-Benz Sprinter und Vito
        for vehicle in vehicles:
            # Mercedes-Benz spezifische Korrekturen
            if vehicle['marke'] == 'Mercedes-Benz' or vehicle['modell'].startswith('Sprinter') or vehicle['modell'].startswith('Vito'):
                if 'Sprinter' in vehicle['modell'] and vehicle['marke'] != 'Mercedes-Benz':
                    vehicle['marke'] = 'Mercedes-Benz'
                    vehicle['modell'] = 'Sprinter ' + vehicle['modell'].replace('Sprinter', '').strip()
                elif 'Vito' in vehicle['modell'] and vehicle['marke'] != 'Mercedes-Benz':
                    vehicle['marke'] = 'Mercedes-Benz'
                    vehicle['modell'] = 'Vito ' + vehicle['modell'].replace('Vito', '').strip()

        # Identifiziere und verbinde unvollständige Einträge
        merged_vehicles = []
        i = 0
        while i < len(vehicles):
            vehicle = vehicles[i]

            # Prüfe, ob das aktuelle Fahrzeug unvollständig ist und das nächste fehlende Daten ergänzen kann
            if i + 1 < len(vehicles):
                next_vehicle = vehicles[i + 1]

                # Wenn das aktuelle Fahrzeug keinen Preis hat, aber das nächste schon
                if vehicle['auktionspreis'] is None and next_vehicle['auktionspreis'] is not None:
                    vehicle['auktionspreis'] = next_vehicle['auktionspreis']
                    i += 2  # Überspringe beide Fahrzeuge
                    merged_vehicles.append(vehicle)
                    continue

            # Füge das aktuelle Fahrzeug hinzu
            merged_vehicles.append(vehicle)
            i += 1

        return merged_vehicles

    def _extract_with_line_by_line(self, pdf_path):
        """
        Alternative Extraktionsmethode mit zeilenweiser Analyse.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            list: Liste von extrahierten Fahrzeugen
        """
        vehicles = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    all_text += page.extract_text() + "\n"

                lines = all_text.split('\n')
                i = 0

                while i < len(lines):
                    line = lines[i]

                    # Suche nach Zeilen, die ein Auktionsfahrzeug beschreiben könnten
                    nr_match = re.match(r'^\s*(\d+)\s+', line)
                    if nr_match:
                        # Identifiziere Teile: Marke, Modell, etc.
                        # Dies ist ein komplexes Muster für autobid.de Kataloge
                        if i + 2 < len(lines):  # Mindestens 3 Zeilen für einen Eintrag
                            # Versuche, Marke und Modell zu extrahieren
                            marke_modell_match = re.search(r'^\s*\d+\s+([A-Za-z\-]+)\s+(.+?)(?=\s+\d+\s+\d+|$)', line)

                            if marke_modell_match:
                                marke, modell = marke_modell_match.groups()
                                marke = self._normalize_marke(marke)

                                # Suche nach technischen Daten in den nächsten Zeilen
                                tech_data_line = lines[i+1]
                                tech_match = re.search(r'(\d+)\s+(\d+)\s+(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\s+(\d{2})\s+(\d+)', tech_data_line)

                                if tech_match:
                                    ps, kw, monat, jahr, km = tech_match.groups()

                                    # Suche nach Preis und Ausstattung
                                    price = None
                                    for j in range(i, min(i+5, len(lines))):
                                        price_match = re.search(r'(\d+[\.,]\d+)\s*(?:€|EUR|Netto)', lines[j])
                                        if price_match:
                                            price = price_match.group(1).replace('.', '').replace(',', '.')
                                            break

                                    # Erstelle Fahrzeugobjekt
                                    vehicle = {
                                        'marke': marke,
                                        'modell': modell.strip(),
                                        'leistung': ps,
                                        'leistung_kw': kw,
                                        'baujahr': f"20{jahr}" if int(jahr) < 50 else f"19{jahr}",
                                        'kilometerstand': km,
                                        'auktionspreis': price
                                    }

                                    vehicles.append(vehicle)

                    i += 1

                # Wenn noch keine Fahrzeuge gefunden wurden, versuche einen letzten Ansatz
                if not vehicles:
                    # Letzter Versuch: Suche nach bekannten Mustern wie "Mercedes-Benz Sprinter"
                    known_models = [
                        (r'Mercedes[-\s]Benz\s+Sprinter', 'Mercedes-Benz', 'Sprinter'),
                        (r'Mercedes[-\s]Benz\s+Vito', 'Mercedes-Benz', 'Vito'),
                        (r'VW\s+Crafter', 'Volkswagen', 'Crafter'),
                        (r'Ford\s+Transit', 'Ford', 'Transit'),
                        (r'Ford\s+Tourneo', 'Ford', 'Tourneo'),
                        (r'Opel\s+Movano', 'Opel', 'Movano'),
                        (r'Fiat\s+Ducato', 'Fiat', 'Ducato')
                    ]

                    for i, line in enumerate(lines):
                        for pattern, marke, base_modell in known_models:
                            match = re.search(pattern, line)
                            if match:
                                # Suche weitere Details in den nahegelegenen Zeilen
                                ps_match = None
                                date_match = None
                                km_match = None
                                price_match = None

                                # Suche in den nächsten 5 Zeilen
                                for j in range(max(0, i-2), min(i+5, len(lines))):
                                    if ps_match is None:
                                        ps_match = re.search(r'(\d+)\s+PS', lines[j])
                                    if date_match is None:
                                        date_match = re.search(r'(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\s+(\d{2})', lines[j])
                                    if km_match is None:
                                        km_match = re.search(r'(\d+)\s+(?:km|Tkm|abgelesen)', lines[j])
                                    if price_match is None:
                                        price_match = re.search(r'(\d+[\.,]\d+)\s*(?:€|EUR|Netto)', lines[j])

                                # Extrahiere Details
                                ps = ps_match.group(1) if ps_match else None
                                jahr = date_match.group(2) if date_match else None
                                km = km_match.group(1) if km_match else None
                                price = price_match.group(1).replace('.', '').replace(',', '.') if price_match else None

                                # Erstelle ein Fahrzeugobjekt mit den verfügbaren Daten
                                vehicle = {
                                    'marke': marke,
                                    'modell': base_modell,
                                    'leistung': ps,
                                    'baujahr': f"20{jahr}" if jahr and int(jahr) < 50 else f"19{jahr}" if jahr else None,
                                    'kilometerstand': km,
                                    'auktionspreis': price
                                }

                                vehicles.append(vehicle)

                return vehicles

        except Exception as e:
            logger.error(f"Fehler bei der alternativen Extraktion: {str(e)}")
            return []

    def _normalize_marke(self, marke):
        """Normalisiert Markennamen"""
        if not marke:
            return marke

        marke = marke.strip()
        return self.marken_mapping.get(marke, marke)

    def process_directory(self, directory_path):
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
            logger.info(f"Erfolgreich {len(results)} Fahrzeuge aus PDFs extrahiert.")
            return df
        else:
            logger.warning(f"Keine Fahrzeuge aus PDFs in {directory_path} extrahiert.")
            return pd.DataFrame()

    def process_auction_pdfs(self, uploaded_files):
        """
        Verarbeitet hochgeladene PDF-Dateien und extrahiert Fahrzeugdaten.

        Args:
            uploaded_files: Liste von hochgeladenen Streamlit-Dateiobjekten

        Returns:
            pd.DataFrame: DataFrame mit extrahierten Fahrzeugdaten
        """
        all_vehicles = []

        for uploaded_file in uploaded_files:
            # Speichere die Datei temporär
            temp_path = f"temp_{uploaded_file.name}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extrahiere Fahrzeuge aus der Datei
                vehicles = self.extract_from_pdf(temp_path)

                # Füge Dateinamen hinzu
                for vehicle in vehicles:
                    vehicle['dateiname'] = uploaded_file.name

                all_vehicles.extend(vehicles)

                # Lösche temporäre Datei
                import os
                os.remove(temp_path)

            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten von {uploaded_file.name}: {str(e)}")

        # Erstelle DataFrame aus allen extrahierten Fahrzeugen
        if all_vehicles:
            return pd.DataFrame(all_vehicles)
        else:
            return pd.DataFrame()

# Wenn direkt ausgeführt, starte interaktive Extraktion
if __name__ == "__main__":
    extractor = AuctionCatalogExtractor()

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        vehicles = extractor.extract_from_pdf(pdf_path)

        if vehicles:
            print(f"Erfolgreich {len(vehicles)} Fahrzeuge extrahiert.")
            for i, vehicle in enumerate(vehicles[:5]):  # Zeige die ersten 5 Fahrzeuge
                print(f"\nFahrzeug {i+1}:")
                for key, value in vehicle.items():
                    print(f"  {key}: {value}")

            # Speichere als CSV
            df = pd.DataFrame(vehicles)
            output_path = "extrahierte_fahrzeugdaten.csv"
            df.to_csv(output_path, index=False)
            print(f"\nAlle Fahrzeuge wurden in {output_path} gespeichert.")
        else:
            print("Keine Fahrzeuge extrahiert.")
    else:
        print("Bitte geben Sie den Pfad zur PDF-Datei als Argument an.")