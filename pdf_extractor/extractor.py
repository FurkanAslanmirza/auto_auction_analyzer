# auction_catalog_extractor.py
import pdfplumber
import re
import pandas as pd
import logging
import os
import sys

# Konfiguriere ausführliches Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auction_debug.log')
    ]
)
logger = logging.getLogger(__name__)

class AuctionCatalogExtractor:
    """Extraktor speziell für Auktionskataloge im Format der Autobid/Auktion&Markt AG"""

    def __init__(self):
        pass

    def extract_from_pdf(self, pdf_path):
        """
        Extrahiert Fahrzeugdaten aus einem Auktionskatalog-PDF.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            list: Liste von extrahierten Fahrzeugdaten
        """
        logger.info(f"Extrahiere Daten aus: {pdf_path}")
        vehicles = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extrahiere Text aus allen Seiten
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

                # Erkenne alle Fahrzeugeinträge im typischen Format:
                # Nr Hersteller Modell ... Preis

                # Normalisiere Zeilenumbrüche
                all_text = re.sub(r'\n+', '\n', all_text)
                lines = all_text.split('\n')

                current_vehicle = {}
                pattern = r'^\s*(\d+)\s+([A-Za-z\-]+)\s+(.+?)\s+(\d+)\s+(\d+)\s+([A-Za-z]{3})\s+(\d{2})\s+(\d+)\s+(.+?)\s+(Netto|§25a)\s+(\d+[\.,]\d+)'

                for line in lines:
                    # Suche nach Zeilen, die mit einer Fahrzeugnummer beginnen
                    match = re.match(pattern, line)
                    if match:
                        # Extrahiere die Daten
                        (nr, hersteller, modell, ps, kw, monat, jahr, km,
                         ausstattung, mwst, preis) = match.groups()

                        # Behandle Mercedes-Benz als speziellen Fall
                        if hersteller == "Mercedes":
                            if "Benz" in modell:
                                hersteller = "Mercedes-Benz"
                                modell = modell.replace("Benz", "").strip()

                        # Baue Fahrzeugdaten zusammen
                        vehicle = {
                            'nummer': nr,
                            'marke': hersteller,
                            'modell': modell.strip(),
                            'leistung': ps,
                            'leistung_kw': kw,
                            'baujahr': f"20{jahr}" if int(jahr) < 50 else f"19{jahr}",  # Annahme 2-stelliges Jahr
                            'kilometerstand': km,
                            'ausstattung': ausstattung.strip(),
                            'mwst': mwst,
                            'auktionspreis': preis.replace('.', '').replace(',', '.')
                        }

                        logger.debug(f"Fahrzeug gefunden: {vehicle['marke']} {vehicle['modell']}")
                        vehicles.append(vehicle)

                # Alternativer Ansatz für schwierigere Kataloge
                if not vehicles:
                    vehicles = self._extract_from_complex_layout(all_text)

                logger.info(f"Insgesamt {len(vehicles)} Fahrzeuge extrahiert")
                return vehicles

        except Exception as e:
            logger.error(f"Fehler bei der Extraktion: {str(e)}")
            return []

    def _extract_from_complex_layout(self, text):
        """
        Extrahiert Fahrzeugdaten aus komplexeren Layouts.

        Args:
            text (str): Extrahierter Text aus dem PDF

        Returns:
            list: Liste von extrahierten Fahrzeugdaten
        """
        vehicles = []
        lines = text.split('\n')

        # Mehrere Muster für unterschiedliche Formate probieren
        patterns = [
            # Muster 1 - Mit Hersteller/Modell getrennt
            r'^\s*(\d+)\s+([A-Za-z\-]+)\s+([A-Za-z0-9\s\-\.]+)\s+(\d+)\s+(\d+)\s+([A-Za-z]{3})\s+(\d{2})',

            # Muster 2 - Mercedes-Benz Format
            r'Mercedes-\s*Benz\s+([A-Za-z0-9\s\-\.]+)\s+(\d+)\s+(\d+)\s+([A-Za-z]{3})\s+(\d{2})',

            # Muster 3 - Einfacheres Format
            r'([A-Za-z\-]+)\s+([A-Za-z0-9\s\-\.]+)\s+(\d+)\s+(\d+)\s+([A-Za-z]{3})\s+(\d{2})'
        ]

        # Preismuster
        price_pattern = r'(\d+[\.,]\d+)\s*(?:€|EUR|Netto)'

        current_vehicle = None

        for i, line in enumerate(lines):
            # Überprüfe jedes Muster
            for pattern_idx, pattern in enumerate(patterns):
                match = re.search(pattern, line)
                if match:
                    # Wenn wir ein vorheriges Fahrzeug haben, füge es zur Liste hinzu
                    if current_vehicle and 'marke' in current_vehicle:
                        vehicles.append(current_vehicle)

                    # Starte ein neues Fahrzeug
                    current_vehicle = {}

                    # Extrahiere Daten basierend auf dem Muster
                    if pattern_idx == 0:  # Muster 1
                        nr, marke, modell, ps, kw, monat, jahr = match.groups()
                        current_vehicle['nummer'] = nr
                        current_vehicle['marke'] = marke
                        current_vehicle['modell'] = modell.strip()
                        current_vehicle['leistung'] = ps
                        current_vehicle['leistung_kw'] = kw
                        current_vehicle['baujahr'] = f"20{jahr}" if int(jahr) < 50 else f"19{jahr}"

                    elif pattern_idx == 1:  # Muster 2 - Mercedes-Benz
                        modell, ps, kw, monat, jahr = match.groups()
                        current_vehicle['marke'] = "Mercedes-Benz"
                        current_vehicle['modell'] = modell.strip()
                        current_vehicle['leistung'] = ps
                        current_vehicle['leistung_kw'] = kw
                        current_vehicle['baujahr'] = f"20{jahr}" if int(jahr) < 50 else f"19{jahr}"

                    elif pattern_idx == 2:  # Muster 3
                        marke, modell, ps, kw, monat, jahr = match.groups()
                        current_vehicle['marke'] = marke
                        current_vehicle['modell'] = modell.strip()
                        current_vehicle['leistung'] = ps
                        current_vehicle['leistung_kw'] = kw
                        current_vehicle['baujahr'] = f"20{jahr}" if int(jahr) < 50 else f"19{jahr}"

                    # Suche nach Kilometerstand in derselben oder benachbarten Zeilen
                    km_match = re.search(r'(\d+)\s+(?:km|Tkm|abgelesen)', line)
                    if km_match:
                        current_vehicle['kilometerstand'] = km_match.group(1)
                    elif i < len(lines) - 1:
                        km_match = re.search(r'(\d+)\s+(?:km|Tkm|abgelesen)', lines[i+1])
                        if km_match:
                            current_vehicle['kilometerstand'] = km_match.group(1)

                    # Suche nach Preis in derselben oder benachbarten Zeilen
                    price_match = re.search(price_pattern, line)
                    if price_match:
                        current_vehicle['auktionspreis'] = price_match.group(1).replace('.', '').replace(',', '.')
                    elif i < len(lines) - 1:
                        price_match = re.search(price_pattern, lines[i+1])
                        if price_match:
                            current_vehicle['auktionspreis'] = price_match.group(1).replace('.', '').replace(',', '.')

                    break  # Wenn ein Muster passt, keine weiteren probieren

        # Das letzte Fahrzeug auch hinzufügen
        if current_vehicle and 'marke' in current_vehicle:
            vehicles.append(current_vehicle)

        return vehicles

    def save_to_csv(self, vehicles, output_path):
        """
        Speichert extrahierte Fahrzeugdaten als CSV.

        Args:
            vehicles (list): Liste von Fahrzeugdaten
            output_path (str): Pfad zur Ausgabedatei
        """
        if not vehicles:
            logger.warning("Keine Fahrzeuge zum Speichern vorhanden.")
            return

        try:
            # Konvertiere in DataFrame und speichere als CSV
            df = pd.DataFrame(vehicles)
            df.to_csv(output_path, index=False)
            logger.info(f"{len(vehicles)} Fahrzeuge in {output_path} gespeichert.")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der CSV: {str(e)}")

    def extract_single_vehicle(self, pdf_path):
        """
        Extrahiert das erste Fahrzeug aus einem PDF (für einfaches Testen).

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            dict: Extrahierte Fahrzeugdaten oder leeres dict bei Fehler
        """
        vehicles = self.extract_from_pdf(pdf_path)
        return vehicles[0] if vehicles else {}

    def interactive_extraction(self, pdf_path):
        """
        Führt eine interaktive Extraktion durch und zeigt dem Benutzer die Ergebnisse.

        Args:
            pdf_path (str): Pfad zur PDF-Datei
        """
        print(f"\n=== Interaktive Extraktion aus {os.path.basename(pdf_path)} ===\n")

        vehicles = self.extract_from_pdf(pdf_path)

        if not vehicles:
            print("Keine Fahrzeuge gefunden!")
            return

        print(f"{len(vehicles)} Fahrzeuge gefunden.\n")

        # Zeige die ersten 5 Fahrzeuge
        for i, vehicle in enumerate(vehicles[:5]):
            print(f"Fahrzeug {i+1}:")
            for key, value in vehicle.items():
                print(f"  {key}: {value}")
            print()

        # Frage, ob alle Fahrzeuge gespeichert werden sollen
        if input("Möchten Sie alle Fahrzeuge als CSV speichern? (j/n): ").lower().startswith('j'):
            output_path = input("Bitte geben Sie den Ausgabepfad an (standard: extracted_vehicles.csv): ") or "extracted_vehicles.csv"
            self.save_to_csv(vehicles, output_path)

# Wenn direkt ausgeführt, starte das Programm
if __name__ == "__main__":
    extractor = AuctionCatalogExtractor()

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        extractor.interactive_extraction(pdf_path)
    else:
        print("Bitte geben Sie den Pfad zur PDF-Datei an.")
        print("Verwendung: python auction_catalog_extractor.py pfad/zur/datei.pdf")