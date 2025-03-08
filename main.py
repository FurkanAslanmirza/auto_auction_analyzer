# auto_auction_analyzer/main.py
import os
import logging
import argparse
from pathlib import Path
import pandas as pd

# Module importieren
from pdf_extractor.extractor import VehicleDataExtractor
from scraper.mobile_de_scraper import MobileDeScraper
from data_analysis.analyzer import VehicleProfitabilityAnalyzer
from ai_integration.deepseek_client import DeepSeekClient
import subprocess

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_auction_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

class AutoAuctionAnalyzer:
    """
    Hauptklasse für die Auto-Auktions-Analyse-Anwendung.
    Integriert alle Module in einen vollständigen Workflow.
    """

    def __init__(self, config=None):
        """
        Initialisiert die Anwendung mit den angegebenen Konfigurationen.

        Args:
            config (dict, optional): Konfigurationseinstellungen für die Anwendung
        """
        # Standardkonfiguration
        self.config = {
            'pdf_dir': './pdf_files',
            'output_dir': './output',
            'min_profit_margin': 15,
            'min_profit_amount': 2000,
            'headless_browser': True,
            'max_pages': 3,
            'max_vehicles': 20,
            'year_tolerance': 1,
            'km_tolerance_percent': 20,
            'use_ai': True
        }

        # Konfiguration überschreiben, falls angegeben
        if config:
            self.config.update(config)

        # Ausgabeverzeichnis erstellen, falls nicht vorhanden
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # Module initialisieren
        self.pdf_extractor = VehicleDataExtractor()
        self.scraper = MobileDeScraper(headless=self.config['headless_browser'])
        self.analyzer = VehicleProfitabilityAnalyzer()

        # Analyseeinstellungen anpassen
        self.analyzer.parameters['min_profit_margin_percent'] = self.config['min_profit_margin']
        self.analyzer.parameters['min_profit_amount'] = self.config['min_profit_amount']

        # KI-Client nur initialisieren, wenn benötigt
        self.ai_client = None
        if self.config['use_ai']:
            self.ai_client = DeepSeekClient()

    def run_dashboard(self):
        """Startet das Streamlit-Dashboard"""
        try:
            logger.info("Starte Streamlit-Dashboard...")
            dashboard_path = Path(__file__).parent / "dashboard" / "main.py"

            if not dashboard_path.exists():
                logger.error(f"Dashboard-Datei nicht gefunden: {dashboard_path}")
                return False

            subprocess.Popen(["streamlit", "run", str(dashboard_path)],
                             shell=False,
                             stdout=subprocess.PIPE)

            logger.info(f"Dashboard gestartet. Öffne http://localhost:8501 in deinem Browser.")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Starten des Dashboards: {str(e)}")
            return False

    def extract_from_pdfs(self):
        """
        Extrahiert Fahrzeugdaten aus PDFs im konfigurierten Verzeichnis.

        Returns:
            pd.DataFrame: DataFrame mit extrahierten Fahrzeugdaten oder None bei Fehler
        """
        try:
            logger.info(f"Extrahiere Daten aus PDFs in {self.config['pdf_dir']}...")
            pdf_data = self.pdf_extractor.process_directory(self.config['pdf_dir'])

            if pdf_data.empty:
                logger.warning("Keine Daten aus PDFs extrahiert.")
                return None

            # Speichern der extrahierten Daten
            pdf_output_path = Path(self.config['output_dir']) / "extracted_vehicles.csv"
            pdf_data.to_csv(pdf_output_path, index=False)
            logger.info(f"Extrahierte Daten gespeichert in {pdf_output_path}")

            return pdf_data

        except Exception as e:
            logger.error(f"Fehler bei der PDF-Extraktion: {str(e)}")
            return None

    def scrape_market_data(self, vehicle_df):
        """
        Scrapt Marktdaten von mobile.de für die gegebenen Fahrzeuge.

        Args:
            vehicle_df (pd.DataFrame): DataFrame mit Fahrzeugdaten

        Returns:
            pd.DataFrame: DataFrame mit Marktdaten oder None bei Fehler
        """
        if vehicle_df is None or vehicle_df.empty:
            logger.error("Keine Fahrzeugdaten für das Scraping vorhanden.")
            return None

        try:
            logger.info("Starte Scraping von mobile.de...")
            all_market_data = []

            for idx, vehicle in vehicle_df.iterrows():
                logger.info(f"Suche nach: {vehicle['marke']} {vehicle['modell']} ({vehicle['baujahr']})")

                # Baujahr-Toleranz berechnen
                min_year = vehicle['baujahr'] - self.config['year_tolerance']

                # Kilometerstand-Toleranz berechnen
                max_km = vehicle['kilometerstand'] * (1 + self.config['km_tolerance_percent']/100)

                # Marktdaten abrufen
                market_df = self.scraper.scrape_listings(
                    marke=vehicle['marke'],
                    modell=vehicle['modell'],
                    baujahr=min_year,
                    max_kilometer=max_km
                )

                if not market_df.empty:
                    # Fahrzeug-ID hinzufügen
                    market_df['vehicle_id'] = idx
                    all_market_data.append(market_df)
                    logger.info(f"✓ {len(market_df)} Angebote für {vehicle['marke']} {vehicle['modell']} gefunden.")
                else:
                    logger.warning(f"Keine Angebote für {vehicle['marke']} {vehicle['modell']} gefunden.")

            # Alle Marktdaten kombinieren
            if all_market_data:
                combined_market_data = pd.concat(all_market_data, ignore_index=True)

                # Speichern der Marktdaten
                market_output_path = Path(self.config['output_dir']) / "market_data.csv"
                combined_market_data.to_csv(market_output_path, index=False)
                logger.info(f"Marktdaten gespeichert in {market_output_path}")

                return combined_market_data
            else:
                logger.warning("Keine Marktdaten konnten abgerufen werden.")
                return None

        except Exception as e:
            logger.error(f"Fehler beim Scraping der Marktdaten: {str(e)}")
            return None

    def analyze_data(self, vehicle_df, market_df):
        """
        Analysiert die Fahrzeug- und Marktdaten.

        Args:
            vehicle_df (pd.DataFrame): DataFrame mit Fahrzeugdaten
            market_df (pd.DataFrame): DataFrame mit Marktdaten

        Returns:
            tuple: (analysis_df, summary, visualizations) oder (None, None, None) bei Fehler
        """
        if vehicle_df is None or vehicle_df.empty or market_df is None or market_df.empty:
            logger.error("Keine Daten für die Analyse vorhanden.")
            return None, None, None

        try:
            logger.info("Analysiere Daten...")
            analysis_df, summary, visualizations = self.analyzer.analyze(vehicle_df, market_df)

            if analysis_df is not None and not analysis_df.empty:
                # Speichern der Analyseergebnisse
                analysis_output_path = Path(self.config['output_dir']) / "analysis_results.csv"
                analysis_df.to_csv(analysis_output_path, index=False)
                logger.info(f"Analyseergebnisse gespeichert in {analysis_output_path}")

                return analysis_df, summary, visualizations
            else:
                logger.warning("Analyse konnte nicht durchgeführt werden.")
                return None, None, None

        except Exception as e:
            logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            return None, None, None

    def generate_ai_recommendations(self, analysis_df, summary):
        """
        Generiert KI-basierte Empfehlungen mit DeepSeek-R1.

        Args:
            analysis_df (pd.DataFrame): DataFrame mit Analyseergebnissen
            summary (dict): Zusammenfassung der Analyse

        Returns:
            tuple: (vehicle_analyses, summary_report) oder (None, None) bei Fehler
        """
        if not self.config['use_ai']:
            logger.info("KI-Analyse ist deaktiviert.")
            return None, None

        if analysis_df is None or analysis_df.empty or summary is None:
            logger.error("Keine Analysedaten für KI-Empfehlungen vorhanden.")
            return None, None

        if self.ai_client is None:
            self.ai_client = DeepSeekClient()

        try:
            # Überprüfe Systemanforderungen
            system_req = self.ai_client.check_system_requirements()
            if not system_req.get('overall_sufficient', False):
                logger.warning("System erfüllt möglicherweise nicht alle Anforderungen für DeepSeek-R1.")
                logger.warning(f"RAM: {system_req.get('ram_gb', 0)} GB (erforderlich: {system_req.get('required_ram_gb', 0)} GB)")
                for gpu in system_req.get('gpu_info', []):
                    logger.warning(f"GPU: {gpu.get('name', 'Unbekannt')} mit {gpu.get('memory_total_gb', 0)} GB VRAM")

            # Einzelanalysen für Fahrzeuge
            logger.info("Generiere KI-Analysen für einzelne Fahrzeuge...")
            vehicle_analyses = self.ai_client.analyze_all_vehicles(analysis_df)

            # Gesamtanalyse
            logger.info("Generiere zusammenfassenden KI-Bericht...")
            profitability_distribution = summary['profitabilitaetsverteilung']
            summary_report = self.ai_client.generate_summary_report(summary, profitability_distribution)

            # Speichern der KI-Empfehlungen
            ai_output_dir = Path(self.config['output_dir']) / "ai_recommendations"
            ai_output_dir.mkdir(exist_ok=True)

            # Speichere Einzelanalysen
            for vehicle_id, data in vehicle_analyses.items():
                analysis_path = ai_output_dir / f"{vehicle_id}_analysis.txt"
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    f.write(data['analysis'])

            # Speichere Gesamtanalyse
            summary_path = ai_output_dir / "summary_report.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)

            logger.info(f"KI-Empfehlungen gespeichert in {ai_output_dir}")

            return vehicle_analyses, summary_report

        except Exception as e:
            logger.error(f"Fehler bei der Generierung von KI-Empfehlungen: {str(e)}")
            return None, None

    def run_full_analysis(self):
        """
        Führt die vollständige Analyse durch - von PDF-Extraktion bis hin zu KI-Empfehlungen.

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            # 1. PDF-Extraktion
            vehicle_df = self.extract_from_pdfs()
            if vehicle_df is None:
                return False

            # 2. Market-Scraping
            market_df = self.scrape_market_data(vehicle_df)
            if market_df is None:
                return False

            # 3. Datenanalyse
            analysis_df, summary, visualizations = self.analyze_data(vehicle_df, market_df)
            if analysis_df is None:
                return False

            # 4. KI-Empfehlungen
            if self.config['use_ai']:
                vehicle_analyses, summary_report = self.generate_ai_recommendations(analysis_df, summary)

                if vehicle_analyses is None or summary_report is None:
                    logger.warning("KI-Empfehlungen konnten nicht generiert werden.")
                    # Wir kehren hier nicht zurück, da die grundlegende Analyse erfolgreich war

            logger.info("Vollständige Analyse erfolgreich abgeschlossen!")
            logger.info(f"Ergebnisse gespeichert in {self.config['output_dir']}")

            return True

        except Exception as e:
            logger.error(f"Fehler bei der vollständigen Analyse: {str(e)}")
            return False

def parse_arguments():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description='Auto Auction Analyzer')

    parser.add_argument('--pdf_dir', type=str, default='./pdf_files',
                        help='Verzeichnis mit Auktions-PDFs')

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Ausgabeverzeichnis für Ergebnisse')

    parser.add_argument('--min_profit_margin', type=float, default=15,
                        help='Minimale Gewinnmarge in Prozent')

    parser.add_argument('--min_profit_amount', type=float, default=2000,
                        help='Minimaler Gewinnbetrag in Euro')

    parser.add_argument('--no_headless', action='store_true',
                        help='Browser im sichtbaren Modus ausführen')

    parser.add_argument('--max_pages', type=int, default=3,
                        help='Maximale Anzahl an Seiten für Scraping')

    parser.add_argument('--max_vehicles', type=int, default=20,
                        help='Maximale Anzahl an Fahrzeugen pro Suche')

    parser.add_argument('--year_tolerance', type=int, default=1,
                        help='Toleranz für Baujahr in Jahren')

    parser.add_argument('--km_tolerance', type=float, default=20,
                        help='Toleranz für Kilometerstand in Prozent')

    parser.add_argument('--no_ai', action='store_true',
                        help='KI-Analyse deaktivieren')

    parser.add_argument('--dashboard', action='store_true',
                        help='Streamlit-Dashboard starten')

    parser.add_argument('--analyze', action='store_true',
                        help='Vollständige Analyse durchführen')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Konfiguration erstellen
    config = {
        'pdf_dir': args.pdf_dir,
        'output_dir': args.output_dir,
        'min_profit_margin': args.min_profit_margin,
        'min_profit_amount': args.min_profit_amount,
        'headless_browser': not args.no_headless,
        'max_pages': args.max_pages,
        'max_vehicles': args.max_vehicles,
        'year_tolerance': args.year_tolerance,
        'km_tolerance_percent': args.km_tolerance,
        'use_ai': not args.no_ai
    }

    # Anwendung initialisieren
    analyzer = AutoAuctionAnalyzer(config)

    # Aktion basierend auf Argumenten ausführen
    if args.dashboard:
        analyzer.run_dashboard()
    elif args.analyze:
        analyzer.run_full_analysis()
    else:
        print("Keine Aktion angegeben. Verwende --dashboard oder --analyze.")
        print("Für Hilfe: python main.py --help")