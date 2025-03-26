#!/usr/bin/env python3
# auto_auction_analyzer/main.py
import os
import logging
import argparse
from pathlib import Path
import pandas as pd
import subprocess
import sys

# Import fortschrittlicher Versionen der Module
from pdf_extractor.enhanced_extractor import EnhancedPdfExtractor
from market_data.enhanced_market_provider import EnhancedMarketDataProvider
from data_analysis.enhanced_analyzer import EnhancedVehicleAnalyzer
from ai_integration.deepseek_client import DeepSeekClient
from utils.config import OUTPUT_DIR, DATA_DIR

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
            'pdf_dir': str(DATA_DIR / 'pdf_files'),
            'output_dir': str(OUTPUT_DIR),
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

        # Module initialisieren mit optimierten Versionen
        self.pdf_extractor = EnhancedPdfExtractor()
        self.market_provider = EnhancedMarketDataProvider()
        self.analyzer = EnhancedVehicleAnalyzer()

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

            # Starte das Dashboard als Subprocess
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
            return self.pdf_extractor.process_directory(self.config['pdf_dir'])
        except Exception as e:
            logger.error(f"Fehler bei der PDF-Extraktion: {str(e)}")
            return None

    def get_market_data(self, vehicle_df):
        """
        Holt Marktdaten für die gegebenen Fahrzeuge.

        Args:
            vehicle_df (pd.DataFrame): DataFrame mit Fahrzeugdaten

        Returns:
            dict: Marktdaten oder None bei Fehler
        """
        if vehicle_df is None or vehicle_df.empty:
            logger.error("Keine Fahrzeugdaten für die Marktdatenabfrage vorhanden.")
            return None

        try:
            logger.info("Starte Marktdatenabfrage...")

            # Konvertiere DataFrame zu Liste von Fahrzeugdaten
            vehicles = vehicle_df.to_dict('records')

            # Marktdaten abrufen mit verbesserten Parametern
            market_data = self.market_provider.get_market_data_for_vehicles(
                vehicles=vehicles,
                force_refresh=False,  # Cache nutzen wenn verfügbar
                parallel=True  # Parallele Verarbeitung
            )

            if market_data:
                logger.info(f"Marktdaten für {len(market_data)} Fahrzeuge erfolgreich abgerufen.")

                # Speichern für spätere Verwendung
                market_data_path = Path(self.config['output_dir']) / "market_data.json"
                from utils.helpers import FileHelper
                FileHelper.save_as_json(market_data, str(market_data_path))

                return market_data
            else:
                logger.warning("Keine Marktdaten konnten abgerufen werden.")
                return None

        except Exception as e:
            logger.error(f"Fehler bei der Marktdatenabfrage: {str(e)}")
            return None

    def analyze_data(self, vehicle_df, market_data):
        """
        Analysiert die Fahrzeug- und Marktdaten.

        Args:
            vehicle_df (pd.DataFrame): DataFrame mit Fahrzeugdaten
            market_data (dict): Marktdaten, indiziert nach Fahrzeug-ID

        Returns:
            tuple: (analysis_df, summary, visualizations) oder (None, None, None) bei Fehler
        """
        if vehicle_df is None or vehicle_df.empty or market_data is None:
            logger.error("Keine Daten für die Analyse vorhanden.")
            return None, None, None

        try:
            logger.info("Analysiere Daten...")
            analysis_df, summary = self.analyzer.analyze_vehicles(vehicle_df, market_data)

            if analysis_df is not None and not analysis_df.empty:
                # Visualisierungen erstellen
                visualizations = self.analyzer.generate_visualizations(
                    analysis_df,
                    output_dir=self.config['output_dir']
                )

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
        Generiert KI-basierte Empfehlungen.

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
            # Systemanforderungen überprüfen
            system_req = self.ai_client.check_system_requirements()
            if not system_req.get('overall_sufficient', False):
                logger.warning("System erfüllt möglicherweise nicht alle Anforderungen für DeepSeek.")

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
            if vehicle_df is None or vehicle_df.empty:
                logger.error("Keine Fahrzeugdaten extrahiert.")
                return False

            # 2. Market-Daten
            market_data = self.get_market_data(vehicle_df)
            if market_data is None:
                logger.error("Keine Marktdaten abgerufen.")
                return False

            # 3. Datenanalyse
            analysis_df, summary, visualizations = self.analyze_data(vehicle_df, market_data)
            if analysis_df is None:
                logger.error("Analyse fehlgeschlagen.")
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