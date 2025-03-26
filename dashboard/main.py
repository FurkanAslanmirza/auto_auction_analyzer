import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import time

# Pfad zum Projektroot hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importe aus anderen Modulen
from pdf_extractor.auction_catalog_extractor import AuctionCatalogExtractor
from pdf_extractor.ai_pdf_extractor import AIPdfExtractor
from scraper.mobile_de_scraper import MobileDeScraper
from data_analysis.analyzer import VehicleProfitabilityAnalyzer
from ai_integration.deepseek_client import DeepSeekClient

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'dashboard.log'))
    ]
)
logger = logging.getLogger(__name__)

# Pfade
TEMP_DIR = Path(os.path.join(os.path.dirname(__file__), 'temp'))
TEMP_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = TEMP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = TEMP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

class FahrzeugAnalyseDashboard:
    """
    Streamlit-Dashboard für die Fahrzeugdatenanalyse
    """

    def __init__(self):
        """Initialisiert das Dashboard"""
        self.setup_page()
        self.init_session_state()
        self.setup_sidebar()

        # Beide Extraktoren initialisieren
        self.pdf_extractor = AuctionCatalogExtractor()  # Regelbasierter Extraktor
        self.ai_pdf_extractor = AIPdfExtractor()        # KI-basierter Extraktor

        self.scraper = MobileDeScraper(headless=True)
        self.analyzer = VehicleProfitabilityAnalyzer()
        self.ai_client = DeepSeekClient()

    def setup_page(self):
        """Konfiguriert die Seite"""
        st.set_page_config(
            page_title="Fahrzeugdaten-Auswertungs-Tool",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("🚗 Fahrzeugdaten-Auswertungs-Tool")
        st.markdown("""
        Dieses Tool extrahiert Fahrzeugdaten aus Auktions-PDFs, vergleicht sie mit aktuellen Marktpreisen 
        von mobile.de und bietet eine fundierte Entscheidungsgrundlage für den Weiterverkauf.
        """)

    def init_session_state(self):
        """Initialisiert den Session State"""
        if 'pdf_data' not in st.session_state:
            st.session_state.pdf_data = None
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        if 'analysis_summary' not in st.session_state:
            st.session_state.analysis_summary = None
        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = None
        if 'ai_analysis' not in st.session_state:
            st.session_state.ai_analysis = None
        if 'ai_summary' not in st.session_state:
            st.session_state.ai_summary = None
        if 'show_legal_info' not in st.session_state:
            st.session_state.show_legal_info = False
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "upload"

    def setup_sidebar(self):
        """Erstellt die Sidebar für Navigation und Einstellungen"""
        st.sidebar.title("Navigation")

        # Navigationsbuttons
        if st.sidebar.button("1. Daten hochladen", use_container_width=True):
            st.session_state.active_tab = "upload"

        if st.sidebar.button("2. Marktdaten abrufen", use_container_width=True):
            st.session_state.active_tab = "market"

        if st.sidebar.button("3. Datenanalyse", use_container_width=True):
            st.session_state.active_tab = "analysis"

        if st.sidebar.button("4. KI-Empfehlungen", use_container_width=True):
            st.session_state.active_tab = "ai"

        # Einstellungen
        st.sidebar.title("Einstellungen")

        # Scraping-Einstellungen
        st.sidebar.subheader("Scraping-Einstellungen")
        st.session_state.max_scrape_pages = st.sidebar.slider(
            "Max. Seiten pro Suche",
            min_value=1, max_value=10, value=3
        )
        st.session_state.max_vehicles = st.sidebar.slider(
            "Max. Fahrzeuge pro Suche",
            min_value=5, max_value=50, value=20
        )

        # Analyseeinstellungen
        st.sidebar.subheader("Analyseeinstellungen")
        st.session_state.min_profit_margin = st.sidebar.slider(
            "Min. Gewinnmarge (%)",
            min_value=5, max_value=30, value=15
        )
        st.session_state.min_profit_amount = st.sidebar.slider(
            "Min. Gewinnbetrag (€)",
            min_value=500, max_value=5000, value=2000, step=100
        )

        # Rechtliche Informationen
        st.sidebar.title("Rechtliche Informationen")
        if st.sidebar.button("Zeige rechtliche Hinweise", use_container_width=True):
            st.session_state.show_legal_info = not st.session_state.show_legal_info

        if st.session_state.show_legal_info:
            st.sidebar.info("""
            **Rechtliche Hinweise zum Web-Scraping:**
            
            1. Das Scraping von mobile.de erfolgt unter Beachtung der robots.txt und mit angemessenen Wartezeiten.
            2. Die Daten werden ausschließlich für private, nicht-kommerzielle Analysen verwendet.
            3. Es werden keine persönlichen Daten extrahiert oder gespeichert.
            4. Das Tool speichert keine Daten auf externen Servern.
            
            **DSGVO-Hinweise:**
            
            1. Alle Daten werden ausschließlich lokal verarbeitet.
            2. Keine Daten werden an Dritte weitergegeben.
            3. Die KI-Analyse erfolgt lokal ohne Übertragung von Daten ins Internet.
            
            **Verantwortung des Nutzers:**
            
            Der Nutzer ist für die rechtskonforme Verwendung dieses Tools verantwortlich.
            """)

    def render_upload_tab(self):
        """Rendert den Upload-Tab für PDF-Dateien"""
        st.header("1. Auktions-PDFs hochladen")

        st.markdown("""
        Laden Sie Ihre Auktions-PDFs hoch, um die Fahrzeugdaten zu extrahieren.
        Das Tool erkennt automatisch Fahrzeugdaten wie Marke, Modell, Baujahr und Preis.
        """)

        uploaded_files = st.file_uploader(
            "Wählen Sie eine oder mehrere PDF-Dateien",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f"{len(uploaded_files)} PDF-Dateien wurden hochgeladen.")

            # Option zur Auswahl der Extraktionsmethode
            extraction_method = st.radio(
                "Extraktionsmethode wählen:",
                ["Regelbasiert (schneller)", "KI-basiert (genauer)"]
            )

            if st.button("PDFs verarbeiten", use_container_width=True):
                with st.spinner("Extrahiere Daten aus PDFs..."):
                    # Je nach Auswahl die passende Methode verwenden
                    if extraction_method == "KI-basiert (genauer)":
                        df = self.ai_pdf_extractor.process_auction_pdfs(uploaded_files)
                        method_used = "KI"
                    else:
                        df = self.pdf_extractor.process_auction_pdfs(uploaded_files)
                        method_used = "Regelbasiert"

                    if not df.empty:
                        st.session_state.pdf_data = df
                        st.success(f"✅ Daten aus {len(df)} Fahrzeugen erfolgreich mit {method_used}-Extraktor extrahiert!")

                        # Zeige extrahierte Daten
                        st.subheader("Extrahierte Fahrzeugdaten")
                        st.dataframe(df)

                        # Navigation zum nächsten Schritt
                        st.markdown("---")
                        st.markdown("##### Nächster Schritt: Marktdaten abrufen")
                        if st.button("Zu 'Marktdaten abrufen'", use_container_width=True):
                            st.session_state.active_tab = "market"
                    else:
                        st.error("❌ Keine Daten konnten aus den PDFs extrahiert werden.")
        else:
            st.info("Bitte laden Sie mindestens eine PDF-Datei hoch.")

        # Beispieldaten anzeigen
        with st.expander("Beispieldaten verwenden"):
            st.markdown("""
            Wenn Sie keine eigenen PDFs haben, können Sie stattdessen Beispieldaten verwenden.
            """)

            if st.button("Beispieldaten laden", use_container_width=True):
                with st.spinner("Lade Beispieldaten..."):
                    # Beispieldaten erstellen
                    example_data = {
                        'marke': ['BMW', 'Audi', 'Mercedes', 'Volkswagen', 'Porsche'],
                        'modell': ['X5', 'A6', 'C-Klasse', 'Golf', '911'],
                        'baujahr': [2018, 2019, 2017, 2020, 2016],
                        'kilometerstand': [75000, 60000, 90000, 45000, 50000],
                        'auktionspreis': [35000, 28000, 25000, 18000, 65000],
                        'kraftstoff': ['Diesel', 'Diesel', 'Diesel', 'Benzin', 'Benzin'],
                        'leistung': [265, 190, 170, 150, 385],
                        'fahrgestellnummer': ['WBA12345', 'WAU67890', 'WDD12345', 'WVW67890', 'WP012345']
                    }

                    st.session_state.pdf_data = pd.DataFrame(example_data)
                    st.success("✅ Beispieldaten erfolgreich geladen!")

                    # Zeige Beispieldaten
                    st.subheader("Beispiel-Fahrzeugdaten")
                    st.dataframe(st.session_state.pdf_data)

                    # Navigation zum nächsten Schritt
                    st.markdown("---")
                    st.markdown("##### Nächster Schritt: Marktdaten abrufen")
                    if st.button("Zu 'Marktdaten abrufen'", key="goto_market_example", use_container_width=True):
                        st.session_state.active_tab = "market"

    def render_market_tab(self):
        """Rendert den Tab für die Marktdatenabfrage"""
        st.header("2. Marktdaten abrufen")

        if st.session_state.pdf_data is None:
            st.warning("⚠️ Bitte laden Sie zuerst Auktions-PDFs hoch oder verwenden Sie Beispieldaten.")
            if st.button("Zurück zu 'Daten hochladen'", use_container_width=True):
                st.session_state.active_tab = "upload"
            return

        st.markdown("""
        In diesem Schritt werden ähnliche Fahrzeugangebote auf mobile.de gesucht,
        um aktuelle Marktpreise zu ermitteln.
        """)

        # Optionen für die Marktdatenabfrage
        st.subheader("Suchoptionen")

        # Pro Fahrzeug eine Checkbox für die Auswahl
        st.markdown("Wählen Sie die Fahrzeuge aus, für die Marktdaten abgerufen werden sollen:")

        vehicle_selection = {}
        for idx, row in st.session_state.pdf_data.iterrows():
            vehicle_name = f"{row['marke']} {row['modell']} ({row['baujahr']})"
            vehicle_selection[idx] = st.checkbox(vehicle_name, value=True)

        selected_indices = [idx for idx, selected in vehicle_selection.items() if selected]

        if not selected_indices:
            st.warning("⚠️ Bitte wählen Sie mindestens ein Fahrzeug aus.")
            return

        # Toleranzeinstellungen
        st.subheader("Toleranzeinstellungen für die Suche")

        col1, col2 = st.columns(2)

        with col1:
            year_tolerance = st.slider(
                "Baujahr-Toleranz (Jahre)",
                min_value=0, max_value=5, value=1
            )

        with col2:
            km_tolerance_percent = st.slider(
                "Kilometerstand-Toleranz (%)",
                min_value=10, max_value=50, value=20
            )

        # Marktdaten abrufen
        if st.button("Marktdaten abrufen", use_container_width=True):
            with st.spinner("Suche auf mobile.de..."):
                # Ergebnisse speichern
                all_market_data = []

                # Progress bar
                progress_bar = st.progress(0)

                # Für jedes ausgewählte Fahrzeug Marktdaten abrufen
                for i, idx in enumerate(selected_indices):
                    vehicle_row = st.session_state.pdf_data.iloc[idx]

                    st.info(f"Suche nach: {vehicle_row['marke']} {vehicle_row['modell']} ({vehicle_row['baujahr']})")

                    # Baujahr-Toleranz berechnen
                    min_year = vehicle_row['baujahr'] - year_tolerance

                    # Kilometerstand-Toleranz berechnen
                    max_km = vehicle_row['kilometerstand'] * (1 + km_tolerance_percent/100)

                    # Marktdaten abrufen
                    market_df = self.scraper.scrape_listings(
                        marke=vehicle_row['marke'],
                        modell=vehicle_row['modell'],
                        baujahr=min_year,
                        max_kilometer=max_km
                    )

                    if not market_df.empty:
                        # Fahrzeug-ID hinzufügen
                        market_df['vehicle_id'] = idx
                        all_market_data.append(market_df)
                        st.success(f"✅ {len(market_df)} Angebote für {vehicle_row['marke']} {vehicle_row['modell']} gefunden.")
                    else:
                        st.warning(f"⚠️ Keine Angebote für {vehicle_row['marke']} {vehicle_row['modell']} gefunden.")

                    # Progress bar aktualisieren
                    progress_bar.progress((i + 1) / len(selected_indices))

                # Alle Marktdaten kombinieren
                if all_market_data:
                    combined_market_data = pd.concat(all_market_data, ignore_index=True)
                    st.session_state.market_data = combined_market_data

                    # Zeige Marktdaten
                    st.subheader("Gesammelte Marktdaten")
                    st.dataframe(combined_market_data)

                    # Speichere Marktdaten
                    market_data_path = OUTPUT_DIR / "marktdaten.csv"
                    combined_market_data.to_csv(market_data_path, index=False)
                    st.markdown(f"Marktdaten gespeichert unter: `{market_data_path}`")

                    # Navigation zum nächsten Schritt
                    st.markdown("---")
                    st.markdown("##### Nächster Schritt: Datenanalyse")
                    if st.button("Zu 'Datenanalyse'", use_container_width=True):
                        st.session_state.active_tab = "analysis"
                else:
                    st.error("❌ Keine Marktdaten konnten abgerufen werden.")

        # Beispieldaten anzeigen
        with st.expander("Beispiel-Marktdaten verwenden"):
            st.markdown("""
            Wenn Sie keine echten Marktdaten abrufen möchten, können Sie stattdessen Beispiel-Marktdaten verwenden.
            """)

            if st.button("Beispiel-Marktdaten laden", use_container_width=True):
                with st.spinner("Lade Beispiel-Marktdaten..."):
                    # Beispieldaten erstellen
                    example_market_data = []

                    for idx, row in st.session_state.pdf_data.iterrows():
                        # Für jedes Fahrzeug 5-10 fiktive Marktangebote erstellen
                        num_listings = np.random.randint(5, 11)

                        for _ in range(num_listings):
                            # Zufällige Variation des Preises
                            price_variation = np.random.uniform(0.8, 1.3)
                            market_price = int(row['auktionspreis'] * price_variation)

                            # Zufällige Variation des Kilometerstands
                            mileage_variation = np.random.uniform(0.9, 1.1)
                            mileage = int(row['kilometerstand'] * mileage_variation)

                            # Jahr (± 1 Jahr)
                            year = row['baujahr'] + np.random.randint(-1, 2)

                            listing = {
                                'title': f"{row['marke']} {row['modell']}",
                                'marktpreis': market_price,
                                'baujahr': year,
                                'kilometerstand': mileage,
                                'kraftstoff': row['kraftstoff'],
                                'leistung_kw': row['leistung'],
                                'url': f"https://www.mobile.de/example/{idx}_{np.random.randint(1000, 9999)}",
                                'vehicle_id': idx
                            }

                            example_market_data.append(listing)

                    # DataFrame erstellen
                    market_df = pd.DataFrame(example_market_data)
                    st.session_state.market_data = market_df

                    st.success("✅ Beispiel-Marktdaten erfolgreich geladen!")

                    # Zeige Marktdaten
                    st.subheader("Beispiel-Marktdaten")
                    st.dataframe(market_df)

                    # Navigation zum nächsten Schritt
                    st.markdown("---")
                    st.markdown("##### Nächster Schritt: Datenanalyse")
                    if st.button("Zu 'Datenanalyse'", key="goto_analysis_example", use_container_width=True):
                        st.session_state.active_tab = "analysis"

    def render_analysis_tab(self):
        """Rendert den Tab für die Datenanalyse"""
        st.header("3. Datenanalyse")

        if st.session_state.pdf_data is None or st.session_state.market_data is None:
            st.warning("⚠️ Bitte laden Sie zuerst Fahrzeugdaten und Marktdaten.")
            if st.button("Zurück zu 'Marktdaten abrufen'", use_container_width=True):
                st.session_state.active_tab = "market"
            return

        st.markdown("""
        In diesem Schritt werden die Auktionspreise mit den Marktpreisen verglichen und die
        Profitabilität der Fahrzeuge bewertet.
        """)

        # Analyseeinstellungen aus der Sidebar übernehmen
        st.subheader("Analyseeinstellungen")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Minimale Gewinnmarge", f"{st.session_state.min_profit_margin}%")

        with col2:
            st.metric("Minimaler Gewinnbetrag", f"{st.session_state.min_profit_amount}€")

        # Analyse durchführen
        if st.button("Analyse durchführen", use_container_width=True):
            with st.spinner("Analysiere Daten..."):
                # Parameter aktualisieren
                self.analyzer.parameters['min_profit_margin_percent'] = st.session_state.min_profit_margin
                self.analyzer.parameters['min_profit_amount'] = st.session_state.min_profit_amount

                # Analyse durchführen
                analysis_df, summary, visualizations = self.analyzer.analyze(
                    st.session_state.pdf_data,
                    st.session_state.market_data
                )

                # Ergebnisse speichern
                st.session_state.analysis_data = analysis_df
                st.session_state.analysis_summary = summary
                st.session_state.visualizations = visualizations

                if analysis_df is not None and not analysis_df.empty:
                    st.success("✅ Analyse erfolgreich durchgeführt!")

                    # Tabs für verschiedene Ansichten
                    tabs = st.tabs(["Übersicht", "Detaildaten", "Visualisierungen"])

                    with tabs[0]:
                        st.subheader("Analysezusammenfassung")

                        # Metriken
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Profitable Fahrzeuge",
                                f"{summary['profitable_fahrzeuge']} von {summary['gesamtanzahl_fahrzeuge']}",
                                f"{summary['profitable_prozent']:.1f}%"
                            )

                        with col2:
                            st.metric(
                                "Durchschn. Gewinnmarge",
                                f"{summary['durchschnittliche_gewinnmarge']:.2f}%"
                            )

                        with col3:
                            st.metric(
                                "Durchschn. ROI",
                                f"{summary['durchschnittlicher_roi']:.2f}%"
                            )

                        # Bestes und schlechtestes Fahrzeug
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Profitabelstes Fahrzeug")
                            best = summary['bestes_fahrzeug']
                            st.markdown(f"**{best['marke']} {best['modell']} ({best['baujahr']})**")
                            st.markdown(f"Gewinnmarge: **{best['gewinnmarge']:.2f}%**")
                            st.markdown(f"Nettogewinn: **{best['nettogewinn']:.2f}€**")

                        with col2:
                            st.subheader("Am wenigsten profitables Fahrzeug")
                            worst = summary['schlechtestes_fahrzeug']
                            st.markdown(f"**{worst['marke']} {worst['modell']} ({worst['baujahr']})**")
                            st.markdown(f"Gewinnmarge: **{worst['gewinnmarge']:.2f}%**")
                            st.markdown(f"Nettogewinn: **{worst['nettogewinn']:.2f}€**")

                        # Profitabilitätsverteilung
                        st.subheader("Profitabilitätsverteilung")
                        prof_dist = summary['profitabilitaetsverteilung']

                        # Erstelle DataFrame für die Darstellung
                        prof_df = pd.DataFrame({
                            'Kategorie': list(prof_dist.keys()),
                            'Anzahl': list(prof_dist.values())
                        })

                        st.bar_chart(prof_df.set_index('Kategorie'))

                    with tabs[1]:
                        st.subheader("Detaillierte Analysedaten")
                        st.dataframe(analysis_df)

                        # CSV-Download
                        csv = analysis_df.to_csv(index=False)
                        st.download_button(
                            label="Analysedaten als CSV herunterladen",
                            data=csv,
                            file_name="fahrzeuganalyse.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with tabs[2]:
                        st.subheader("Visualisierungen")

                        # Zeige alle Visualisierungen
                        for viz_name, viz_path in visualizations.items():
                            st.subheader(viz_name.replace('_', ' ').title())
                            st.image(viz_path)

                    # Navigation zum nächsten Schritt
                    st.markdown("---")
                    st.markdown("##### Nächster Schritt: KI-Empfehlungen")
                    if st.button("Zu 'KI-Empfehlungen'", use_container_width=True):
                        st.session_state.active_tab = "ai"
                else:
                    st.error("❌ Analyse konnte nicht durchgeführt werden.")

        # Falls bereits eine Analyse durchgeführt wurde
        elif st.session_state.analysis_data is not None:
            st.success("✅ Analyse wurde bereits durchgeführt.")

            # Knopf zum Erneuten Anzeigen der Ergebnisse
            if st.button("Analyseergebnisse anzeigen", use_container_width=True):
                # Tabs für verschiedene Ansichten
                tabs = st.tabs(["Übersicht", "Detaildaten", "Visualisierungen"])

                with tabs[0]:
                    st.subheader("Analysezusammenfassung")

                    # Metriken
                    summary = st.session_state.analysis_summary
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Profitable Fahrzeuge",
                            f"{summary['profitable_fahrzeuge']} von {summary['gesamtanzahl_fahrzeuge']}",
                            f"{summary['profitable_prozent']:.1f}%"
                        )

                    with col2:
                        st.metric(
                            "Durchschn. Gewinnmarge",
                            f"{summary['durchschnittliche_gewinnmarge']:.2f}%"
                        )

                    with col3:
                        st.metric(
                            "Durchschn. ROI",
                            f"{summary['durchschnittlicher_roi']:.2f}%"
                        )

                    # Bestes und schlechtestes Fahrzeug
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Profitabelstes Fahrzeug")
                        best = summary['bestes_fahrzeug']
                        st.markdown(f"**{best['marke']} {best['modell']} ({best['baujahr']})**")
                        st.markdown(f"Gewinnmarge: **{best['gewinnmarge']:.2f}%**")
                        st.markdown(f"Nettogewinn: **{best['nettogewinn']:.2f}€**")

                    with col2:
                        st.subheader("Am wenigsten profitables Fahrzeug")
                        worst = summary['schlechtestes_fahrzeug']
                        st.markdown(f"**{worst['marke']} {worst['modell']} ({worst['baujahr']})**")
                        st.markdown(f"Gewinnmarge: **{worst['gewinnmarge']:.2f}%**")
                        st.markdown(f"Nettogewinn: **{worst['nettogewinn']:.2f}€**")

                    # Profitabilitätsverteilung
                    st.subheader("Profitabilitätsverteilung")
                    prof_dist = summary['profitabilitaetsverteilung']

                    # Erstelle DataFrame für die Darstellung
                    prof_df = pd.DataFrame({
                        'Kategorie': list(prof_dist.keys()),
                        'Anzahl': list(prof_dist.values())
                    })

                    st.bar_chart(prof_df.set_index('Kategorie'))

                with tabs[1]:
                    st.subheader("Detaillierte Analysedaten")
                    st.dataframe(st.session_state.analysis_data)

                    # CSV-Download
                    csv = st.session_state.analysis_data.to_csv(index=False)
                    st.download_button(
                        label="Analysedaten als CSV herunterladen",
                        data=csv,
                        file_name="fahrzeuganalyse.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with tabs[2]:
                    st.subheader("Visualisierungen")

                    # Zeige alle Visualisierungen
                    for viz_name, viz_path in st.session_state.visualizations.items():
                        st.subheader(viz_name.replace('_', ' ').title())
                        st.image(viz_path)

                # Navigation zum nächsten Schritt
                st.markdown("---")
                st.markdown("##### Nächster Schritt: KI-Empfehlungen")
                if st.button("Zu 'KI-Empfehlungen'", key="goto_ai_existing", use_container_width=True):
                    st.session_state.active_tab = "ai"

    def render_ai_tab(self):
        """Rendert den Tab für KI-Empfehlungen"""
        st.header("4. KI-Empfehlungen")

        if st.session_state.analysis_data is None:
            st.warning("⚠️ Bitte führen Sie zuerst die Datenanalyse durch.")
            if st.button("Zurück zu 'Datenanalyse'", use_container_width=True):
                st.session_state.active_tab = "analysis"
            return

        st.markdown("""
        In diesem Schritt wird das lokale KI-Modell DeepSeek-R1 verwendet, um die Rentabilität der Fahrzeuge
        zu bewerten und Handlungsempfehlungen zu generieren.
        """)

        # Systemanforderungen überprüfen
        with st.expander("Systemanforderungen für DeepSeek-R1"):
            system_req = self.ai_client.check_system_requirements()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "RAM",
                    f"{system_req.get('ram_gb', 0)} GB",
                    f"Erforderlich: {system_req.get('required_ram_gb', 0)} GB"
                )

                ram_ok = system_req.get('ram_sufficient', False)
                if ram_ok:
                    st.success("✅ RAM ausreichend")
                else:
                    st.warning("⚠️ RAM möglicherweise nicht ausreichend")

            with col2:
                if system_req.get('gpu_info'):
                    gpu = system_req.get('gpu_info')[0]
                    st.metric(
                        "GPU VRAM",
                        f"{gpu.get('memory_total_gb', 0)} GB",
                        f"Erforderlich: {system_req.get('required_gpu_vram_gb', 0)} GB"
                    )

                    gpu_ok = system_req.get('gpu_sufficient', False)
                    if gpu_ok:
                        st.success("✅ GPU ausreichend")
                    else:
                        st.warning("⚠️ GPU möglicherweise nicht ausreichend")
                else:
                    st.warning("⚠️ Keine GPU erkannt")

            overall_ok = system_req.get('overall_sufficient', False)
            if overall_ok:
                st.success("✅ System erfüllt die Anforderungen für DeepSeek-R1")
            else:
                st.warning("""
                ⚠️ System erfüllt möglicherweise nicht alle Anforderungen für DeepSeek-R1.
                Die Ausführung kann langsamer sein oder fehlschlagen.
                """)

        # Optionen für KI-Analyse
        st.subheader("KI-Analyseoptionen")

        analysis_types = [
            "Einzelanalyse pro Fahrzeug",
            "Zusammenfassende Gesamtanalyse"
        ]

        analysis_type = st.radio(
            "Wählen Sie die Art der KI-Analyse:",
            analysis_types
        )

        if analysis_type == "Einzelanalyse pro Fahrzeug":
            # Fahrzeugauswahl
            st.subheader("Fahrzeug für detaillierte Analyse auswählen")

            # Erstelle eine Liste von Fahrzeugen zur Auswahl
            vehicles = []
            for idx, row in st.session_state.analysis_data.iterrows():
                vehicle_name = f"{row['marke']} {row['modell']} ({row['baujahr']}) - {row['profitabilitaet']}"
                vehicles.append({"name": vehicle_name, "idx": idx})

            selected_vehicle = st.selectbox(
                "Fahrzeug auswählen:",
                options=range(len(vehicles)),
                format_func=lambda x: vehicles[x]["name"]
            )

            if st.button("KI-Analyse für ausgewähltes Fahrzeug starten", use_container_width=True):
                with st.spinner("DeepSeek-R1 analysiert das Fahrzeug..."):
                    selected_idx = vehicles[selected_vehicle]["idx"]
                    vehicle_data = st.session_state.analysis_data.iloc[selected_idx].to_dict()

                    # KI-Analyse durchführen
                    vehicle_analysis = self.ai_client.analyze_vehicle(vehicle_data)

                    if vehicle_analysis:
                        # Speichere die Analyse
                        if 'ai_analysis' not in st.session_state:
                            st.session_state.ai_analysis = {}

                        vehicle_id = f"{vehicle_data['marke']}_{vehicle_data['modell']}_{selected_idx}"
                        st.session_state.ai_analysis[vehicle_id] = {
                            'vehicle_data': vehicle_data,
                            'analysis': vehicle_analysis
                        }

                        st.success("✅ KI-Analyse erfolgreich durchgeführt!")

                        # Zeige das Analyseergebnis
                        st.subheader(f"KI-Analyse für {vehicle_data['marke']} {vehicle_data['modell']}")
                        st.markdown(vehicle_analysis)
                    else:
                        st.error("❌ KI-Analyse konnte nicht durchgeführt werden.")

            # Falls bereits Analysen für einzelne Fahrzeuge vorhanden sind
            elif 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
                st.info("Bereits analysierte Fahrzeuge:")

                # Erstelle eine Liste der bereits analysierten Fahrzeuge
                analyzed_vehicles = list(st.session_state.ai_analysis.keys())

                if analyzed_vehicles:
                    selected_analyzed = st.selectbox(
                        "Bereits analysierte Fahrzeuge:",
                        options=range(len(analyzed_vehicles)),
                        format_func=lambda x: analyzed_vehicles[x].replace('_', ' ')
                    )

                    # Zeige die bereits durchgeführte Analyse
                    selected_id = analyzed_vehicles[selected_analyzed]
                    analysis_data = st.session_state.ai_analysis[selected_id]

                    st.subheader(f"KI-Analyse für {selected_id.replace('_', ' ')}")
                    st.markdown(analysis_data['analysis'])

        elif analysis_type == "Zusammenfassende Gesamtanalyse":
            st.subheader("Zusammenfassende Gesamtanalyse aller Fahrzeuge")

            if st.button("Gesamtanalyse starten", use_container_width=True):
                with st.spinner("DeepSeek-R1 erstellt eine Gesamtanalyse..."):
                    # Zusammenfassungsdaten verwenden
                    summary_data = st.session_state.analysis_summary
                    prof_dist = summary_data['profitabilitaetsverteilung']

                    # KI-Analyse durchführen
                    summary_report = self.ai_client.generate_summary_report(
                        summary_data,
                        prof_dist
                    )

                    if summary_report:
                        # Speichere den Bericht
                        st.session_state.ai_summary = summary_report

                        st.success("✅ Gesamtanalyse erfolgreich durchgeführt!")

                        # Zeige das Analyseergebnis
                        st.subheader("KI-Gesamtanalyse und Empfehlungen")
                        st.markdown(summary_report)
                    else:
                        st.error("❌ Gesamtanalyse konnte nicht durchgeführt werden.")

            # Falls bereits eine Gesamtanalyse vorhanden ist
            elif 'ai_summary' in st.session_state and st.session_state.ai_summary:
                st.success("✅ Gesamtanalyse wurde bereits durchgeführt.")

                # Knopf zum Erneuten Anzeigen der Ergebnisse
                if st.button("Gesamtanalyse anzeigen", use_container_width=True):
                    st.subheader("KI-Gesamtanalyse und Empfehlungen")
                    st.markdown(st.session_state.ai_summary)

        # Exportoptionen
        st.subheader("Exportoptionen")

        exportable = False

        if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
            exportable = True

            if st.button("Einzelanalysen als PDF exportieren", use_container_width=True):
                st.info("Diese Funktion würde die Einzelanalysen als PDF exportieren.")
                st.warning("Diese Funktion ist in der Beispielimplementierung nicht vollständig implementiert.")

        if 'ai_summary' in st.session_state and st.session_state.ai_summary:
            exportable = True

            if st.button("Gesamtanalyse als PDF exportieren", use_container_width=True):
                st.info("Diese Funktion würde die Gesamtanalyse als PDF exportieren.")
                st.warning("Diese Funktion ist in der Beispielimplementierung nicht vollständig implementiert.")

        if not exportable:
            st.info("Führen Sie zuerst eine KI-Analyse durch, um Exportoptionen zu erhalten.")

    def render_active_tab(self):
        """Rendert den aktiven Tab"""
        if st.session_state.active_tab == "upload":
            self.render_upload_tab()
        elif st.session_state.active_tab == "market":
            self.render_market_tab()
        elif st.session_state.active_tab == "analysis":
            self.render_analysis_tab()
        elif st.session_state.active_tab == "ai":
            self.render_ai_tab()

    def run(self):
        """Startet das Dashboard"""
        self.render_active_tab()

if __name__ == "__main__":
    dashboard = FahrzeugAnalyseDashboard()
    dashboard.run()