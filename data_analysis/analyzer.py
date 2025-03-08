# auto_auction_analyzer/data_analysis/analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehicleProfitabilityAnalyzer:
    """
    Analysiert die Profitabilität von Fahrzeugen basierend auf Auktionspreisen und Marktpreisen.
    """

    def __init__(self):
        """Initialisiert den Analyzer."""
        # Parameter für die Profitabilitätsberechnung
        self.parameters = {
            'min_profit_margin_percent': 15,  # Minimale Gewinnmarge in Prozent
            'min_profit_amount': 2000,        # Minimaler Gewinn in Euro
            'very_profitable_threshold': 25,   # Schwellenwert für "sehr profitabel" in Prozent
            'marketing_costs': 500,           # Marketingkosten in Euro
            'renovation_cost_per_year': 300,  # Renovierungskosten pro Jahr in Euro
            'taxes_and_fees_percent': 3,      # Steuern und Gebühren in Prozent
        }

    def merge_auction_and_market_data(self, auction_df, market_df):
        """
        Führt die Auktionsdaten und Marktdaten zusammen.

        Args:
            auction_df (pd.DataFrame): DataFrame mit den Auktionsdaten
            market_df (pd.DataFrame): DataFrame mit den Marktdaten

        Returns:
            pd.DataFrame: Zusammengeführter DataFrame
        """
        if auction_df.empty or market_df.empty:
            logger.warning("Auktions- oder Marktdaten sind leer.")
            return pd.DataFrame()

        # Erstelle Kopien, um die Originaldaten nicht zu verändern
        auction_data = auction_df.copy()
        market_data = market_df.copy()

        # Berechne durchschnittliche Marktpreise pro Marke/Modell/Baujahr-Kombination
        market_summary = market_data.groupby(['title', 'baujahr']).agg({
            'marktpreis': ['mean', 'median', 'min', 'max', 'count']
        }).reset_index()

        # Flatten the multi-index columns
        market_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in market_summary.columns.values]

        # Erstelle für jedes Auktionsfahrzeug ein neues DataFrame mit den berechneten Marktdaten
        results = []

        for _, auction_row in auction_data.iterrows():
            # Finde passende Marktdaten
            # Zunächst versuchen wir, eine exakte Übereinstimmung zu finden
            match_title = f"{auction_row['marke']} {auction_row['modell']}"
            match_year = auction_row['baujahr']

            # Suche nach ähnlichen Fahrzeugen im Markt-Summary
            matched_market = market_summary[
                (market_summary['title_'].str.contains(auction_row['marke'], case=False)) &
                (market_summary['baujahr_'] == match_year)
                ]

            # Wenn keine genaue Übereinstimmung gefunden wurde, suche nach ähnlichen Fahrzeugen
            if matched_market.empty:
                matched_market = market_summary[
                    market_summary['title_'].str.contains(auction_row['marke'], case=False)
                ]

            # Wenn immer noch keine Übereinstimmung gefunden wurde, verwende den Durchschnitt
            if matched_market.empty:
                logger.warning(f"Keine Marktdaten für {match_title}, Baujahr {match_year} gefunden.")
                auction_row_dict = auction_row.to_dict()
                auction_row_dict.update({
                    'marktpreis_mean': None,
                    'marktpreis_median': None,
                    'marktpreis_min': None,
                    'marktpreis_max': None,
                    'anzahl_angebote': 0
                })
                results.append(auction_row_dict)
                continue

            # Beste Übereinstimmung auswählen (die mit den meisten Angeboten)
            best_match = matched_market.sort_values('marktpreis_count', ascending=False).iloc[0]

            # Kombiniere Auktions- und Marktdaten
            auction_row_dict = auction_row.to_dict()
            auction_row_dict.update({
                'marktpreis_mean': best_match['marktpreis_mean'],
                'marktpreis_median': best_match['marktpreis_median'],
                'marktpreis_min': best_match['marktpreis_min'],
                'marktpreis_max': best_match['marktpreis_max'],
                'anzahl_angebote': best_match['marktpreis_count']
            })

            results.append(auction_row_dict)

        # Erstelle DataFrame aus den Ergebnissen
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()

    def calculate_profitability(self, merged_df):
        """
        Berechnet die Profitabilität für jedes Fahrzeug.

        Args:
            merged_df (pd.DataFrame): DataFrame mit zusammengeführten Auktions- und Marktdaten

        Returns:
            pd.DataFrame: DataFrame mit hinzugefügten Profitabilitätsmetriken
        """
        if merged_df.empty:
            logger.warning("Keine Daten für die Profitabilitätsberechnung vorhanden.")
            return pd.DataFrame()

        # Erstelle Kopie der Daten
        df = merged_df.copy()

        # Berechne das Alter der Fahrzeuge in Jahren (aktuelles Jahr - Baujahr)
        current_year = pd.Timestamp.now().year
        df['alter_jahre'] = current_year - df['baujahr']

        # Berechne die Renovierungskosten basierend auf dem Alter
        df['renovierungskosten'] = df['alter_jahre'] * self.parameters['renovation_cost_per_year']

        # Berechne die Gesamtkosten (Auktionspreis + Marketingkosten + Renovierungskosten)
        df['gesamtkosten'] = df['auktionspreis'] + self.parameters['marketing_costs'] + df['renovierungskosten']

        # Berechne die Steuern und Gebühren
        df['steuern_gebuehren'] = df['marktpreis_median'] * (self.parameters['taxes_and_fees_percent'] / 100)

        # Berechne den Nettogewinn (Marktpreis - Gesamtkosten - Steuern und Gebühren)
        df['nettogewinn'] = df['marktpreis_median'] - df['gesamtkosten'] - df['steuern_gebuehren']

        # Berechne die Gewinnmarge in Prozent
        df['gewinnmarge_prozent'] = (df['nettogewinn'] / df['auktionspreis']) * 100

        # Bewerte die Profitabilität
        conditions = [
            (df['gewinnmarge_prozent'] >= self.parameters['very_profitable_threshold']),
            (df['gewinnmarge_prozent'] >= self.parameters['min_profit_margin_percent']) &
            (df['nettogewinn'] >= self.parameters['min_profit_amount']),
            (df['nettogewinn'] > 0),
            (df['nettogewinn'] <= 0)
        ]

        choices = ['Sehr profitabel', 'Profitabel', 'Geringer Gewinn', 'Verlustgeschäft']

        df['profitabilitaet'] = pd.Series(np.select(conditions, choices, default='Unbekannt'), index=df.index)

        # ROI (Return on Investment) berechnen
        df['roi'] = df['nettogewinn'] / df['gesamtkosten'] * 100

        return df

    def generate_visualizations(self, df, output_dir='./output'):
        """
        Generiert Visualisierungen für die Profitabilitätsanalyse.

        Args:
            df (pd.DataFrame): DataFrame mit Profitabilitätsdaten
            output_dir (str): Ausgabeverzeichnis für die Visualisierungen

        Returns:
            dict: Dictionary mit Pfaden zu den generierten Visualisierungen
        """
        if df.empty:
            logger.warning("Keine Daten für Visualisierungen vorhanden.")
            return {}

        # Erstelle Ausgabeverzeichnis, falls es nicht existiert
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        visualizations = {}

        # 1. Gewinnmargen nach Fahrzeug
        plt.figure(figsize=(12, 6))
        sns.barplot(x=df['marke'] + ' ' + df['modell'], y='gewinnmarge_prozent', data=df)
        plt.title('Gewinnmargen nach Fahrzeug')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Gewinnmarge (%)')
        plt.axhline(y=self.parameters['min_profit_margin_percent'], color='r', linestyle='--',
                    label=f'Min. profitabel ({self.parameters["min_profit_margin_percent"]}%)')
        plt.axhline(y=self.parameters['very_profitable_threshold'], color='g', linestyle='--',
                    label=f'Sehr profitabel ({self.parameters["very_profitable_threshold"]}%)')
        plt.tight_layout()
        plt.legend()

        profit_margins_path = output_path / 'gewinnmargen.png'
        plt.savefig(profit_margins_path)
        plt.close()
        visualizations['gewinnmargen'] = str(profit_margins_path)

        # 2. Verteilung der Profitabilitätskategorien
        plt.figure(figsize=(10, 6))
        prof_counts = df['profitabilitaet'].value_counts()
        plt.pie(prof_counts, labels=prof_counts.index, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Verteilung der Profitabilitätskategorien')

        profitability_dist_path = output_path / 'profitabilitaet_verteilung.png'
        plt.savefig(profitability_dist_path)
        plt.close()
        visualizations['profitabilitaet_verteilung'] = str(profitability_dist_path)

        # 3. Marktpreise vs. Auktionspreise
        plt.figure(figsize=(12, 6))
        x = range(len(df))
        plt.bar(x, df['auktionspreis'], width=0.4, align='edge', label='Auktionspreis', alpha=0.7)
        plt.bar([i+0.4 for i in x], df['marktpreis_median'], width=0.4, align='edge',
                label='Median Marktpreis', alpha=0.7)
        plt.xticks([i+0.4 for i in x], df['marke'] + ' ' + df['modell'], rotation=45, ha='right')
        plt.title('Auktionspreise vs. Marktpreise')
        plt.ylabel('Preis (€)')
        plt.legend()
        plt.tight_layout()

        price_comparison_path = output_path / 'preisvergleich.png'
        plt.savefig(price_comparison_path)
        plt.close()
        visualizations['preisvergleich'] = str(price_comparison_path)

        # 4. ROI nach Fahrzeug
        plt.figure(figsize=(12, 6))
        sns.barplot(x=df['marke'] + ' ' + df['modell'], y='roi', data=df)
        plt.title('Return on Investment (ROI) nach Fahrzeug')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('ROI (%)')
        plt.axhline(y=0, color='r', linestyle='--', label='Break-even')
        plt.tight_layout()
        plt.legend()

        roi_path = output_path / 'roi.png'
        plt.savefig(roi_path)
        plt.close()
        visualizations['roi'] = str(roi_path)

        logger.info(f"Visualisierungen generiert: {visualizations}")
        return visualizations

    def generate_summary_report(self, df):
        """
        Generiert einen zusammenfassenden Bericht der Profitabilitätsanalyse.

        Args:
            df (pd.DataFrame): DataFrame mit Profitabilitätsdaten

        Returns:
            dict: Dictionary mit Zusammenfassungsdaten
        """
        if df.empty:
            logger.warning("Keine Daten für den Zusammenfassungsbericht vorhanden.")
            return {}

        # Allgemeine Statistiken
        total_vehicles = len(df)
        profitable_vehicles = len(df[df['nettogewinn'] > 0])
        very_profitable_vehicles = len(df[
                                           (df['gewinnmarge_prozent'] >= self.parameters['very_profitable_threshold']) &
                                           (df['nettogewinn'] >= self.parameters['min_profit_amount'])
                                           ])
        unprofitable_vehicles = len(df[df['nettogewinn'] <= 0])

        # Durchschnittliche Gewinnmarge und ROI
        avg_profit_margin = df['gewinnmarge_prozent'].mean()
        avg_roi = df['roi'].mean()

        # Bestes und schlechtestes Fahrzeug
        best_vehicle_idx = df['gewinnmarge_prozent'].idxmax()
        worst_vehicle_idx = df['gewinnmarge_prozent'].idxmin()

        best_vehicle = {
            'marke': df.loc[best_vehicle_idx, 'marke'],
            'modell': df.loc[best_vehicle_idx, 'modell'],
            'baujahr': df.loc[best_vehicle_idx, 'baujahr'],
            'gewinnmarge': df.loc[best_vehicle_idx, 'gewinnmarge_prozent'],
            'nettogewinn': df.loc[best_vehicle_idx, 'nettogewinn']
        }

        worst_vehicle = {
            'marke': df.loc[worst_vehicle_idx, 'marke'],
            'modell': df.loc[worst_vehicle_idx, 'modell'],
            'baujahr': df.loc[worst_vehicle_idx, 'baujahr'],
            'gewinnmarge': df.loc[worst_vehicle_idx, 'gewinnmarge_prozent'],
            'nettogewinn': df.loc[worst_vehicle_idx, 'nettogewinn']
        }

        # Profitabilitätsverteilung
        profitability_distribution = df['profitabilitaet'].value_counts().to_dict()

        # Zusammenfassenden Bericht erstellen
        summary = {
            'gesamtanzahl_fahrzeuge': total_vehicles,
            'profitable_fahrzeuge': profitable_vehicles,
            'profitable_prozent': (profitable_vehicles / total_vehicles) * 100 if total_vehicles > 0 else 0,
            'sehr_profitable_fahrzeuge': very_profitable_vehicles,
            'sehr_profitable_prozent': (very_profitable_vehicles / total_vehicles) * 100 if total_vehicles > 0 else 0,
            'unprofitable_fahrzeuge': unprofitable_vehicles,
            'unprofitable_prozent': (unprofitable_vehicles / total_vehicles) * 100 if total_vehicles > 0 else 0,
            'durchschnittliche_gewinnmarge': avg_profit_margin,
            'durchschnittlicher_roi': avg_roi,
            'bestes_fahrzeug': best_vehicle,
            'schlechtestes_fahrzeug': worst_vehicle,
            'profitabilitaetsverteilung': profitability_distribution
        }

        logger.info(f"Zusammenfassender Bericht generiert.")
        return summary

    def analyze(self, auction_df, market_df):
        """
        Führt eine vollständige Analyse durch.

        Args:
            auction_df (pd.DataFrame): DataFrame mit Auktionsdaten
            market_df (pd.DataFrame): DataFrame mit Marktdaten

        Returns:
            tuple: (analysierte_daten, zusammenfassung, visualisierungen)
        """
        # Daten zusammenführen
        merged_data = self.merge_auction_and_market_data(auction_df, market_df)

        if merged_data.empty:
            logger.warning("Keine zusammengeführten Daten vorhanden. Analyse wird abgebrochen.")
            return None, None, None

        # Profitabilität berechnen
        analysis_df = self.calculate_profitability(merged_data)

        # Visualisierungen erstellen
        visualizations = self.generate_visualizations(analysis_df)

        # Zusammenfassenden Bericht erstellen
        summary = self.generate_summary_report(analysis_df)

        return analysis_df, summary, visualizations

# Beispielverwendung
if __name__ == "__main__":
    # Laden von Beispieldaten
    try:
        auction_data = pd.read_csv("extrahierte_fahrzeugdaten.csv")
        market_data = pd.read_csv("mobile_de_listings.csv")

        analyzer = VehicleProfitabilityAnalyzer()
        analysis_df, summary, visualizations = analyzer.analyze(auction_data, market_data)

        if analysis_df is not None:
            print("Analyse erfolgreich durchgeführt.")
            print(f"Analysierte Daten: {len(analysis_df)} Fahrzeuge")
            print(f"Zusammenfassung: {summary}")
            print(f"Visualisierungen gespeichert unter: {visualizations}")

            # Speichern der Analyseergebnisse
            analysis_df.to_csv("fahrzeug_analyse.csv", index=False)
        else:
            print("Analyse konnte nicht durchgeführt werden.")

    except Exception as e:
        print(f"Fehler bei der Beispielverwendung: {str(e)}")