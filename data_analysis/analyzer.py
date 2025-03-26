# auto_auction_analyzer/data_analysis/analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import json
import datetime
from scipy import stats
import os

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehicleAnalyzer:
    """
    Optimierte Klasse für die Analyse von Fahrzeugen, Berechnung der Profitabilität
    und Erstellung von Visualisierungen.
    """

    def __init__(self):
        """Initialisiert den Analyzer mit Standard-Parametern."""
        # Parameter für die Profitabilitätsberechnung
        self.parameters = {
            'min_profit_margin_percent': 15,  # Minimale Gewinnmarge in Prozent
            'min_profit_amount': 2000,        # Minimaler Gewinn in Euro
            'very_profitable_threshold': 25,   # Schwellenwert für "sehr profitabel" in Prozent
            'marketing_costs': 500,           # Marketingkosten in Euro
            'base_renovation_cost': 500,      # Basis-Renovierungskosten in Euro
            'renovation_cost_per_year': 300,  # Renovierungskosten pro Jahr in Euro
            'renovation_cost_per_100000km': 1500,  # Renovierungskosten pro 100.000 km in Euro
            'taxes_and_fees_percent': 3,      # Steuern und Gebühren in Prozent
            'additional_costs_percent': 2,    # Zusätzliche Kosten in Prozent
            'depreciation_percent_per_year': 15,  # Wertverlust pro Jahr in Prozent
            'confidence_interval': 0.9,       # Konfidenzintervall für Preisprognosen
        }

        # Marktsegmente und ihre typischen Merkmale
        self.market_segments = {
            'Economy': {
                'description': 'Einsteigerfahrzeuge, günstig in Anschaffung und Unterhalt',
                'examples': ['Opel Corsa', 'VW Polo', 'Ford Fiesta', 'Skoda Fabia'],
                'avg_price_range': (5000, 15000),
                'depreciation_factor': 0.8,  # niedrigere Abschreibung (weniger zu verlieren)
                'seasonal_factor': 0.05  # geringere saisonale Schwankungen
            },
            'Compact': {
                'description': 'Kompakte Familien- und Alltagsfahrzeuge',
                'examples': ['VW Golf', 'Ford Focus', 'Opel Astra', 'Toyota Corolla'],
                'avg_price_range': (10000, 25000),
                'depreciation_factor': 0.9,
                'seasonal_factor': 0.07
            },
            'Mid-Size': {
                'description': 'Mittelklassewagen mit gutem Komfort',
                'examples': ['VW Passat', 'BMW 3er', 'Mercedes-Benz C-Klasse', 'Audi A4'],
                'avg_price_range': (15000, 40000),
                'depreciation_factor': 1.0,
                'seasonal_factor': 0.08
            },
            'Executive': {
                'description': 'Oberklassefahrzeuge mit Luxusausstattung',
                'examples': ['BMW 5er', 'Mercedes-Benz E-Klasse', 'Audi A6'],
                'avg_price_range': (25000, 60000),
                'depreciation_factor': 1.2,  # höhere Abschreibung
                'seasonal_factor': 0.1
            },
            'Luxury': {
                'description': 'Luxusfahrzeuge der Spitzenklasse',
                'examples': ['BMW 7er', 'Mercedes-Benz S-Klasse', 'Audi A8', 'Porsche Panamera'],
                'avg_price_range': (50000, 120000),
                'depreciation_factor': 1.4,  # sehr hohe Abschreibung
                'seasonal_factor': 0.15  # stärkere saisonale Schwankungen
            },
            'Sports': {
                'description': 'Sportfahrzeuge mit Fokus auf Performance',
                'examples': ['Porsche 911', 'BMW M-Serie', 'Mercedes-AMG', 'Audi RS'],
                'avg_price_range': (40000, 150000),
                'depreciation_factor': 1.3,
                'seasonal_factor': 0.2  # starke saisonale Schwankungen (Sommer/Winter)
            },
            'SUV': {
                'description': 'Sport Utility Vehicles verschiedener Größen',
                'examples': ['BMW X5', 'Mercedes-Benz GLE', 'Audi Q5', 'VW Tiguan'],
                'avg_price_range': (20000, 70000),
                'depreciation_factor': 1.1,
                'seasonal_factor': 0.12
            },
            'Van/MPV': {
                'description': 'Vans und Mehrzweckfahrzeuge für Familien',
                'examples': ['VW Touran', 'Mercedes-Benz B-Klasse', 'Ford S-Max'],
                'avg_price_range': (15000, 35000),
                'depreciation_factor': 1.0,
                'seasonal_factor': 0.07
            },
            'Commercial': {
                'description': 'Nutzfahrzeuge für gewerbliche Zwecke',
                'examples': ['VW Transporter', 'Mercedes-Benz Sprinter', 'Ford Transit'],
                'avg_price_range': (15000, 40000),
                'depreciation_factor': 0.85,  # niedrigere Abschreibung (Nutzwert)
                'seasonal_factor': 0.05  # geringe saisonale Schwankungen
            }
        }

        # Saisonale Faktoren (monatliche Preisindexe, 1.0 = durchschnittlicher Jahrespreis)
        self.seasonal_factors = {
            1: 0.97,  # Januar: Niedrigere Preise nach Weihnachten
            2: 0.98,  # Februar
            3: 1.0,   # März: Beginn der Frühjahrssaison
            4: 1.02,  # April
            5: 1.05,  # Mai: Hohe Nachfrage im Frühjahr
            6: 1.07,  # Juni: Sommerhoch, Urlaubszeit beginnt
            7: 1.05,  # Juli
            8: 1.03,  # August
            9: 1.0,   # September
            10: 0.98, # Oktober
            11: 0.95, # November: Niedrigere Nachfrage
            12: 0.93  # Dezember: Jahresendtief, geringe Nachfrage vor Weihnachten
        }

    def analyze_vehicles(self, vehicles_df, market_data=None):
        """
        Analysiert alle Fahrzeuge in einem DataFrame.

        Args:
            vehicles_df (pd.DataFrame): DataFrame mit Fahrzeugdaten
            market_data (dict, optional): Dictionary mit Marktdaten, indiziert nach Fahrzeug-ID

        Returns:
            tuple: (DataFrame mit Analyseergebnissen, Zusammenfassung)
        """
        if vehicles_df is None or vehicles_df.empty:
            logger.warning("Keine Fahrzeugdaten für Analyse vorhanden")
            return pd.DataFrame(), {}

        # Erstelle Kopie für die Analyse
        analysis_df = vehicles_df.copy()

        # Analysiere jedes Fahrzeug
        for idx, row in analysis_df.iterrows():
            vehicle = row.to_dict()

            # Marktdaten zuordnen, falls vorhanden
            vehicle_market_data = None
            if market_data and idx in market_data:
                vehicle_market_data = market_data[idx]

            # Profitabilität berechnen
            profitability = self.calculate_profitability(vehicle, vehicle_market_data)

            # Ergebnisse in DataFrame schreiben
            for key, value in profitability.items():
                if key not in analysis_df.columns:
                    analysis_df.loc[idx, key] = value
                else:
                    analysis_df.at[idx, key] = value

        # Zusammenfassung erstellen
        summary = self.generate_summary(analysis_df)

        return analysis_df, summary

    def identify_market_segment(self, vehicle):
        """
        Identifiziert das Marktsegment eines Fahrzeugs.

        Args:
            vehicle (dict): Fahrzeugdaten

        Returns:
            str: Name des Marktsegments
        """
        if not vehicle:
            return "Compact"  # Default

        # Extrahiere Marke und Modell
        marke = vehicle.get('marke', '').lower()
        modell = vehicle.get('modell', '').lower()
        full_name = f"{marke} {modell}".lower()

        # Suche nach exakten Übereinstimmungen in Beispielen
        for segment, data in self.market_segments.items():
            examples = [ex.lower() for ex in data.get('examples', [])]
            for example in examples:
                if example.lower() in full_name:
                    return segment

        # Heuristiken für verschiedene Segmente
        # Luxus- und Oberklassefahrzeuge
        if any(brand in marke for brand in ['mercedes', 'bmw', 'audi', 'porsche']):
            if any(high_end in modell for high_end in ['7', 's', 'a8', 'panamera']):
                return "Luxury"
            elif any(mid_high in modell for mid_high in ['5', 'e', 'a6']):
                return "Executive"
            elif any(mid in modell for mid in ['3', 'c', 'a4']):
                return "Mid-Size"
            elif any(sport in modell for sport in ['m', 'amg', 'rs']):
                return "Sports"

        # SUVs
        if any(suv in modell for suv in ['x', 'q', 'gle', 'glc', 'tiguan', 'touareg']):
            return "SUV"

        # Vans
        if any(van in modell for van in ['touran', 'sharan', 'b-klass', 'vito']):
            return "Van/MPV"

        # Nutzfahrzeuge
        if any(commercial in full_name for commercial in ['transit', 'sprinter', 'transporter', 'crafter']):
            return "Commercial"

        # Kleinwagen
        if any(small in full_name for small in ['polo', 'fiesta', 'corsa', 'fabia']):
            return "Economy"

        # Kompaktklasse (Default)
        return "Compact"

    def calculate_profitability(self, vehicle, market_data=None):
        """
        Berechnet die Profitabilität eines Fahrzeugs.

        Args:
            vehicle (dict): Fahrzeugdaten mit Auktionspreis
            market_data (dict, optional): Marktdaten oder None

        Returns:
            dict: Berechnete Profitabilitätskennzahlen
        """
        try:
            # Kopie des Fahrzeugs für die Berechnung
            vehicle_data = vehicle.copy()

            # Market-Daten integrieren, falls vorhanden
            if market_data:
                for key in ['marktpreis_median', 'marktpreis_mean', 'marktpreis_min', 'marktpreis_max', 'anzahl_angebote']:
                    if key in market_data:
                        vehicle_data[key] = market_data[key]

            # Berechnung des Alters
            current_year = datetime.datetime.now().year
            if 'baujahr' in vehicle_data:
                vehicle_data['alter_jahre'] = current_year - int(vehicle_data['baujahr'])
            else:
                vehicle_data['alter_jahre'] = 5  # Annahme: 5 Jahre alt

            # Berechnung der Renovierungskosten
            base_renovation = self.parameters['base_renovation_cost']
            age_factor = vehicle_data['alter_jahre'] * self.parameters['renovation_cost_per_year']

            # Kilometerstand-Faktor
            km_factor = 0
            if 'kilometerstand' in vehicle_data:
                km = int(vehicle_data['kilometerstand'])
                km_factor = (km / 100000) * self.parameters['renovation_cost_per_100000km']

            # Identifiziere Segment für segmentspezifische Faktoren
            segment = self.identify_market_segment(vehicle_data)
            segment_data = self.market_segments.get(segment, {})
            depreciation_factor = segment_data.get('depreciation_factor', 1.0)

            # Gesamtrenovierungskosten mit Segmentfaktor
            vehicle_data['renovierungskosten'] = (base_renovation + age_factor + km_factor) * depreciation_factor

            # Gesamtkosten (Auktionspreis + Marketingkosten + Renovierungskosten)
            auction_price = float(vehicle_data.get('auktionspreis', 0))
            vehicle_data['gesamtkosten'] = auction_price + self.parameters['marketing_costs'] + vehicle_data['renovierungskosten']

            # Verkaufspreis bestimmen (entweder aus Marktdaten oder Schätzung)
            if 'marktpreis_median' in vehicle_data and vehicle_data['marktpreis_median']:
                market_price = float(vehicle_data['marktpreis_median'])
            else:
                # Preisschätzung
                market_price = self._estimate_market_price(vehicle_data)
                vehicle_data['marktpreis_schaetzung'] = market_price

            # Steuern und Gebühren
            tax_rate = self.parameters['taxes_and_fees_percent'] / 100
            vehicle_data['steuern_gebuehren'] = market_price * tax_rate

            # Zusätzliche Kosten
            additional_rate = self.parameters['additional_costs_percent'] / 100
            vehicle_data['zusaetzliche_kosten'] = market_price * additional_rate

            # Nettogewinn berechnen
            total_costs = (
                    vehicle_data['gesamtkosten'] +
                    vehicle_data['steuern_gebuehren'] +
                    vehicle_data['zusaetzliche_kosten']
            )

            vehicle_data['nettogewinn'] = market_price - total_costs

            # Rentabilitätskennzahlen
            if auction_price > 0:
                vehicle_data['gewinnmarge_prozent'] = (vehicle_data['nettogewinn'] / auction_price) * 100
            else:
                vehicle_data['gewinnmarge_prozent'] = 0

            if total_costs > 0:
                vehicle_data['roi'] = (vehicle_data['nettogewinn'] / total_costs) * 100
            else:
                vehicle_data['roi'] = 0

            # Ampelklassifikation der Profitabilität
            min_margin = self.parameters['min_profit_margin_percent']
            min_amount = self.parameters['min_profit_amount']
            very_profitable = self.parameters['very_profitable_threshold']

            if vehicle_data['gewinnmarge_prozent'] >= very_profitable and vehicle_data['nettogewinn'] >= min_amount:
                vehicle_data['profitabilitaet'] = 'Sehr profitabel'
                vehicle_data['profitability_score'] = 5
            elif vehicle_data['gewinnmarge_prozent'] >= min_margin and vehicle_data['nettogewinn'] >= min_amount:
                vehicle_data['profitabilitaet'] = 'Profitabel'
                vehicle_data['profitability_score'] = 4
            elif vehicle_data['nettogewinn'] > 0:
                vehicle_data['profitabilitaet'] = 'Geringer Gewinn'
                vehicle_data['profitability_score'] = 3
            elif vehicle_data['nettogewinn'] > -min_amount:
                vehicle_data['profitabilitaet'] = 'Grenzwertig'
                vehicle_data['profitability_score'] = 2
            else:
                vehicle_data['profitabilitaet'] = 'Verlustgeschäft'
                vehicle_data['profitability_score'] = 1

            # Konfidenzwert berechnen (basierend auf Datenverfügbarkeit)
            confidence_factors = [
                1.0 if 'marktpreis_median' in vehicle_data and vehicle_data['marktpreis_median'] else 0.6,  # Marktdaten
                1.0 if 'anzahl_angebote' in vehicle_data and vehicle_data['anzahl_angebote'] > 5 else 0.7,  # Ausreichende Vergleichsdaten
                0.9 if 'marktpreis_schaetzung' not in vehicle_data else 0.7  # Vorhersagemethode
            ]

            vehicle_data['confidence'] = sum(confidence_factors) / len(confidence_factors)

            # Zeitliche Einschätzung
            vehicle_data['zeitpunkt_bewertung'] = datetime.datetime.now().isoformat()
            current_month = datetime.datetime.now().month
            if current_month in [3, 4, 5, 6]:  # Frühling/Frühsommer
                vehicle_data['saisonale_empfehlung'] = 'Guter Verkaufszeitraum'
            elif current_month in [11, 12, 1, 2]:  # Winter
                vehicle_data['saisonale_empfehlung'] = 'Ungünstiger Verkaufszeitraum'
            else:
                vehicle_data['saisonale_empfehlung'] = 'Durchschnittlicher Verkaufszeitraum'

            # Marktsegment und typische Käufergruppe
            vehicle_data['marktsegment'] = segment
            vehicle_data['segment_info'] = segment_data.get('description', '')

            return vehicle_data

        except Exception as e:
            logger.error(f"Fehler bei der Profitabilitätsberechnung: {str(e)}")
            return {
                'error': str(e),
                'profitabilitaet': 'Unbekannt',
                'profitability_score': 0
            }

    def _estimate_market_price(self, vehicle):
        """
        Schätzt den Marktpreis eines Fahrzeugs.

        Args:
            vehicle (dict): Fahrzeugdaten

        Returns:
            float: Geschätzter Marktpreis
        """
        try:
            # Identifiziere Segment
            segment = self.identify_market_segment(vehicle)
            segment_data = self.market_segments.get(segment, {})
            price_range = segment_data.get('avg_price_range', (10000, 30000))

            # Basispreis aus dem Bereich
            base_price = (price_range[0] + price_range[1]) / 2

            # Berücksichtige Alter
            current_year = datetime.datetime.now().year
            if 'baujahr' in vehicle:
                age = current_year - int(vehicle['baujahr'])
                # Wertverlust pro Jahr (exponentiell)
                depreciation_rate = self.parameters['depreciation_percent_per_year'] / 100
                depreciation_factor = (1 - depreciation_rate) ** age
                base_price *= depreciation_factor

            # Berücksichtige Kilometerstand
            if 'kilometerstand' in vehicle:
                km = int(vehicle['kilometerstand'])
                # Wertverlust für hohen Kilometerstand
                km_factor = max(0.7, 1 - (km / 200000) * 0.3)  # Maximal 30% Abzug
                base_price *= km_factor

            # Saisonaler Faktor
            current_month = datetime.datetime.now().month
            seasonal_factor = self.seasonal_factors.get(current_month, 1.0)
            base_price *= seasonal_factor

            # Aufschlag für Auktionspreis
            auction_price = float(vehicle.get('auktionspreis', 0))
            if auction_price > 0:
                # Typischer Aufschlag: ~20% über Auktionspreis
                if base_price < auction_price * 1.1:
                    base_price = auction_price * 1.2

            return base_price

        except Exception as e:
            logger.error(f"Fehler bei der Preisschätzung: {str(e)}")
            auction_price = float(vehicle.get('auktionspreis', 10000))
            return auction_price * 1.2  # Fallback: 20% Aufschlag

    def generate_summary(self, analysis_df):
        """
        Generiert eine Zusammenfassung der Analyseergebnisse.

        Args:
            analysis_df (pd.DataFrame): DataFrame mit Analyseergebnissen

        Returns:
            dict: Zusammenfassung der Analyse
        """
        if analysis_df is None or analysis_df.empty:
            return {}

        # Basis-Statistiken
        total_vehicles = len(analysis_df)
        profitable_vehicles = len(analysis_df[analysis_df['nettogewinn'] > 0])
        very_profitable_vehicles = len(analysis_df[
                                           (analysis_df['gewinnmarge_prozent'] >= self.parameters['very_profitable_threshold']) &
                                           (analysis_df['nettogewinn'] >= self.parameters['min_profit_amount'])
                                           ])
        unprofitable_vehicles = len(analysis_df[analysis_df['nettogewinn'] <= 0])

        # Durchschnittliche Gewinnmarge und ROI
        avg_profit_margin = analysis_df['gewinnmarge_prozent'].mean()
        avg_roi = analysis_df['roi'].mean()

        # Gesamtgewinn und -investition
        total_profit = analysis_df['nettogewinn'].sum()
        total_investment = analysis_df['gesamtkosten'].sum()
        overall_roi = (total_profit / total_investment * 100) if total_investment > 0 else 0

        # Bestes und schlechtestes Fahrzeug
        best_idx = analysis_df['gewinnmarge_prozent'].idxmax()
        worst_idx = analysis_df['gewinnmarge_prozent'].idxmin()

        best_vehicle = {
            'id': best_idx,
            'marke': analysis_df.loc[best_idx, 'marke'],
            'modell': analysis_df.loc[best_idx, 'modell'],
            'baujahr': analysis_df.loc[best_idx, 'baujahr'],
            'gewinnmarge': float(analysis_df.loc[best_idx, 'gewinnmarge_prozent']),
            'nettogewinn': float(analysis_df.loc[best_idx, 'nettogewinn']),
            'roi': float(analysis_df.loc[best_idx, 'roi']),
            'segment': analysis_df.loc[best_idx, 'marktsegment'] if 'marktsegment' in analysis_df.columns else None
        }

        worst_vehicle = {
            'id': worst_idx,
            'marke': analysis_df.loc[worst_idx, 'marke'],
            'modell': analysis_df.loc[worst_idx, 'modell'],
            'baujahr': analysis_df.loc[worst_idx, 'baujahr'],
            'gewinnmarge': float(analysis_df.loc[worst_idx, 'gewinnmarge_prozent']),
            'nettogewinn': float(analysis_df.loc[worst_idx, 'nettogewinn']),
            'roi': float(analysis_df.loc[worst_idx, 'roi']),
            'segment': analysis_df.loc[worst_idx, 'marktsegment'] if 'marktsegment' in analysis_df.columns else None
        }

        # Profitabilitätsverteilung
        if 'profitabilitaet' in analysis_df.columns:
            profitability_distribution = analysis_df['profitabilitaet'].value_counts().to_dict()
        else:
            profitability_distribution = {}

        # Segmentverteilung
        if 'marktsegment' in analysis_df.columns:
            segment_distribution = analysis_df['marktsegment'].value_counts().to_dict()

            # Durchschnittliche Gewinnmarge pro Segment
            segment_margins = {}
            for segment in segment_distribution.keys():
                segment_df = analysis_df[analysis_df['marktsegment'] == segment]
                segment_margins[segment] = segment_df['gewinnmarge_prozent'].mean()

            segment_distribution = {
                'counts': segment_distribution,
                'avg_margins': segment_margins
            }
        else:
            segment_distribution = {}

        # Saisonale Empfehlung
        current_month = datetime.datetime.now().month
        seasonal_factor = self.seasonal_factors.get(current_month, 1.0)

        if seasonal_factor > 1.02:
            seasonal_recommendation = "Aktuell günstige Verkaufssaison"
        elif seasonal_factor < 0.98:
            seasonal_recommendation = "Aktuell ungünstige Verkaufssaison"
        else:
            seasonal_recommendation = "Durchschnittliche Verkaufssaison"

        # Zusammenfassung erstellen
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
            'gesamtgewinn': total_profit,
            'gesamtinvestition': total_investment,
            'gesamt_roi': overall_roi,
            'bestes_fahrzeug': best_vehicle,
            'schlechtestes_fahrzeug': worst_vehicle,
            'profitabilitaetsverteilung': profitability_distribution,
            'segment_verteilung': segment_distribution,
            'saisonale_empfehlung': seasonal_recommendation,
            'saisonaler_faktor': seasonal_factor,
            'zeitpunkt_analyse': datetime.datetime.now().isoformat(),
            'analyseparameter': self.parameters
        }

        return summary

    def generate_visualizations(self, analysis_df, output_dir='./output'):
        """
        Generiert Visualisierungen für die Analyseergebnisse.

        Args:
            analysis_df (pd.DataFrame): DataFrame mit Analyseergebnissen
            output_dir (str): Ausgabeverzeichnis für die Visualisierungen

        Returns:
            dict: Pfade zu den erstellten Visualisierungen
        """
        if analysis_df is None or analysis_df.empty:
            logger.warning("Keine Daten für Visualisierungen vorhanden")
            return {}

        # Erstelle Ausgabeverzeichnis
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)

        visualizations = {}

        try:
            # 1. Gewinnmargen nach Fahrzeug mit Konfidenzintervall
            plt.figure(figsize=(12, 8))

            # Fahrzeugnamen erstellen
            vehicle_names = analysis_df['marke'] + ' ' + analysis_df['modell']

            # Sortiere nach Gewinnmarge absteigend
            sorted_indices = analysis_df['gewinnmarge_prozent'].sort_values(ascending=False).index
            sorted_names = vehicle_names[sorted_indices]
            sorted_margins = analysis_df.loc[sorted_indices, 'gewinnmarge_prozent']

            # Farben basierend auf Profitabilität
            colors = []
            for idx in sorted_indices:
                score = analysis_df.loc[idx, 'profitability_score']
                if score == 5:  # Sehr profitabel
                    colors.append('#2ecc71')  # Grün
                elif score == 4:  # Profitabel
                    colors.append('#27ae60')  # Dunkelgrün
                elif score == 3:  # Geringer Gewinn
                    colors.append('#f39c12')  # Orange
                elif score == 2:  # Grenzwertig
                    colors.append('#e67e22')  # Dunkles Orange
                else:  # Verlustgeschäft
                    colors.append('#c0392b')  # Rot

            # Erstelle Balkendiagramm
            bars = plt.bar(range(len(sorted_names)), sorted_margins, color=colors)

            # Schwellenwerte einzeichnen
            plt.axhline(y=self.parameters['min_profit_margin_percent'], color='r', linestyle='--',
                        label=f"Min. profitabel ({self.parameters['min_profit_margin_percent']}%)")
            plt.axhline(y=self.parameters['very_profitable_threshold'], color='g', linestyle='--',
                        label=f"Sehr profitabel ({self.parameters['very_profitable_threshold']}%)")
            plt.axhline(y=0, color='k', linestyle='-')

            # Beschriftung und Format
            plt.xlabel('Fahrzeug')
            plt.ylabel('Gewinnmarge (%)')
            plt.title('Gewinnmargen nach Fahrzeug', fontweight='bold')
            plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
            plt.tight_layout()
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Speichern
            margin_path = output_path / 'gewinnmargen.png'
            plt.savefig(margin_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['gewinnmargen'] = str(margin_path)

            # 2. Profitabilitätsverteilung als Donut-Chart
            if 'profitabilitaet' in analysis_df.columns:
                plt.figure(figsize=(10, 8))

                # Zähle die Kategorien
                prof_counts = analysis_df['profitabilitaet'].value_counts()

                # Ordne die Kategorien für bessere Darstellung
                ordered_categories = [
                    'Sehr profitabel', 'Profitabel', 'Geringer Gewinn',
                    'Grenzwertig', 'Verlustgeschäft'
                ]

                # Erstelle geordnete Serie
                ordered_counts = pd.Series([
                    prof_counts.get(cat, 0) for cat in ordered_categories
                ], index=ordered_categories)

                # Entferne leere Kategorien
                ordered_counts = ordered_counts[ordered_counts > 0]

                # Farben entsprechend der Profitabilität
                colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#c0392b']
                colors = colors[:len(ordered_counts)]

                # Erstelle Donut-Chart
                plt.pie(
                    ordered_counts,
                    labels=ordered_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    wedgeprops=dict(width=0.4, edgecolor='w')
                )

                circle = plt.Circle((0,0), 0.2, fc='white')
                plt.gca().add_artist(circle)

                plt.axis('equal')
                plt.title('Profitabilitätsverteilung', fontweight='bold')

                # Anzahl und Prozentsatz in die Mitte
                plt.text(0, 0, f"{len(analysis_df)}\nFahrzeuge",
                         horizontalalignment='center', verticalalignment='center',
                         fontweight='bold', fontsize=12)

                # Speichern
                prof_path = output_path / 'profitabilitaet_verteilung.png'
                plt.savefig(prof_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations['profitabilitaet_verteilung'] = str(prof_path)

            # 3. Auktionspreis vs. Marktpreis vs. Gesamtkosten
            plt.figure(figsize=(14, 8))

            # Erstelle x-Achse für Fahrzeuge
            x = range(len(sorted_indices))

            # Auktionspreis
            auktion = analysis_df.loc[sorted_indices, 'auktionspreis']
            plt.bar(x, auktion, width=0.25, align='center', label='Auktionspreis', alpha=0.7)

            # Gesamtkosten
            if 'gesamtkosten' in analysis_df.columns:
                kosten = analysis_df.loc[sorted_indices, 'gesamtkosten']
                plt.bar([i+0.25 for i in x], kosten, width=0.25, align='center', label='Gesamtkosten', alpha=0.7)

            # Marktpreis
            if 'marktpreis_median' in analysis_df.columns:
                markt = analysis_df.loc[sorted_indices, 'marktpreis_median']
                plt.bar([i+0.5 for i in x], markt, width=0.25, align='center', label='Median Marktpreis', alpha=0.7)

            # Beschriftung und Format
            plt.xlabel('Fahrzeug')
            plt.ylabel('Preis (€)')
            plt.title('Auktionspreise vs. Gesamtkosten vs. Marktpreise', fontweight='bold')
            plt.xticks([i+0.25 for i in x], sorted_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Speichern
            price_path = output_path / 'preisvergleich.png'
            plt.savefig(price_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['preisvergleich'] = str(price_path)

            # 4. ROI nach Fahrzeug
            plt.figure(figsize=(12, 8))

            # ROI-Werte
            roi_values = analysis_df.loc[sorted_indices, 'roi']

            # Balkendiagramm mit gleichen Farben wie bei Gewinnmarge
            plt.bar(range(len(sorted_names)), roi_values, color=colors)

            # Break-even-Linie
            plt.axhline(y=0, color='r', linestyle='--', label='Break-even')

            # Ziel-ROI-Linie
            target_roi = 15  # Beispiel für Ziel-ROI
            plt.axhline(y=target_roi, color='g', linestyle='--', label=f'Ziel-ROI ({target_roi}%)')

            # Beschriftung und Format
            plt.xlabel('Fahrzeug')
            plt.ylabel('ROI (%)')
            plt.title('Return on Investment (ROI) nach Fahrzeug', fontweight='bold')
            plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()

            # Speichern
            roi_path = output_path / 'roi.png'
            plt.savefig(roi_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['roi'] = str(roi_path)

            # 5. Marktsegmentanalyse, falls vorhanden
            if 'marktsegment' in analysis_df.columns:
                plt.figure(figsize=(12, 8))

                # Gruppieren nach Segment
                segment_profits = analysis_df.groupby('marktsegment')['nettogewinn'].sum()
                segment_counts = analysis_df['marktsegment'].value_counts()

                # Nur Segmente mit Daten
                segments_with_data = segment_profits.index

                # Erstelle ein neues DataFrame für das Diagramm
                plot_data = pd.DataFrame({
                    'Gesamtgewinn': segment_profits,
                    'Anzahl': segment_counts,
                    'Durchschnittsgewinn': segment_profits / segment_counts
                })

                # Sortiere nach Durchschnittsgewinn
                plot_data = plot_data.sort_values('Durchschnittsgewinn', ascending=False)

                # Farben basierend auf Durchschnittsgewinn
                segment_colors = plt.cm.coolwarm(np.linspace(0, 1, len(plot_data)))

                # Doppelachsendiagramm
                ax1 = plt.gca()
                ax2 = ax1.twinx()

                # Balken für Durchschnittsgewinn
                bars = ax1.bar(plot_data.index, plot_data['Durchschnittsgewinn'], color=segment_colors, alpha=0.7)

                # Linien für Anzahl der Fahrzeuge
                line = ax2.plot(plot_data.index, plot_data['Anzahl'], 'o-', color='#2c3e50', linewidth=2, markersize=8)

                # Beschriftung
                ax1.set_xlabel('Marktsegment')
                ax1.set_ylabel('Durchschnittlicher Gewinn (€)', color='#c0392b')
                ax2.set_ylabel('Anzahl Fahrzeuge', color='#2c3e50')

                # Farbanpassungen
                ax1.tick_params(axis='y', colors='#c0392b')
                ax2.tick_params(axis='y', colors='#2c3e50')

                # Werte anzeigen
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                             f'{int(height)}€',
                             ha='center', va='bottom', rotation=0,
                             fontweight='bold', color='#c0392b')

                plt.title('Profitabilität nach Marktsegment', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                # Speichern
                segment_path = output_path / 'segmentanalyse.png'
                plt.savefig(segment_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations['segmentanalyse'] = str(segment_path)

            # 6. Saisonale Analyse mit aktueller Position
            plt.figure(figsize=(12, 6))

            # Monate
            months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']

            # Saisonale Faktoren
            seasonal_values = [self.seasonal_factors[m] for m in months]

            # Liniendiagramm
            plt.plot(months, seasonal_values, 'o-', linewidth=2, markersize=8, color='#3498db')

            # Aktuellen Monat markieren
            current_month = datetime.datetime.now().month
            current_factor = self.seasonal_factors[current_month]
            plt.plot(current_month, current_factor, 'o', markersize=12, color='#e74c3c')

            # Texte hinzufügen
            for i, (m, v) in enumerate(zip(months, seasonal_values)):
                if m == current_month:
                    plt.text(m, v + 0.02, f"Aktuell: {v:.2f}", ha='center', fontweight='bold', color='#e74c3c')
                else:
                    plt.text(m, v + 0.01, f"{v:.2f}", ha='center')

            # Durchschnittslinie
            plt.axhline(y=1.0, color='#7f8c8d', linestyle='--', label='Durchschnitt')

            # Beschriftung und Format
            plt.xlabel('Monat')
            plt.ylabel('Preisindex (1.0 = Durchschnitt)')
            plt.title('Saisonale Preisschwankungen im Jahresverlauf', fontweight='bold')
            plt.xticks(months, month_names)
            plt.ylim(0.9, 1.15)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Speichern
            seasonal_path = output_path / 'saisonale_analyse.png'
            plt.savefig(seasonal_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['saisonale_analyse'] = str(seasonal_path)

            logger.info(f"Visualisierungen erfolgreich erstellt: {visualizations}")
            return visualizations

        except Exception as e:
            logger.error(f"Fehler bei der Erstellung von Visualisierungen: {str(e)}")
            return visualizations

    def analyze(self, vehicles_df, market_data=None):
        """
        Führt eine vollständige Analyse durch.

        Args:
            vehicles_df (pd.DataFrame): DataFrame mit Fahrzeugdaten
            market_data (dict, optional): Dictionary mit Marktdaten, indiziert nach Fahrzeug-ID

        Returns:
            tuple: (analysis_df, summary, visualizations)
        """
        # Analysiere Fahrzeuge
        analysis_df, summary = self.analyze_vehicles(vehicles_df, market_data)

        if analysis_df.empty:
            logger.warning("Keine Analyseergebnisse vorhanden")
            return None, None, None

        # Erstelle Visualisierungen
        visualizations = self.generate_visualizations(analysis_df)

        return analysis_df, summary, visualizations