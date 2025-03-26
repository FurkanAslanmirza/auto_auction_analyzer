# auto_auction_analyzer/market_data/market_provider.py
import os
import time
import random
import json
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketProviderConfig:
    """Konfiguration für den Market Data Provider"""
    # Allgemeine Einstellungen
    CACHE_DIR = "cache/market_data"
    CACHE_EXPIRY_DAYS = 7  # Erhöht auf 7 Tage für bessere Datenbeständigkeit

    # Scraper-Einstellungen
    BASE_URL = "https://www.mobile.de/de/fahrzeug/search.html"
    MAX_PAGES = 3
    MAX_VEHICLES = 20
    MIN_WAIT_SECONDS = 2
    MAX_WAIT_SECONDS = 5
    HEADLESS_BROWSER = True
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    # Parallele Verarbeitung
    MAX_CONCURRENT_REQUESTS = 2

class MarketDataProvider:
    """
    Optimierte Klasse für die Beschaffung von Fahrzeugmarktdaten mit fortschrittlichen Features:
    - Robustes Caching
    - Fallback-Mechanismen
    - Proxy-Rotation
    - Smarte Wiederholungsversuche
    """

    def __init__(self, config=None):
        """
        Initialisiert den Marktdaten-Provider.

        Args:
            config (MarketProviderConfig, optional): Konfiguration für den Provider
        """
        self.config = config or MarketProviderConfig()

        # Stelle sicher, dass das Cache-Verzeichnis existiert
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

        # Serien-Mapping für Fahrzeuge
        self._initialize_series_mapping()

        # WebDriver
        self.driver = None

        # Request-Session für verbesserte Effizienz
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.USER_AGENT,
            'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def _initialize_series_mapping(self):
        """Initialisiert das Mapping zwischen Marken, Serien und Modellen"""
        self.series_mapping = {
            "bmw": {
                "1er": ["114", "116", "118", "120", "123", "125", "128", "130", "135", "1er"],
                "2er": ["214", "216", "218", "220", "225", "228", "230", "235", "240", "2er"],
                "3er": ["315", "316", "318", "320", "323", "325", "328", "330", "335", "340", "3er"],
                "4er": ["418", "420", "425", "428", "430", "435", "440", "4er"],
                "5er": ["518", "520", "523", "525", "528", "530", "535", "540", "545", "550", "5er"],
                "6er": ["620", "630", "635", "640", "645", "650", "6er"],
                "7er": ["725", "728", "730", "735", "740", "745", "750", "760", "7er"],
                "8er": ["840", "850", "8er"],
                "X-Reihe": ["X1", "X2", "X3", "X4", "X5", "X6", "X7"],
                "Z-Reihe": ["Z1", "Z3", "Z4", "Z8"],
                "i-Reihe": ["i3", "i4", "i8", "iX", "iX1", "iX3", "iX5"],
                "M-Modelle": ["M2", "M3", "M4", "M5", "M6", "M8", "X3M", "X4M", "X5M", "X6M"]
            },
            "mercedes-benz": {
                "A-Klasse": ["A140", "A150", "A160", "A170", "A180", "A200", "A220", "A250", "A35", "A45"],
                "B-Klasse": ["B150", "B160", "B170", "B180", "B200", "B220", "B250"],
                "C-Klasse": ["C160", "C180", "C200", "C220", "C230", "C240", "C250", "C280", "C300", "C320", "C350", "C400", "C450"],
                "E-Klasse": ["E200", "E220", "E230", "E240", "E250", "E260", "E270", "E280", "E300", "E320", "E350", "E400", "E420", "E450", "E500", "E550"],
                "S-Klasse": ["S280", "S300", "S320", "S350", "S400", "S420", "S450", "S500", "S550", "S560", "S580", "S600", "S650"],
                "G-Klasse": ["G320", "G350", "G400", "G500", "G55", "G63", "G65"],
                "GLC": ["GLC200", "GLC220", "GLC250", "GLC300", "GLC350", "GLC400", "GLC43", "GLC63"],
                "GLE": ["GLE250", "GLE300", "GLE350", "GLE400", "GLE450", "GLE500", "GLE53", "GLE63"],
                "GLS": ["GLS350", "GLS400", "GLS450", "GLS500", "GLS580", "GLS63"],
                "AMG": ["A45 AMG", "C43 AMG", "C63 AMG", "E53 AMG", "E63 AMG", "S63 AMG", "S65 AMG"]
            },
            "audi": {
                "A1": ["A1"],
                "A3": ["A3"],
                "A4": ["A4"],
                "A5": ["A5"],
                "A6": ["A6"],
                "A7": ["A7"],
                "A8": ["A8"],
                "Q-Reihe": ["Q2", "Q3", "Q5", "Q7", "Q8"],
                "TT": ["TT"],
                "R8": ["R8"],
                "e-tron": ["e-tron", "Q4 e-tron", "e-tron GT"],
                "S-Modelle": ["S1", "S3", "S4", "S5", "S6", "S7", "S8", "SQ5", "SQ7", "SQ8"],
                "RS-Modelle": ["RS3", "RS4", "RS5", "RS6", "RS7", "RS Q3", "RS Q8"]
            },
            "volkswagen": {
                "Polo": ["Polo"],
                "Golf": ["Golf", "Golf Plus", "Golf Sportsvan"],
                "ID": ["ID.3", "ID.4", "ID.5", "ID.6", "ID.7", "ID.Buzz"],
                "Passat": ["Passat"],
                "Arteon": ["Arteon"],
                "Tiguan": ["Tiguan"],
                "Touareg": ["Touareg"],
                "T-Reihe": ["T-Cross", "T-Roc", "Touran", "Taigo"],
                "Transporter": ["T5", "T6", "T6.1", "Multivan", "Caravelle", "California"],
                "Sharan": ["Sharan"],
                "Caddy": ["Caddy"],
                "up!": ["up!"]
            }
        }

    def get_market_data(self, vehicle_data, force_refresh=False):
        """
        Holt Marktdaten für ein Fahrzeug mit optimaler Strategie.

        Args:
            vehicle_data (dict): Fahrzeugdaten
            force_refresh (bool): Cache umgehen und frische Daten holen

        Returns:
            dict: Marktdaten mit Statistiken und Listings
        """
        logger.info(f"Hole Marktdaten für {vehicle_data.get('marke', 'Unbekannt')} {vehicle_data.get('modell', 'Unbekannt')}")

        # Normalisiere Fahrzeugdaten für konsistente Suche
        search_vehicle = self._normalize_vehicle_data(vehicle_data)

        # Prüfe Cache, wenn nicht erzwungen wird
        if not force_refresh:
            cached_data = self._load_from_cache(search_vehicle)
            if cached_data:
                logger.info("Verwende gecachte Marktdaten.")
                return cached_data

        # Scraper als Hauptmethode
        try:
            market_data = self._scrape_listings(search_vehicle)
            if market_data and market_data.get('listings') and len(market_data.get('listings')) > 0:
                logger.info(f"Marktdaten über Scraper erhalten: {len(market_data['listings'])} Listings")
                self._save_to_cache(search_vehicle, market_data)
                return market_data
            else:
                logger.warning("Keine Daten über Scraper erhalten.")

                # Wenn keine Ergebnisse gefunden wurden, versuche mit vereinfachten Suchparametern
                logger.info("Versuche vereinfachte Suche...")
                simplified_vehicle = self._simplify_search_parameters(search_vehicle)

                if simplified_vehicle != search_vehicle:
                    market_data = self._scrape_listings(simplified_vehicle)
                    if market_data and market_data.get('listings') and len(market_data.get('listings')) > 0:
                        logger.info(f"Marktdaten mit vereinfachter Suche erhalten: {len(market_data['listings'])} Listings")
                        self._save_to_cache(search_vehicle, market_data)  # Original-Fahrzeug für Cache
                        return market_data
        except Exception as e:
            logger.error(f"Fehler beim Scraping: {str(e)}")

        # Wenn keine Daten gefunden wurden, gib ein leeres Ergebnis zurück
        return {
            'quelle': 'Keine Daten gefunden',
            'timestamp': datetime.now().isoformat(),
            'anzahl_angebote': 0,
            'marktpreis_min': None,
            'marktpreis_max': None,
            'marktpreis_mean': None,
            'marktpreis_median': None,
            'listings': []
        }

    def get_market_data_for_vehicles(self, vehicles, force_refresh=False, parallel=False):
        """
        Holt Marktdaten für mehrere Fahrzeuge.

        Args:
            vehicles (list): Liste von Fahrzeugdaten
            force_refresh (bool): Cache umgehen und frische Daten holen
            parallel (bool): Ob parallel verarbeitet werden soll (derzeit nicht implementiert)

        Returns:
            dict: Dictionary mit Fahrzeug-IDs als Schlüssel und Marktdaten als Werte
        """
        if not vehicles:
            logger.warning("Keine Fahrzeuge für Marktdatenabfrage bereitgestellt.")
            return {}

        # Sequentielle Verarbeitung
        results = {}
        for i, vehicle in enumerate(vehicles):
            # Eindeutige ID erstellen
            vehicle_id = self._get_vehicle_id(vehicle, i)

            try:
                logger.info(f"Hole Marktdaten für Fahrzeug {i+1}/{len(vehicles)}: {vehicle.get('marke', '')} {vehicle.get('modell', '')}")
                market_data = self.get_market_data(vehicle, force_refresh)
                if market_data:
                    results[vehicle_id] = market_data

                # Kurze Pause zwischen Anfragen
                if i < len(vehicles) - 1:
                    time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                logger.error(f"Fehler bei Marktdatenabfrage für Fahrzeug {i+1}: {str(e)}")

        logger.info(f"Marktdaten für {len(results)}/{len(vehicles)} Fahrzeuge erfolgreich abgerufen.")
        return results

    def _get_vehicle_id(self, vehicle, index=0):
        """
        Generiert eine eindeutige ID für ein Fahrzeug.

        Args:
            vehicle (dict): Fahrzeugdaten
            index (int): Index des Fahrzeugs in der Liste (Fallback)

        Returns:
            str: Eindeutige Fahrzeug-ID
        """
        # Prüfe auf eindeutige Identifikatoren
        if 'id' in vehicle:
            return str(vehicle['id'])

        if 'fahrgestellnummer' in vehicle and vehicle['fahrgestellnummer']:
            return f"vin_{vehicle['fahrgestellnummer']}"

        # Erstelle eine ID aus Marke, Modell, Baujahr und Kilometerstand
        marke = vehicle.get('marke', '').lower().replace(' ', '_')
        modell = vehicle.get('modell', '').lower().replace(' ', '_')
        baujahr = vehicle.get('baujahr', '')
        km = vehicle.get('kilometerstand', '')

        if marke and modell and baujahr:
            if km:
                return f"{marke}_{modell}_{baujahr}_{km}"
            return f"{marke}_{modell}_{baujahr}_{index}"

        # Fallback
        return f"vehicle_{index}"

    def _normalize_vehicle_data(self, vehicle_data):
        """
        Normalisiert Fahrzeugdaten für konsistente Suche.

        Args:
            vehicle_data (dict): Rohe Fahrzeugdaten

        Returns:
            dict: Normalisierte Fahrzeugdaten
        """
        normalized = {}

        # Marke normalisieren
        if 'marke' in vehicle_data:
            marke = str(vehicle_data['marke']).strip().lower()
            # Standardisiere Markennamen
            brand_mapping = {
                'mercedes': 'mercedes-benz',
                'vw': 'volkswagen',
                'opel': 'opel',
                'ford': 'ford',
                'bmw': 'bmw',
                'audi': 'audi',
                'seat': 'seat',
                'skoda': 'skoda',
                'citroen': 'citroen',
                'peugeot': 'peugeot',
                'toyota': 'toyota',
                'nissan': 'nissan',
                'hyundai': 'hyundai',
                'kia': 'kia'
            }
            normalized['marke'] = brand_mapping.get(marke, marke)

        # Modell normalisieren
        if 'modell' in vehicle_data:
            modell = str(vehicle_data['modell']).strip()
            # Entferne häufige Zusätze
            modell = re.sub(r'\s+(Limousine|Kombi|Cabrio|Cabriolet|Coupe|SUV|Automatik|Schaltgetriebe)\b', '', modell, flags=re.IGNORECASE)
            normalized['modell'] = modell

        # Baujahr normalisieren
        if 'baujahr' in vehicle_data:
            try:
                baujahr = int(vehicle_data['baujahr'])
                normalized['baujahr'] = baujahr

                # Berechne Baujahr-Toleranz (±1 Jahr)
                normalized['baujahr_min'] = baujahr - 1
                normalized['baujahr_max'] = baujahr + 1
            except (ValueError, TypeError):
                pass

        # Kilometerstand normalisieren
        if 'kilometerstand' in vehicle_data:
            try:
                km = int(vehicle_data['kilometerstand'])
                normalized['kilometerstand'] = km

                # Berechne Kilometerstand-Toleranz (+30%)
                normalized['kilometerstand_max'] = int(km * 1.3)
            except (ValueError, TypeError):
                pass

        # Kraftstofftyp normalisieren
        if 'kraftstoff' in vehicle_data:
            kraftstoff = str(vehicle_data['kraftstoff']).strip().lower()
            fuel_mapping = {
                'diesel': 'diesel',
                'benzin': 'benzin',
                'petrol': 'benzin',
                'gasoline': 'benzin',
                'hybrid': 'hybrid',
                'elektro': 'elektro',
                'electric': 'elektro',
                'autogas': 'autogas',
                'lpg': 'autogas',
                'erdgas': 'erdgas',
                'cng': 'erdgas'
            }
            normalized['kraftstoff'] = fuel_mapping.get(kraftstoff, kraftstoff)

        # Leistung normalisieren
        if 'leistung' in vehicle_data:
            try:
                ps = int(vehicle_data['leistung'])
                normalized['leistung'] = ps

                # Berechne Leistung in kW, falls nicht vorhanden
                if 'leistung_kw' not in vehicle_data:
                    normalized['leistung_kw'] = int(ps * 0.735)
            except (ValueError, TypeError):
                pass
        elif 'leistung_kw' in vehicle_data:
            try:
                kw = int(vehicle_data['leistung_kw'])
                normalized['leistung_kw'] = kw

                # Berechne Leistung in PS, falls nicht vorhanden
                normalized['leistung'] = int(kw / 0.735)
            except (ValueError, TypeError):
                pass

        return normalized

    def _simplify_search_parameters(self, vehicle_data):
        """
        Vereinfacht die Suchparameter für eine breitere Suche.

        Args:
            vehicle_data (dict): Fahrzeugdaten

        Returns:
            dict: Vereinfachte Fahrzeugdaten
        """
        simplified = vehicle_data.copy()

        # Entferne spezifische Modellvarianten, behalte nur die Basismodellreihe
        if 'modell' in simplified:
            modell = simplified['modell']

            # BMW-Beispiel: "320d xDrive" -> "3er" oder "320"
            if simplified.get('marke') == 'bmw':
                for series, models in self.series_mapping['bmw'].items():
                    if any(model_prefix in modell for model_prefix in models):
                        simplified['modell'] = series
                        break

            # Mercedes-Beispiel: "C220d 4MATIC" -> "C-Klasse"
            elif simplified.get('marke') == 'mercedes-benz':
                for series, models in self.series_mapping['mercedes-benz'].items():
                    if any(model_prefix in modell for model_prefix in models):
                        simplified['modell'] = series
                        break

            # Generischer Ansatz: Entferne alle Zeichen nach einer Zahl
            else:
                simplified['modell'] = re.sub(r'([A-Za-z]+)[\s-]?(\d+).*', r'\1 \2', modell)

        # Erweitere Baujahr-Toleranz
        if 'baujahr' in simplified:
            baujahr = simplified['baujahr']
            simplified['baujahr_min'] = baujahr - 2
            simplified['baujahr_max'] = baujahr + 2

        # Erhöhe Kilometerstand-Toleranz
        if 'kilometerstand' in simplified:
            km = simplified['kilometerstand']
            simplified['kilometerstand_max'] = int(km * 1.5)  # 50% mehr

        # Entferne zu spezifische Parameter
        for key in ['leistung', 'leistung_kw', 'kraftstoff']:
            if key in simplified:
                del simplified[key]

        return simplified

    def _get_cache_path(self, vehicle):
        """
        Gibt den Pfad zur Cache-Datei für ein Fahrzeug zurück.

        Args:
            vehicle (dict): Fahrzeugdaten

        Returns:
            Path: Pfad zur Cache-Datei
        """
        # Erstelle einen eindeutigen Schlüssel für das Fahrzeug
        key_parts = [
            str(vehicle.get('marke', '')).lower().replace(' ', '_'),
            str(vehicle.get('modell', '')).lower().replace(' ', '_'),
            str(vehicle.get('baujahr', '')),
            str(vehicle.get('kilometerstand', ''))
        ]

        vehicle_key = '_'.join(filter(None, key_parts))

        # Stelle sicher, dass der Schlüssel gültig ist
        vehicle_key = re.sub(r'[\\/*?:"<>|]', '_', vehicle_key)

        # Begrenze die Länge
        if len(vehicle_key) > 100:
            import hashlib
            hash_part = hashlib.md5(vehicle_key.encode()).hexdigest()[:8]
            vehicle_key = vehicle_key[:90] + '_' + hash_part

        return Path(self.config.CACHE_DIR) / f"{vehicle_key}.json"

    def _load_from_cache(self, vehicle):
        """
        Lädt Marktdaten aus dem Cache.

        Args:
            vehicle (dict): Fahrzeugdaten

        Returns:
            dict: Gecachte Marktdaten oder None, wenn kein gültiger Cache gefunden wurde
        """
        cache_path = self._get_cache_path(vehicle)

        if not cache_path.exists():
            return None

        try:
            # Prüfe, ob der Cache gültig ist
            cache_stat = cache_path.stat()
            cache_time = datetime.fromtimestamp(cache_stat.st_mtime)
            cache_age = datetime.now() - cache_time

            if cache_age > timedelta(days=self.config.CACHE_EXPIRY_DAYS):
                logger.info(f"Cache für {vehicle.get('marke')} {vehicle.get('modell')} ist abgelaufen")
                return None

            # Lade den Cache
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            logger.info(f"Marktdaten für {vehicle.get('marke')} {vehicle.get('modell')} aus Cache geladen")
            return cache_data

        except Exception as e:
            logger.error(f"Fehler beim Laden aus Cache: {str(e)}")
            return None

    def _save_to_cache(self, vehicle, market_data):
        """
        Speichert Marktdaten im Cache.

        Args:
            vehicle (dict): Fahrzeugdaten
            market_data (dict): Marktdaten

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        cache_path = self._get_cache_path(vehicle)

        try:
            # Füge Zeitstempel hinzu
            market_data['cached_at'] = datetime.now().isoformat()

            # Füge Fahrzeugdaten für Referenz hinzu
            if 'vehicle_data' not in market_data:
                market_data['vehicle_data'] = {
                    'marke': vehicle.get('marke'),
                    'modell': vehicle.get('modell'),
                    'baujahr': vehicle.get('baujahr'),
                    'kilometerstand': vehicle.get('kilometerstand')
                }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(market_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Marktdaten für {vehicle.get('marke')} {vehicle.get('modell')} im Cache gespeichert")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern im Cache: {str(e)}")
            return False

    def build_search_url(self, vehicle_data):
        """
        Erstellt eine Suchanfrage-URL für mobile.de

        Args:
            vehicle_data (dict): Fahrzeugdaten

        Returns:
            str: Die generierte Suchanfrage-URL
        """
        # Basisparameter
        params = []

        # Marke hinzufügen
        marke = vehicle_data.get('marke', '')
        if marke:
            make_id = self._get_mobile_make_id(marke)
            if make_id:
                params.append(f"makeModelVariant1.makeId={make_id}")

        # Modell hinzufügen
        modell = vehicle_data.get('modell', '')
        if modell:
            # Kodiere das Modell für die URL
            encoded_model = quote_plus(modell)
            params.append(f"makeModelVariant1.modelDescription={encoded_model}")

        # Baujahr (Mindest)
        if 'baujahr_min' in vehicle_data:
            params.append(f"minFirstRegistrationDate={vehicle_data['baujahr_min']}")
        elif 'baujahr' in vehicle_data:
            # Fallback: -1 Jahr
            params.append(f"minFirstRegistrationDate={vehicle_data['baujahr']-1}")

        # Kilometerstand (Maximum)
        if 'kilometerstand_max' in vehicle_data:
            params.append(f"maxMileage={vehicle_data['kilometerstand_max']}")
        elif 'kilometerstand' in vehicle_data:
            # Fallback: +30%
            max_km = int(vehicle_data['kilometerstand'] * 1.3)
            params.append(f"maxMileage={max_km}")

        # Kraftstofftyp
        if 'kraftstoff' in vehicle_data:
            fuel_mapping = {
                'benzin': 'PETROL',
                'diesel': 'DIESEL',
                'elektro': 'ELECTRICITY',
                'hybrid': 'HYBRID',
                'autogas': 'LPG',
                'erdgas': 'CNG'
            }
            if vehicle_data['kraftstoff'] in fuel_mapping:
                params.append(f"fuel={fuel_mapping[vehicle_data['kraftstoff']]}")

        # Sortierung (günstigste zuerst)
        params.append("sortOption.sortBy=price")
        params.append("sortOption.sortOrder=ASCENDING")

        # Ergebnisse pro Seite (maximieren für Effizienz)
        params.append("pageSize=20")

        # URL zusammenbauen
        url = self.config.BASE_URL
        if params:
            url += "?" + "&".join(params)

        return url

    def _get_mobile_make_id(self, marke):
        """
        Gibt die mobile.de-Make-ID für eine Marke zurück.

        Args:
            marke (str): Markenname

        Returns:
            int: mobile.de-Make-ID oder None, wenn nicht gefunden
        """
        # Mapping von Markennamen zu mobile.de-IDs
        make_ids = {
            'audi': 1900,
            'bmw': 3500,
            'ford': 9000,
            'mercedes-benz': 17200,
            'mercedes': 17200,
            'opel': 19000,
            'volkswagen': 25200,
            'vw': 25200,
            'porsche': 20000,
            'toyota': 24100,
            'volvo': 25100,
            'skoda': 21700,
            'seat': 22500,
            'renault': 21600,
            'peugeot': 19800,
            'nissan': 18700,
            'mini': 17600,
            'mazda': 16800,
            'kia': 13200,
            'hyundai': 11600,
            'honda': 11000,
            'fiat': 8800,
            'citroen': 5900,
            'chevrolet': 5600
        }

        return make_ids.get(marke.lower())

    def _setup_driver(self):
        """Richtet den Selenium WebDriver ein"""
        if self.driver:
            return  # Driver bereits eingerichtet

        try:
            chrome_options = Options()
            if self.config.HEADLESS_BROWSER:
                chrome_options.add_argument("--headless")

            chrome_options.add_argument(f"user-agent={self.config.USER_AGENT}")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--lang=de-DE")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("WebDriver initialisiert")

        except Exception as e:
            logger.error(f"Fehler beim Einrichten des WebDrivers: {str(e)}")
            self.driver = None

    def _quit_driver(self):
        """Beendet den WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Fehler beim Beenden des WebDrivers: {str(e)}")
            finally:
                self.driver = None

    def _scrape_listings(self, vehicle_data):
        """
        Scrapt Fahrzeugdaten von mobile.de.

        Args:
            vehicle_data (dict): Fahrzeugdaten

        Returns:
            dict: Marktdaten mit Listings
        """
        search_url = self.build_search_url(vehicle_data)
        logger.info(f"Scrape URL: {search_url}")

        try:
            self._setup_driver()
            self.driver.get(search_url)

            # Warten auf Seitenladung und Cookie-Banner akzeptieren
            self._handle_cookie_banner()

            # Sammle Listings
            all_listings = []
            pages_scraped = 0

            while pages_scraped < self.config.MAX_PAGES and len(all_listings) < self.config.MAX_VEHICLES:
                # Warte auf Seitenladung
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "cBox-body--resultlist"))
                    )
                except TimeoutException:
                    logger.warning("Timeout beim Warten auf Suchergebnisse")
                    break

                # Extrahiere Fahrzeuge von der aktuellen Seite
                page_listings = self._extract_listings_from_page()

                if page_listings:
                    all_listings.extend(page_listings)
                    logger.info(f"Seite {pages_scraped + 1}: {len(page_listings)} Fahrzeuge gefunden")
                else:
                    logger.warning(f"Keine Fahrzeuge auf Seite {pages_scraped + 1} gefunden")
                    break

                # Zur nächsten Seite
                if len(all_listings) < self.config.MAX_VEHICLES and not self._go_to_next_page():
                    logger.info("Keine weitere Seite verfügbar")
                    break

                pages_scraped += 1
                self._random_wait()

            # Erstelle Marktdaten
            if all_listings:
                # Berechne Statistiken
                prices = [listing['preis'] for listing in all_listings if listing['preis'] > 0]

                statistics = {
                    'quelle': 'mobile.de-scraper',
                    'timestamp': datetime.now().isoformat(),
                    'anzahl_angebote': len(all_listings),
                    'marktpreis_min': min(prices) if prices else None,
                    'marktpreis_max': max(prices) if prices else None,
                    'marktpreis_mean': sum(prices) / len(prices) if prices else None,
                    'marktpreis_median': self._median(prices) if prices else None,
                    'listings': all_listings
                }

                return statistics
            else:
                logger.warning("Keine Fahrzeuge gefunden")
                return {
                    'quelle': 'mobile.de-scraper',
                    'timestamp': datetime.now().isoformat(),
                    'anzahl_angebote': 0,
                    'listings': []
                }

        except Exception as e:
            logger.error(f"Fehler beim Scrapen: {str(e)}")
            return None

        finally:
            self._quit_driver()

    def _handle_cookie_banner(self):
        """Behandelt das Cookie-Banner, falls es erscheint"""
        try:
            cookie_accept = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "mde-consent-accept-btn"))
            )
            cookie_accept.click()
            logger.debug("Cookie-Banner akzeptiert")
            time.sleep(1)  # Kurze Pause nach dem Klick
        except (TimeoutException, NoSuchElementException):
            logger.debug("Kein Cookie-Banner gefunden oder nicht klickbar")

    def _extract_listings_from_page(self):
        """
        Extrahiert Fahrzeugdaten von der aktuellen Seite.

        Returns:
            list: Liste von Fahrzeug-Dictionaries
        """
        if not self.driver:
            return []

        listings = []

        try:
            # Finde alle Fahrzeug-Listings
            listing_elements = self.driver.find_elements(By.CSS_SELECTOR, ".cBox-body--resultlist .cBox-body--vehicleDetails")

            for element in listing_elements:
                try:
                    # Titel
                    title_element = element.find_element(By.CSS_SELECTOR, "h2.title")
                    title = title_element.text.strip()

                    # Preis
                    try:
                        price_element = element.find_element(By.CSS_SELECTOR, ".vehicle-data .pricePrimaryCountup")
                        price_text = price_element.text.strip()
                        price = self._clean_price(price_text)
                    except NoSuchElementException:
                        price = None

                    # Details (Baujahr, Kilometerstand, etc.)
                    try:
                        details_element = element.find_element(By.CSS_SELECTOR, ".vehicle-data .container .rbt-regMilPow")
                        details_text = details_element.text.strip()
                    except NoSuchElementException:
                        details_text = ""

                    # URL
                    try:
                        url_element = title_element.find_element(By.XPATH, "./..")
                        url = url_element.get_attribute("href")
                    except NoSuchElementException:
                        url = ""

                    # Parse Details
                    year = None
                    mileage = None
                    fuel_type = None
                    power = None

                    if details_text:
                        # Format: "EZ 01/2020, 50.000 km, Benzin, 100 kW"
                        details = details_text.split(',')

                        # Baujahr
                        if len(details) > 0 and "EZ" in details[0]:
                            year_part = details[0].replace("EZ", "").strip()
                            if "/" in year_part:
                                try:
                                    year = int(year_part.split('/')[1])
                                    # 2-stelliges Jahr in 4-stelliges konvertieren
                                    if year < 100:
                                        if year < 50:  # Annahme: 00-49 -> 2000-2049
                                            year += 2000
                                        else:  # Annahme: 50-99 -> 1950-1999
                                            year += 1900
                                except (ValueError, IndexError):
                                    pass

                        # Kilometerstand
                        if len(details) > 1 and "km" in details[1]:
                            mileage_part = details[1].strip()
                            mileage = self._clean_price(mileage_part)

                        # Kraftstoff und Leistung
                        if len(details) > 2:
                            fuel_power = details[2].strip().split(',')
                            if fuel_power and len(fuel_power) > 0:
                                fuel_type = fuel_power[0].strip()

                            if len(fuel_power) > 1 and "kW" in fuel_power[1]:
                                power_part = fuel_power[1].strip()
                                try:
                                    power = int(''.join(filter(str.isdigit, power_part)))
                                except ValueError:
                                    pass

                    # Erstelle Listing-Objekt
                    listing = {
                        'title': title,
                        'preis': price,
                        'baujahr': year,
                        'kilometerstand': mileage,
                        'kraftstoff': fuel_type,
                        'leistung_kw': power,
                        'url': url,
                        'haendler': True,  # Annahme: Alle Listings sind von Händlern
                        'standort': None,
                        'datum_erstellt': None
                    }

                    # Füge PS hinzu, wenn kW bekannt ist
                    if power:
                        listing['leistung'] = int(power / 0.735)

                    listings.append(listing)

                except Exception as e:
                    logger.error(f"Fehler beim Parsen eines Listings: {str(e)}")

            return listings

        except Exception as e:
            logger.error(f"Fehler beim Extrahieren der Listings: {str(e)}")
            return []

    def _go_to_next_page(self):
        """
        Navigiert zur nächsten Seite der Suchergebnisse.

        Returns:
            bool: True, wenn zur nächsten Seite navigiert wurde
        """
        if not self.driver:
            return False

        try:
            # Finde den "Weiter"-Button
            try:
                next_button = self.driver.find_element(By.CSS_SELECTOR, ".pagination-nav .btn--orange.btn--s")

                if next_button and next_button.is_enabled() and next_button.is_displayed():
                    # Scroll zum Button (für sicheres Klicken)
                    self.driver.execute_script("arguments[0].scrollIntoView();", next_button)
                    time.sleep(0.5)

                    # Klicke auf den Button
                    next_button.click()

                    # Warte auf Seitenladung
                    WebDriverWait(self.driver, 10).until(
                        EC.staleness_of(next_button)
                    )

                    return True
                else:
                    logger.info("Keine weitere Seite verfügbar")
                    return False

            except NoSuchElementException:
                logger.info("Kein 'Weiter'-Button gefunden")
                return False

            except TimeoutException:
                logger.warning("Timeout beim Warten auf die nächste Seite")
                return False

        except Exception as e:
            logger.error(f"Fehler beim Navigieren zur nächsten Seite: {str(e)}")
            return False

    def _random_wait(self):
        """Wartet eine zufällige Zeit zwischen MIN_WAIT_TIME und MAX_WAIT_TIME"""
        wait_time = random.uniform(self.config.MIN_WAIT_SECONDS, self.config.MAX_WAIT_SECONDS)
        time.sleep(wait_time)

    def _clean_price(self, price_str):
        """
        Bereinigt einen Preisstring und konvertiert ihn in eine Zahl.

        Args:
            price_str (str): Preisstring (z.B. "15.990 €")

        Returns:
            int: Bereinigter Preis
        """
        if not price_str:
            return None

        # Entferne alle Nicht-Zahlen und behalte Punkte und Kommas
        cleaned = ''.join(c for c in price_str if c.isdigit() or c in '.,')

        if not cleaned:
            return None

        # Behandle deutsches Zahlenformat (1.234,56 -> 1234.56)
        if ',' in cleaned and '.' in cleaned:
            # Format: 1.234,56
            cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            # Format: 1234,56
            cleaned = cleaned.replace(',', '.')

        try:
            # Konvertiere zu Float und dann zu Int (wenn keine Nachkommastellen)
            price = float(cleaned)
            return int(price) if price == int(price) else price
        except ValueError:
            return None

    def _median(self, numbers):
        """
        Berechnet den Median einer Liste von Zahlen.

        Args:
            numbers (list): Liste von Zahlen

        Returns:
            float: Median oder None, wenn die Liste leer ist
        """
        if not numbers:
            return None

        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)

        if n % 2 == 0:
            return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
        else:
            return sorted_numbers[n//2]

    def market_data_to_dataframe(self, market_data):
        """
        Konvertiert Marktdaten in einen DataFrame.

        Args:
            market_data (dict): Marktdaten

        Returns:
            pd.DataFrame: DataFrame mit Marktdaten
        """
        if not market_data or 'listings' not in market_data:
            return pd.DataFrame()

        # Erstelle DataFrame aus Listings
        df = pd.DataFrame(market_data['listings'])

        # Füge Metadaten hinzu
        if df.empty:
            return df

        # Füge statistische Daten als neue Spalten hinzu
        for key in ['quelle', 'timestamp', 'anzahl_angebote', 'marktpreis_min',
                    'marktpreis_max', 'marktpreis_mean', 'marktpreis_median']:
            if key in market_data:
                df[f'stat_{key}'] = market_data[key]

        return df

    def clear_cache(self, vehicle=None):
        """
        Löscht den Cache für ein Fahrzeug oder alle Fahrzeuge.

        Args:
            vehicle (dict, optional): Fahrzeugdaten oder None für alle

        Returns:
            int: Anzahl der gelöschten Cache-Dateien
        """
        if vehicle:
            # Lösche Cache für ein bestimmtes Fahrzeug
            cache_path = self._get_cache_path(vehicle)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cache für {vehicle.get('marke')} {vehicle.get('modell')} gelöscht")
                return 1
            return 0
        else:
            # Lösche alle Cache-Dateien
            count = 0
            for cache_file in Path(self.config.CACHE_DIR).glob("*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Fehler beim Löschen von {cache_file}: {str(e)}")

            logger.info(f"{count} Cache-Dateien gelöscht")
            return count