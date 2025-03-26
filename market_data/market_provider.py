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
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class MarketDataConfig:
    """Konfigurationsklasse für Marktdaten-Provider"""
    # Allgemeine Einstellungen
    CACHE_DIR = "cache/market_data"
    CACHE_EXPIRY_DAYS = 3

    # API-Einstellungen
    API_KEY = os.environ.get("MOBILE_API_KEY", "")
    API_SECRET = os.environ.get("MOBILE_API_SECRET", "")
    USE_MOBILE_API = bool(API_KEY and API_SECRET)

    # Scraper-Einstellungen
    MAX_PAGES = 3
    MAX_VEHICLES = 30
    MIN_WAIT_SECONDS = 2
    MAX_WAIT_SECONDS = 5
    HEADLESS_BROWSER = True
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


class MarketDataProvider(ABC):
    """Abstrakte Basisklasse für Marktdaten-Provider"""

    def __init__(self, config=None):
        """
        Initialisiert den Marktdaten-Provider.

        Args:
            config (MarketDataConfig, optional): Konfiguration für den Provider
        """
        self.config = config or MarketDataConfig()

        # Stelle sicher, dass das Cache-Verzeichnis existiert
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

    @abstractmethod
    def get_market_data(self, vehicle, force_refresh=False):
        """
        Holt Marktdaten für ein Fahrzeug.

        Args:
            vehicle (dict): Fahrzeugdaten mit Marke, Modell, Baujahr, etc.
            force_refresh (bool): Cache umgehen und frische Daten holen

        Returns:
            dict: Marktdaten mit Statistiken und Listings
        """
        pass

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

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(market_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Marktdaten für {vehicle.get('marke')} {vehicle.get('modell')} im Cache gespeichert")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern im Cache: {str(e)}")
            return False

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
                cache_file.unlink()
                count += 1

            logger.info(f"{count} Cache-Dateien gelöscht")
            return count


class MobileDeAPI(MarketDataProvider):
    """Provider für Marktdaten von mobile.de über die offizielle API"""

    def __init__(self, config=None):
        """
        Initialisiert den mobile.de API-Provider.

        Args:
            config (MarketDataConfig, optional): Konfiguration für den Provider
        """
        super().__init__(config)
        self.api_key = self.config.API_KEY
        self.api_secret = self.config.API_SECRET
        self.base_url = "https://services.mobile.de/search-api/v1/vehicles"

        # Validiere API-Zugangsdaten
        if not self.api_key or not self.api_secret:
            logger.warning("Mobile.de API-Schlüssel oder Secret nicht konfiguriert")

    def get_market_data(self, vehicle, force_refresh=False):
        """
        Holt Marktdaten für ein Fahrzeug über die mobile.de API.

        Args:
            vehicle (dict): Fahrzeugdaten mit Marke, Modell, Baujahr, etc.
            force_refresh (bool): Cache umgehen und frische Daten holen

        Returns:
            dict: Marktdaten mit Statistiken und Listings
        """
        if not self.api_key or not self.api_secret:
            logger.error("Mobile.de API-Zugangsdaten fehlen")
            return None

        # Prüfe Cache, wenn nicht erzwungen wird
        if not force_refresh:
            cached_data = self._load_from_cache(vehicle)
            if cached_data:
                return cached_data

        try:
            # Erstelle API-Parameter basierend auf Fahrzeugdaten
            params = self._build_api_params(vehicle)

            # Authentifizierungsheader
            headers = {
                "Authorization": f"Basic {self._get_auth_token()}",
                "Accept": "application/json"
            }

            # API-Anfrage senden
            response = requests.get(self.base_url, params=params, headers=headers)

            if response.status_code == 200:
                # Verarbeite die Antwort
                api_data = response.json()

                # Transformiere in einheitliches Format
                market_data = self._transform_api_data(api_data, vehicle)

                # Speichere im Cache
                self._save_to_cache(vehicle, market_data)

                return market_data
            else:
                logger.error(f"API-Fehler: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Fehler bei der API-Anfrage: {str(e)}")
            return None

    def _get_auth_token(self):
        """
        Erstellt den Base64-codierten Authentifizierungstoken.

        Returns:
            str: Base64-codierter Token
        """
        import base64
        auth_string = f"{self.api_key}:{self.api_secret}"
        return base64.b64encode(auth_string.encode()).decode()

    def _build_api_params(self, vehicle):
        """
        Erstellt API-Parameter basierend auf Fahrzeugdaten.

        Args:
            vehicle (dict): Fahrzeugdaten

        Returns:
            dict: API-Parameter
        """
        params = {}

        # Basis-Parameter
        if 'marke' in vehicle:
            params['makeKey'] = self._normalize_make_for_api(vehicle['marke'])

        if 'modell' in vehicle:
            params['modelKey'] = self._normalize_model_for_api(vehicle['modell'])

        # Baujahr (Bereich um das angegebene Jahr)
        if 'baujahr' in vehicle:
            baujahr = int(vehicle['baujahr'])
            params['minFirstRegistrationDate'] = f"{baujahr-1}-01"
            params['maxFirstRegistrationDate'] = f"{baujahr+1}-12"

        # Kilometerstand (bis zum angegebenen Wert + Toleranz)
        if 'kilometerstand' in vehicle:
            max_km = int(vehicle['kilometerstand'] * 1.3)  # 30% Toleranz
            params['maxMileage'] = max_km

        # Weitere Parameter
        params['pageSize'] = self.config.MAX_VEHICLES
        params['pageNumber'] = 1
        params['sort'] = 'price_asc'  # Sortierung nach Preis aufsteigend

        return params

    def _normalize_make_for_api(self, make):
        """
        Normalisiert den Markennamen für die API.

        Args:
            make (str): Markenname

        Returns:
            str: Normalisierter Markenname für die API
        """
        # Mapping von normalisierten Markennamen zu API-Schlüsseln
        make_mapping = {
            'mercedes-benz': 'MERCEDES_BENZ',
            'bmw': 'BMW',
            'volkswagen': 'VOLKSWAGEN',
            'audi': 'AUDI',
            'opel': 'OPEL',
            'ford': 'FORD',
            'porsche': 'PORSCHE',
            # Weitere Mappings hier hinzufügen
        }

        normalized = make.lower().strip()
        return make_mapping.get(normalized, make.upper())

    def _normalize_model_for_api(self, model):
        """
        Normalisiert den Modellnamen für die API.

        Args:
            model (str): Modellname

        Returns:
            str: Normalisierter Modellname für die API
        """
        # Einfache Normalisierung - kann erweitert werden
        return model.upper().replace(' ', '_')

    def _transform_api_data(self, api_data, vehicle):
        """
        Transformiert API-Daten in ein einheitliches Format.

        Args:
            api_data (dict): API-Antwort
            vehicle (dict): Fahrzeugdaten

        Returns:
            dict: Transformierte Marktdaten
        """
        # Extrahiere Listings
        listings = []
        for item in api_data.get('items', []):
            try:
                listing = {
                    'titel': item.get('title', ''),
                    'preis': item.get('price', {}).get('amount', 0),
                    'baujahr': self._extract_year(item.get('firstRegistrationDate', '')),
                    'kilometerstand': item.get('mileage', {}).get('value', 0),
                    'kraftstoff': item.get('fuel', ''),
                    'leistung_kw': item.get('power', {}).get('value', 0),
                    'url': item.get('url', ''),
                    'haendler': item.get('seller', {}).get('type', '') == 'DEALER',
                    'standort': item.get('location', {}).get('city', ''),
                    'datum_erstellt': item.get('creationDate', '')
                }
                listings.append(listing)
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung eines Listings: {str(e)}")

        # Berechne Statistiken
        prices = [listing['preis'] for listing in listings if listing['preis'] > 0]

        statistics = {
            'quelle': 'mobile.de-api',
            'timestamp': datetime.now().isoformat(),
            'anzahl_angebote': len(listings),
            'marktpreis_min': min(prices) if prices else 0,
            'marktpreis_max': max(prices) if prices else 0,
            'marktpreis_mean': sum(prices) / len(prices) if prices else 0,
            'marktpreis_median': self._median(prices) if prices else 0,
            'raw_data': api_data,  # Rohdaten für spätere Nutzung
            'listings': listings
        }

        return statistics

    def _extract_year(self, date_str):
        """
        Extrahiert das Jahr aus einem Datumsstring.

        Args:
            date_str (str): Datumsstring im Format YYYY-MM-DD

        Returns:
            int: Jahr oder None
        """
        if not date_str:
            return None

        try:
            return int(date_str.split('-')[0])
        except (IndexError, ValueError):
            return None

    def _median(self, numbers):
        """
        Berechnet den Median einer Zahlenreihe.

        Args:
            numbers (list): Liste von Zahlen

        Returns:
            float: Median
        """
        if not numbers:
            return 0

        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)

        if n % 2 == 0:
            return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
        else:
            return sorted_numbers[n//2]


class MobileDeScraperProvider(MarketDataProvider):
    """Provider für Marktdaten von mobile.de über Web-Scraping"""

    def __init__(self, config=None):
        """
        Initialisiert den mobile.de Scraper-Provider.

        Args:
            config (MarketDataConfig, optional): Konfiguration für den Provider
        """
        super().__init__(config)
        self.base_url = "https://www.mobile.de/de/fahrzeug/search.html"
        self.driver = None

    def get_market_data(self, vehicle, force_refresh=False):
        """
        Holt Marktdaten für ein Fahrzeug durch Web-Scraping.

        Args:
            vehicle (dict): Fahrzeugdaten mit Marke, Modell, Baujahr, etc.
            force_refresh (bool): Cache umgehen und frische Daten holen

        Returns:
            dict: Marktdaten mit Statistiken und Listings
        """
        # Prüfe Cache, wenn nicht erzwungen wird
        if not force_refresh:
            cached_data = self._load_from_cache(vehicle)
            if cached_data:
                return cached_data

        try:
            # Initialisiere WebDriver bei Bedarf
            if not self.driver:
                self._setup_driver()

            # Sammle Listings durch Scraping
            listings = self._scrape_listings(vehicle)

            if not listings:
                logger.warning(f"Keine Listings gefunden für {vehicle.get('marke')} {vehicle.get('modell')}")
                return None

            # Berechne Statistiken
            prices = [listing['preis'] for listing in listings if listing['preis'] > 0]

            market_data = {
                'quelle': 'mobile.de-scraper',
                'timestamp': datetime.now().isoformat(),
                'anzahl_angebote': len(listings),
                'marktpreis_min': min(prices) if prices else 0,
                'marktpreis_max': max(prices) if prices else 0,
                'marktpreis_mean': sum(prices) / len(prices) if prices else 0,
                'marktpreis_median': self._median(prices) if prices else 0,
                'raw_data': None,  # Keine Rohdaten beim Scraping
                'listings': listings
            }

            # Speichere im Cache
            self._save_to_cache(vehicle, market_data)

            return market_data

        except Exception as e:
            logger.error(f"Fehler beim Scraping: {str(e)}")
            return None

        finally:
            # WebDriver schließen, wenn verwendet
            self._quit_driver()

    def _setup_driver(self):
        """Richtet den Selenium WebDriver ein"""
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        chrome_options = Options()
        if self.config.HEADLESS_BROWSER:
            chrome_options.add_argument("--headless")

        chrome_options.add_argument(f"user-agent={self.config.USER_AGENT}")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.set_page_load_timeout(30)

        logger.info("WebDriver initialisiert")

    def _quit_driver(self):
        """Schließt den WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Fehler beim Schließen des WebDrivers: {str(e)}")
            finally:
                self.driver = None

    def _scrape_listings(self, vehicle):
        """
        Scrapt Fahrzeugangebote von mobile.de.

        Args:
            vehicle (dict): Fahrzeugdaten

        Returns:
            list: Liste von Listings
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException

        # Erstelle Suchurl
        search_url = self._build_search_url(vehicle)
        logger.info(f"Scrape URL: {search_url}")

        # Öffne die Seite
        self.driver.get(search_url)

        # Warte auf Seitenladung und akzeptiere Cookie-Banner
        self._handle_cookie_banner()

        # Sammle Listings von allen relevanten Seiten
        all_listings = []
        pages_scraped = 0

        while pages_scraped < self.config.MAX_PAGES and len(all_listings) < self.config.MAX_VEHICLES:
            # Extrahiere Listings von der aktuellen Seite
            try:
                # Warte, bis die Fahrzeugliste geladen ist
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "cBox-body--resultlist"))
                )

                # Finde alle Fahrzeug-Listings
                listing_elements = self.driver.find_elements(By.CSS_SELECTOR, ".cBox-body--resultlist .cBox-body--vehicleDetails")

                # Verarbeite jedes Listing
                for element in listing_elements:
                    listing = self._parse_listing(element)
                    if listing:
                        all_listings.append(listing)

                    # Prüfe, ob wir die maximale Anzahl erreicht haben
                    if len(all_listings) >= self.config.MAX_VEHICLES:
                        break

                logger.info(f"Seite {pages_scraped + 1}: {len(listing_elements)} Listings gefunden")

                # Zur nächsten Seite navigieren, falls möglich
                if not self._go_to_next_page():
                    break

                pages_scraped += 1

                # Zufällige Wartezeit
                self._random_wait()

            except (TimeoutException, NoSuchElementException) as e:
                logger.error(f"Fehler beim Scrapen der Seite: {str(e)}")
                break

        logger.info(f"Insgesamt {len(all_listings)} Listings gescrapt")
        return all_listings

    def _build_search_url(self, vehicle):
        """
        Erstellt die Suchanfrage-URL für mobile.de.

        Args:
            vehicle (dict): Fahrzeugdaten

        Returns:
            str: URL für die Suchanfrage
        """
        # Basisparameter
        params = []

        # Marke
        if 'marke' in vehicle:
            make_id = self._get_make_id(vehicle['marke'])
            if make_id:
                params.append(f"makeModelVariant1.makeId={make_id}")

        # Modell
        if 'modell' in vehicle:
            params.append(f"makeModelVariant1.modelDescription={vehicle['modell']}")

        # Baujahr (Minimum)
        if 'baujahr' in vehicle:
            # Ein Jahr weniger als Toleranz
            min_year = int(vehicle['baujahr']) - 1
            params.append(f"minFirstRegistrationDate={min_year}")

        # Kilometerstand (Maximum mit Toleranz)
        if 'kilometerstand' in vehicle:
            # 30% mehr als Toleranz
            max_km = int(vehicle['kilometerstand'] * 1.3)
            params.append(f"maxMileage={max_km}")

        # Sortierung (günstigste zuerst)
        params.append("sortOption.sortBy=price")
        params.append("sortOption.sortOrder=ASCENDING")

        # URL zusammenbauen
        url = self.base_url
        if params:
            url += "?" + "&".join(params)

        return url

    def _get_make_id(self, make):
        """
        Gibt die mobile.de-ID für eine Fahrzeugmarke zurück.

        Args:
            make (str): Markenname

        Returns:
            int: Marken-ID oder None
        """
        # Mapping von Markennamen zu mobile.de IDs
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
            # Weitere IDs hier hinzufügen
        }

        return make_ids.get(make.lower().strip())

    def _handle_cookie_banner(self):
        """Akzeptiert das Cookie-Banner, falls vorhanden"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException

        try:
            # Warte auf das Banner und klicke auf "Akzeptieren"
            WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "mde-consent-accept-btn"))
            ).click()
            logger.debug("Cookie-Banner akzeptiert")
        except TimeoutException:
            logger.debug("Kein Cookie-Banner gefunden oder nicht klickbar")

    def _parse_listing(self, element):
        """
        Extrahiert Daten aus einem Listing-Element.

        Args:
            element: Selenium WebElement des Listings

        Returns:
            dict: Extrahierte Daten oder None bei Fehler
        """
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException

        try:
            # Titel
            try:
                title_element = element.find_element(By.CSS_SELECTOR, "h2.title")
                title = title_element.text.strip()
            except NoSuchElementException:
                title = ""

            # Preis
            try:
                price_element = element.find_element(By.CSS_SELECTOR, ".vehicle-data .pricePrimaryCountup")
                price_text = price_element.text.strip()
                price = self._clean_price(price_text)
            except NoSuchElementException:
                price = 0

            # Details (Baujahr, Kilometerstand, Kraftstoff, Leistung)
            try:
                details_element = element.find_element(By.CSS_SELECTOR, ".vehicle-data .container .rbt-regMilPow")
                details_text = details_element.text.strip()
            except NoSuchElementException:
                details_text = ""

            # URL
            try:
                url_element = title_element.find_element(By.XPATH, "./..") if title_element else None
                url = url_element.get_attribute("href") if url_element else ""
            except (NoSuchElementException, AttributeError):
                url = ""

            # Parse Details
            year = None
            mileage = None
            fuel = None
            power = None

            if details_text:
                details = details_text.split(',')

                # Jahr (EZ MM/YYYY)
                if len(details) > 0 and "EZ" in details[0]:
                    year_part = details[0].replace("EZ", "").strip()
                    if "/" in year_part:
                        year = int(year_part.split('/')[1])

                # Kilometerstand
                if len(details) > 1 and "km" in details[1]:
                    mileage_part = details[1].strip()
                    mileage = self._clean_number(mileage_part)

                # Kraftstoff und Leistung
                if len(details) > 2:
                    fuel_power = details[2].strip().split(',')
                    if len(fuel_power) > 0:
                        fuel = fuel_power[0].strip()
                    if len(fuel_power) > 1 and "kW" in fuel_power[1]:
                        power_part = fuel_power[1].strip()
                        power = self._clean_number(power_part)

            # Erstelle Listing-Objekt
            listing = {
                'titel': title,
                'preis': price,
                'baujahr': year,
                'kilometerstand': mileage,
                'kraftstoff': fuel,
                'leistung_kw': power,
                'url': url,
                'haendler': True,  # Annahme: Alle Listings sind von Händlern
                'standort': None,  # Nicht ohne zusätzliche Extraktion verfügbar
                'datum_erstellt': None  # Nicht direkt verfügbar
            }

            return listing

        except Exception as e:
            logger.error(f"Fehler beim Parsen eines Listings: {str(e)}")
            return None

    def _go_to_next_page(self):
        """
        Navigiert zur nächsten Seite der Suchergebnisse.

        Returns:
            bool: True wenn zur nächsten Seite navigiert wurde, sonst False
        """
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException

        try:
            # Finde den "Weiter"-Button
            next_button = self.driver.find_element(By.CSS_SELECTOR, ".pagination-nav .btn--orange.btn--s")

            if next_button and next_button.is_enabled():
                next_button.click()
                return True
            else:
                return False

        except NoSuchElementException:
            return False

    def _random_wait(self):
        """Wartet eine zufällige Zeit zwischen den Anfragen"""
        wait_time = random.uniform(self.config.MIN_WAIT_SECONDS, self.config.MAX_WAIT_SECONDS)
        time.sleep(wait_time)

    def _clean_price(self, price_str):
        """
        Bereinigt einen Preisstring zu einem numerischen Wert.

        Args:
            price_str (str): Preisstring (z.B. "15.990 €")

        Returns:
            float: Bereinigter Preis
        """
        if not price_str:
            return 0

        # Entferne alle Nicht-Ziffern und Dezimalpunkte/Kommas
        digits_only = ''.join(c for c in price_str if c.isdigit() or c in '.,')

        if not digits_only:
            return 0

        # Ersetze Tausendertrennzeichen und Dezimalkomma
        cleaned = digits_only.replace('.', '').replace(',', '.')

        try:
            return float(cleaned)
        except ValueError:
            return 0

    def _clean_number(self, number_str):
        """
        Bereinigt einen Zahlenstring zu einem numerischen Wert.

        Args:
            number_str (str): Zahlenstring (z.B. "150.000 km")

        Returns:
            int: Bereinigte Zahl
        """
        if not number_str:
            return 0

        # Entferne alle Nicht-Ziffern
        digits_only = ''.join(c for c in number_str if c.isdigit())

        if not digits_only:
            return 0

        try:
            return int(digits_only)
        except ValueError:
            return 0

    def _median(self, numbers):
        """
        Berechnet den Median einer Zahlenreihe.

        Args:
            numbers (list): Liste von Zahlen

        Returns:
            float: Median
        """
        if not numbers:
            return 0

        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)

        if n % 2 == 0:
            return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
        else:
            return sorted_numbers[n//2]


class MarketDataManager:
    """Manager-Klasse, die verschiedene Marktdaten-Provider verwaltet"""

    def __init__(self, config=None):
        """
        Initialisiert den MarketDataManager.

        Args:
            config (MarketDataConfig, optional): Konfiguration für die Provider
        """
        self.config = config or MarketDataConfig()

        # Initialisiere Provider
        self.api_provider = MobileDeAPI(self.config)
        self.scraper_provider = MobileDeScraperProvider(self.config)

    def get_market_data(self, vehicle, force_refresh=False, use_api=None):
        """
        Holt Marktdaten für ein Fahrzeug, mit Fallback von API zu Scraper.

        Args:
            vehicle (dict): Fahrzeugdaten mit Marke, Modell, Baujahr, etc.
            force_refresh (bool): Cache umgehen und frische Daten holen
            use_api (bool, optional): API verwenden (True) oder Scraper (False)

        Returns:
            dict: Marktdaten mit Statistiken und Listings
        """
        logger.info(f"Hole Marktdaten für {vehicle.get('marke')} {vehicle.get('modell')}")

        # Bestimme, ob API verwendet werden soll
        if use_api is None:
            use_api = self.config.USE_MOBILE_API

        # Versuche zuerst mit API, wenn konfiguriert
        if use_api:
            try:
                market_data = self.api_provider.get_market_data(vehicle, force_refresh)
                if market_data and market_data.get('listings'):
                    logger.info(f"Marktdaten über API erhalten: {len(market_data['listings'])} Listings")
                    return market_data
                else:
                    logger.warning("Keine Daten über API erhalten, verwende Fallback-Scraper")
            except Exception as e:
                logger.error(f"Fehler bei der API-Anfrage: {str(e)}")
                logger.warning("Verwende Fallback-Scraper")

        # Fallback oder direkte Verwendung des Scrapers
        try:
            market_data = self.scraper_provider.get_market_data(vehicle, force_refresh)
            if market_data:
                logger.info(f"Marktdaten über Scraper erhalten: {len(market_data['listings'])} Listings")
            else:
                logger.warning("Keine Daten über Scraper erhalten")

            return market_data

        except Exception as e:
            logger.error(f"Fehler beim Scraping: {str(e)}")
            return None

    def get_market_data_for_vehicles(self, vehicles, force_refresh=False, use_api=None):
        """
        Holt Marktdaten für mehrere Fahrzeuge.

        Args:
            vehicles (list): Liste von Fahrzeugdaten
            force_refresh (bool): Cache umgehen und frische Daten holen
            use_api (bool, optional): API verwenden (True) oder Scraper (False)

        Returns:
            dict: Dictionary mit Fahrzeug-IDs als Schlüssel und Marktdaten als Werte
        """
        results = {}

        for vehicle in vehicles:
            # Eindeutiger ID für das Fahrzeug erstellen oder verwenden
            vehicle_id = vehicle.get('id') or f"{vehicle.get('marke')}_{vehicle.get('modell')}_{vehicle.get('baujahr')}"

            try:
                market_data = self.get_market_data(vehicle, force_refresh, use_api)
                if market_data:
                    results[vehicle_id] = market_data
            except Exception as e:
                logger.error(f"Fehler beim Holen der Marktdaten für {vehicle_id}: {str(e)}")

        return results

    def clear_cache(self, vehicle=None):
        """
        Löscht den Cache für ein Fahrzeug oder alle Fahrzeuge.

        Args:
            vehicle (dict, optional): Fahrzeugdaten oder None für alle

        Returns:
            int: Anzahl der gelöschten Cache-Dateien
        """
        # Lösche sowohl API- als auch Scraper-Cache
        api_count = self.api_provider.clear_cache(vehicle)
        scraper_count = self.scraper_provider.clear_cache(vehicle)

        return api_count + scraper_count

    def market_data_to_dataframe(self, market_data):
        """
        Konvertiert Marktdaten in einen DataFrame.

        Args:
            market_data (dict): Marktdaten mit Listings

        Returns:
            pd.DataFrame: DataFrame mit den Listings
        """
        if not market_data or 'listings' not in market_data:
            return pd.DataFrame()

        # Erstelle DataFrame aus Listings
        df = pd.DataFrame(market_data['listings'])

        # Füge Statistiken als Spalten hinzu
        if df.empty:
            return df

        for key in ['quelle', 'timestamp', 'anzahl_angebote', 'marktpreis_min',
                    'marktpreis_max', 'marktpreis_mean', 'marktpreis_median']:
            if key in market_data:
                df[f'stat_{key}'] = market_data[key]

        return df


# Beispielverwendung
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Konfiguration
    config = MarketDataConfig()
    config.CACHE_DIR = "cache/market_data_test"
    config.USE_MOBILE_API = False  # Verwende Scraper für Test

    # Fahrzeugdaten
    test_vehicle = {
        'marke': 'BMW',
        'modell': 'X5',
        'baujahr': 2018,
        'kilometerstand': 75000
    }

    # MarketDataManager initialisieren
    manager = MarketDataManager(config)

    # Marktdaten holen
    market_data = manager.get_market_data(test_vehicle)

    if market_data:
        print(f"Quelle: {market_data['quelle']}")
        print(f"Anzahl Angebote: {market_data['anzahl_angebote']}")
        print(f"Median-Preis: {market_data['marktpreis_median']} €")

        # Erstes Listing anzeigen
        if market_data['listings']:
            first_listing = market_data['listings'][0]
            print("\nErstes Listing:")
            for key, value in first_listing.items():
                print(f"  {key}: {value}")

        # Als DataFrame konvertieren
        df = manager.market_data_to_dataframe(market_data)
        print("\nDataFrame-Info:")
        print(df.info())
    else:
        print("Keine Marktdaten gefunden.")