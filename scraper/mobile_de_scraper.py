# auto_auction_analyzer/scraper/mobile_de_scraper.py
import time
import random
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MobileDeScraperConfig:
    """Konfiguration für den mobile.de Scraper"""
    # Basis-URL für die Suche
    BASE_URL = "https://www.mobile.de/de/fahrzeug/search.html"

    # Wartezeiten in Sekunden
    MIN_WAIT_TIME = 2
    MAX_WAIT_TIME = 5
    PAGE_LOAD_TIMEOUT = 30

    # Maximale Anzahl an Seiten, die durchsucht werden sollen
    MAX_PAGES = 3

    # Maximale Anzahl an Fahrzeugen, die extrahiert werden sollen
    MAX_VEHICLES = 20

    # User-Agent
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

class MobileDeScraperHelper:
    """Helper-Klasse mit häufig verwendeten Methoden"""

    @staticmethod
    def random_wait():
        """Wartet eine zufällige Zeit zwischen MIN_WAIT_TIME und MAX_WAIT_TIME"""
        wait_time = random.uniform(MobileDeScraperConfig.MIN_WAIT_TIME,
                                   MobileDeScraperConfig.MAX_WAIT_TIME)
        time.sleep(wait_time)

    @staticmethod
    def clean_price(price_str):
        """Bereinigt und konvertiert einen Preis-String in eine Zahl"""
        if not price_str:
            return None
        # Entferne alle Nicht-Ziffern
        digits_only = ''.join(filter(str.isdigit, price_str))
        if digits_only:
            return int(digits_only)
        return None

class MobileDeScraper:
    """Scraper für mobile.de"""

    def __init__(self, headless=True):
        """
        Initialisiert den Scraper.

        Args:
            headless (bool): Ob der Browser im Headless-Modus laufen soll
        """
        self.config = MobileDeScraperConfig()
        self.helper = MobileDeScraperHelper()
        self.driver = None
        self.headless = headless

    def _setup_driver(self):
        """Richtet den Selenium WebDriver ein"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")

        chrome_options.add_argument(f"user-agent={self.config.USER_AGENT}")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.set_page_load_timeout(self.config.PAGE_LOAD_TIMEOUT)

    def _quit_driver(self):
        """Beendet den WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def build_search_url(self, marke, modell, baujahr=None, max_kilometer=None):
        """
        Erstellt eine Suchanfrage-URL für mobile.de

        Args:
            marke (str): Fahrzeugmarke
            modell (str): Fahrzeugmodell
            baujahr (int, optional): Mindestbaujahr
            max_kilometer (int, optional): Maximaler Kilometerstand

        Returns:
            str: Die generierte Suchanfrage-URL
        """
        # Basisparameter
        params = []

        # Marke hinzufügen (muss URL-encoded sein)
        if marke:
            params.append(f"makeModelVariant1.makeId={self._get_make_id(marke)}")

        # Modell hinzufügen wenn vorhanden
        if modell:
            params.append(f"makeModelVariant1.modelDescription={modell}")

        # Baujahr hinzufügen wenn vorhanden
        if baujahr:
            params.append(f"minFirstRegistrationDate={baujahr}")

        # Kilometerstand hinzufügen wenn vorhanden
        if max_kilometer:
            params.append(f"maxMileage={max_kilometer}")

        # URL zusammensetzen
        url = self.config.BASE_URL
        if params:
            url += "?" + "&".join(params)

        return url

    def _get_make_id(self, marke):
        """
        Konvertiert einen Markennamen in die entsprechende mobile.de-ID

        Diese Funktion enthält eine kleine Auswahl gängiger Marken.
        In einer vollständigen Implementierung sollte dies erweitert werden.

        Args:
            marke (str): Name der Fahrzeugmarke

        Returns:
            int: Die mobile.de-ID für die Marke
        """
        # Mapping von Markennamen zu mobile.de-IDs
        marken_mapping = {
            "audi": 1900,
            "bmw": 3500,
            "ford": 9000,
            "mercedes-benz": 17200,
            "mercedes": 17200,
            "opel": 19000,
            "volkswagen": 25200,
            "vw": 25200,
            "porsche": 20000,
            "toyota": 24100,
            "volvo": 25100
        }

        # Marke normalisieren und im Mapping suchen
        normalized_marke = marke.lower().strip()
        return marken_mapping.get(normalized_marke, 0)

    def scrape_listings(self, marke, modell, baujahr=None, max_kilometer=None):
        """
        Durchsucht mobile.de nach Fahrzeugangeboten basierend auf den Suchkriterien.

        Args:
            marke (str): Fahrzeugmarke
            modell (str): Fahrzeugmodell
            baujahr (int, optional): Mindestbaujahr
            max_kilometer (int, optional): Maximaler Kilometerstand

        Returns:
            pd.DataFrame: DataFrame mit gefundenen Fahrzeugangeboten
        """
        logger.info(f"Suche nach Fahrzeugen: {marke} {modell}, Baujahr >= {baujahr}, km <= {max_kilometer}")

        search_url = self.build_search_url(marke, modell, baujahr, max_kilometer)
        logger.info(f"Generierte Such-URL: {search_url}")

        try:
            self._setup_driver()
            self.driver.get(search_url)

            # Warten auf die Seite und Cookie-Banner akzeptieren
            self._handle_cookie_banner()

            # Sammle Fahrzeugangebote
            vehicles = []
            pages_scraped = 0

            while pages_scraped < self.config.MAX_PAGES and len(vehicles) < self.config.MAX_VEHICLES:
                # Extrahiere Fahrzeuge von der aktuellen Seite
                page_vehicles = self._extract_vehicles_from_page()
                vehicles.extend(page_vehicles)

                logger.info(f"Seite {pages_scraped + 1} verarbeitet, {len(page_vehicles)} Fahrzeuge gefunden")

                # Zur nächsten Seite navigieren, falls vorhanden
                if not self._go_to_next_page():
                    break

                pages_scraped += 1

            # DataFrame erstellen
            if vehicles:
                df = pd.DataFrame(vehicles)
                logger.info(f"Scraping abgeschlossen. {len(df)} Fahrzeuge gefunden.")
                return df
            else:
                logger.warning("Keine Fahrzeuge gefunden.")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Fehler beim Scrapen: {str(e)}")
            return pd.DataFrame()

        finally:
            self._quit_driver()

    def _handle_cookie_banner(self):
        """Behandelt das Cookie-Banner, falls es erscheint"""
        try:
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "mde-consent-accept-btn"))
            ).click()
            logger.debug("Cookie-Banner akzeptiert")
            self.helper.random_wait()
        except (TimeoutException, NoSuchElementException):
            logger.debug("Kein Cookie-Banner gefunden oder nicht klickbar")

    def _extract_vehicles_from_page(self):
        """
        Extrahiert Fahrzeuginformationen von der aktuellen Seite.

        Returns:
            list: Liste von Fahrzeug-Dictionaries
        """
        vehicles = []

        try:
            # Warten, bis die Fahrzeugliste geladen ist
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "cBox-body--resultlist"))
            )

            # Finde alle Fahrzeug-Listings
            listings = self.driver.find_elements(By.CSS_SELECTOR, ".cBox-body--resultlist .cBox-body--vehicleDetails")

            for listing in listings:
                vehicle_data = self._parse_vehicle_listing(listing)
                if vehicle_data:
                    vehicles.append(vehicle_data)

                # Maximale Anzahl prüfen
                if len(vehicles) >= self.config.MAX_VEHICLES:
                    break

            return vehicles

        except Exception as e:
            logger.error(f"Fehler beim Extrahieren der Fahrzeuge: {str(e)}")
            return vehicles

    def _parse_vehicle_listing(self, listing_element):
        """
        Extrahiert Daten aus einem einzelnen Fahrzeug-Listing.

        Args:
            listing_element (WebElement): Das Selenium WebElement des Listings

        Returns:
            dict: Extrahierte Fahrzeugdaten
        """
        try:
            # Titel (enthält Marke und Modell)
            title_element = listing_element.find_element(By.CSS_SELECTOR, "h2.title")
            title = title_element.text.strip() if title_element else ""

            # Preis
            price_element = listing_element.find_element(By.CSS_SELECTOR, ".vehicle-data .pricePrimaryCountup")
            price_text = price_element.text.strip() if price_element else ""
            price = self.helper.clean_price(price_text)

            # Details (Baujahr, Kilometerstand, etc.)
            details_element = listing_element.find_element(By.CSS_SELECTOR, ".vehicle-data .container .rbt-regMilPow")
            details_text = details_element.text.strip() if details_element else ""

            # URL des Angebots
            url_element = title_element.find_element(By.XPATH, "./..") if title_element else None
            url = url_element.get_attribute("href") if url_element else ""

            # Parsen der Details
            year = None
            mileage = None
            fuel_type = None
            power = None

            if details_text:
                details = details_text.split(',')

                # Baujahr extrahieren (erstes Element, Format: "EZ 01/2020")
                if len(details) > 0 and "EZ" in details[0]:
                    year_part = details[0].replace("EZ", "").strip()
                    if "/" in year_part:
                        year = int(year_part.split('/')[1])

                # Kilometerstand extrahieren (zweites Element, Format: "50.000 km")
                if len(details) > 1 and "km" in details[1]:
                    mileage_part = details[1].strip()
                    mileage = self.helper.clean_price(mileage_part)

                # Kraftstofftyp und Leistung (drittes Element, Format: "Benzin, 100 kW")
                if len(details) > 2:
                    power_fuel = details[2].strip().split(',')
                    if len(power_fuel) > 0:
                        fuel_type = power_fuel[0].strip()
                    if len(power_fuel) > 1 and "kW" in power_fuel[1]:
                        power_part = power_fuel[1].strip()
                        power = int(''.join(filter(str.isdigit, power_part)))

            # Fahrzeugdaten zusammenstellen
            vehicle_data = {
                'title': title,
                'marktpreis': price,
                'baujahr': year,
                'kilometerstand': mileage,
                'kraftstoff': fuel_type,
                'leistung_kw': power,
                'url': url
            }

            return vehicle_data

        except Exception as e:
            logger.error(f"Fehler beim Parsen eines Listings: {str(e)}")
            return None

    def _go_to_next_page(self):
        """
        Navigiert zur nächsten Seite, falls vorhanden.

        Returns:
            bool: True wenn zur nächsten Seite navigiert wurde, sonst False
        """
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, ".pagination-nav .btn--orange.btn--s")

            if next_button and next_button.is_enabled():
                next_button.click()

                # Warten, bis die neue Seite geladen ist
                WebDriverWait(self.driver, 10).until(
                    EC.staleness_of(next_button)
                )

                self.helper.random_wait()
                return True
            else:
                return False

        except (TimeoutException, NoSuchElementException):
            return False

# Beispielverwendung
if __name__ == "__main__":
    scraper = MobileDeScraper(headless=True)
    df = scraper.scrape_listings("BMW", "X5", baujahr=2018, max_kilometer=100000)

    if not df.empty:
        print(df.head())
        df.to_csv("mobile_de_listings.csv", index=False)