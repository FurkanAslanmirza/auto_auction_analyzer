# auto_auction_analyzer/ai_integration/deepseek_client.py
import json
import requests
import pandas as pd
import logging
import time
from pathlib import Path
import subprocess
import os
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekConfig:
    """Konfiguration für die DeepSeek-R1 Integration"""
    # API-Einstellungen
    API_URL = "http://localhost:11434/api"

    # Modellspezifikation
    MODEL_NAME = "deepseek-r1:14b"  # Default

    # Systemanforderungen
    REQUIRED_RAM_GB = 16
    REQUIRED_GPU_VRAM_GB = 8

    # Prompts und Templates
    SYSTEM_PROMPT_TEMPLATE = """Du bist ein Experte für Fahrzeugbewertung und Profitabilitätsanalyse auf dem deutschen Markt. 
    Deine Aufgabe ist es, basierend auf den gegebenen Daten, eine fundierte Bewertung und Empfehlung für den Weiterverkauf von Fahrzeugen zu geben.
    Analysiere die Daten sachlich und gib konkrete, auf Daten basierende Empfehlungen.
    
    Die Daten enthalten folgende Informationen:
    - Fahrzeugdaten (Marke, Modell, Baujahr, Kilometerstand)
    - Auktionspreis (der Preis, zu dem das Fahrzeug ersteigert wurde)
    - Marktpreise (durchschnittlicher, medianer, minimaler und maximaler Preis auf dem Markt)
    - Berechnete Profitabilitätskennzahlen (Nettogewinn, Gewinnmarge, ROI)
    
    Berücksichtige in deiner Analyse besonders folgende Faktoren:
    - Wettbewerbssituation auf dem Markt (Anzahl der Angebote)
    - Alter und Zustand des Fahrzeugs
    - Potenzielle versteckte Kosten oder Risiken
    - Saisonale Markttrends und regionale Besonderheiten
    - Fahrzeugkategorie und Zielkundensegment
    
    Präsentiere deine Ergebnisse in einem klar strukturierten Format mit:
    1. Knapper Zusammenfassung der Gesamtsituation
    2. Fahrzeugbewertung mit Profitabilitätseinschätzung
    3. Konkrete Handlungsempfehlungen mit Begründung
    4. Potenziellen Risiken und Chancen
    """

    VEHICLE_PROMPT_TEMPLATE = """Bitte bewerte das folgende Fahrzeug hinsichtlich seiner Profitabilität für den Weiterverkauf:
    
    FAHRZEUGDATEN:
    {vehicle_data}
    
    MARKTDATEN:
    {market_data}
    
    PROFITABILITÄTSANALYSE:
    {profitability_data}
    
    Gib eine fundierte Bewertung zur Profitabilität und konkrete Handlungsempfehlungen. Beziehe die Marktlage, Fahrzeugeigenschaften und berechnete Kennzahlen in deine Analyse ein.
    """

    SUMMARY_PROMPT_TEMPLATE = """Bitte erstelle eine zusammenfassende Analyse und Empfehlung für die folgende Fahrzeugsammlung:
    
    ANALYSEDATEN:
    {summary_data}
    
    PROFITABILITÄTSVERTEILUNG:
    {profitability_distribution}
    
    Erstelle einen Bericht mit:
    1. Übersicht der Gesamtsituation und genereller Markttrends
    2. Identifikation der vielversprechendsten Fahrzeuge für den Weiterverkauf
    3. Allgemeine Empfehlungen für den Ankauf und Weiterverkauf
    4. Einschätzung der Marktlage und Gewinnpotential
    """

class DeepSeekClient:
    """Client für die Interaktion mit dem DeepSeek-R1 Modell über Ollama"""

    def __init__(self, model_name=None, ensure_running=True):
        """
        Initialisiert den DeepSeek Client.

        Args:
            model_name (str, optional): Name des zu verwendenden Modells
            ensure_running (bool): Ob automatisch überprüft werden soll, dass Ollama läuft
        """
        self.config = DeepSeekConfig()

        # Modellname überschreiben, falls angegeben
        if model_name:
            self.config.MODEL_NAME = model_name

        self.ensure_running = ensure_running

        if ensure_running:
            self._check_ollama_running()

    def _check_ollama_running(self):
        """
        Überprüft, ob der Ollama-Server läuft und das Modell verfügbar ist.

        Returns:
            bool: True wenn Ollama läuft und das Modell verfügbar ist, sonst False
        """
        try:
            # Überprüfe, ob der Ollama-Server erreichbar ist
            response = requests.get(f"{self.config.API_URL}/tags")

            if response.status_code != 200:
                logger.error("Ollama-Server ist nicht erreichbar.")
                return self._start_ollama()

            # Überprüfe, ob das Modell verfügbar ist
            models = response.json().get("models", [])
            model_exists = any(model.get("name") == self.config.MODEL_NAME for model in models)

            if not model_exists:
                logger.warning(f"Modell {self.config.MODEL_NAME} ist nicht verfügbar.")
                # Modell wird automatisch heruntergeladen
                return self._pull_model()

            logger.info(f"Ollama-Server läuft und Modell {self.config.MODEL_NAME} ist verfügbar.")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Fehler bei der Überprüfung des Ollama-Servers: {str(e)}")
            return self._start_ollama()

    def _start_ollama(self):
        """
        Versucht, Ollama zu starten.

        Returns:
            bool: True wenn Ollama erfolgreich gestartet wurde, sonst False
        """
        try:
            logger.info("Versuche, Ollama zu starten...")

            # Ermittle den Betriebssystemtyp
            if os.name == 'nt':  # Windows
                start_cmd = ["start", "cmd", "/c", "ollama", "serve"]
                subprocess.Popen(start_cmd, shell=True)
            else:  # Linux/macOS
                start_cmd = ["ollama", "serve"]
                subprocess.Popen(start_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Warte, bis der Server gestartet ist
            for i in range(10):
                time.sleep(2)
                try:
                    response = requests.get(f"{self.config.API_URL}/tags")
                    if response.status_code == 200:
                        logger.info("Ollama-Server wurde erfolgreich gestartet.")
                        return self._pull_model()
                except:
                    pass
                logger.info(f"Warte auf Ollama-Server... ({i+1}/10)")

            logger.error("Ollama-Server konnte nicht gestartet werden.")
            return False

        except Exception as e:
            logger.error(f"Fehler beim Starten von Ollama: {str(e)}")
            return False

    def _pull_model(self):
        """
        Lädt das DeepSeek-R1 Modell herunter, falls es nicht vorhanden ist.

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            logger.info(f"Lade Modell {self.config.MODEL_NAME} herunter...")
            pull_cmd = ["ollama", "pull", self.config.MODEL_NAME]
            result = subprocess.run(pull_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Fehler beim Herunterladen des Modells: {result.stderr}")
                return False

            logger.info(f"Modell {self.config.MODEL_NAME} wurde erfolgreich heruntergeladen.")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Herunterladen des Modells: {str(e)}")
            return False

    def _generate_response(self, prompt, system_prompt=None, max_retries=3):
        """
        Generiert eine Antwort vom DeepSeek-R1 Modell.

        Args:
            prompt (str): Der Prompt für das Modell
            system_prompt (str, optional): Der System-Prompt
            max_retries (int): Maximale Anzahl an Wiederholungsversuchen

        Returns:
            str: Die generierte Antwort oder None bei Fehler
        """
        if self.ensure_running and not self._check_ollama_running():
            logger.error("Ollama ist nicht verfügbar. Abbruch.")
            return None

        # Definiere den Request-Body
        request_data = {
            "model": self.config.MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }

        if system_prompt:
            request_data["system"] = system_prompt

        # Sende die Anfrage mit Wiederholungsversuchen
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.config.API_URL}/generate",
                    json=request_data,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Fehler bei der API-Anfrage: {response.status_code}, {response.text}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Netzwerkfehler bei der API-Anfrage (Versuch {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(2)  # Kurze Pause vor dem nächsten Versuch

        logger.error(f"Alle {max_retries} Versuche sind fehlgeschlagen.")
        return None

    def analyze_vehicle(self, vehicle_data):
        """
        Analysiert ein einzelnes Fahrzeug mit dem DeepSeek-R1 Modell.

        Args:
            vehicle_data (dict): Fahrzeugdaten

        Returns:
            str: Die generierte Analyse
        """
        # Formatiere die Fahrzeugdaten für den Prompt
        vehicle_info = f"""Marke: {vehicle_data.get('marke', 'Unbekannt')}
Modell: {vehicle_data.get('modell', 'Unbekannt')}
Baujahr: {vehicle_data.get('baujahr', 'Unbekannt')}
Kilometerstand: {vehicle_data.get('kilometerstand', 'Unbekannt')} km
Auktionspreis: {vehicle_data.get('auktionspreis', 'Unbekannt')} €"""

        market_info = f"""Durchschnittlicher Marktpreis: {vehicle_data.get('marktpreis_mean', 'Unbekannt')} €
Medianer Marktpreis: {vehicle_data.get('marktpreis_median', 'Unbekannt')} €
Minimaler Marktpreis: {vehicle_data.get('marktpreis_min', 'Unbekannt')} €
Maximaler Marktpreis: {vehicle_data.get('marktpreis_max', 'Unbekannt')} €
Anzahl Angebote: {vehicle_data.get('anzahl_angebote', 'Unbekannt')}"""

        profitability_info = f"""Gesamtkosten: {vehicle_data.get('gesamtkosten', 'Unbekannt')} €
Renovierungskosten: {vehicle_data.get('renovierungskosten', 'Unbekannt')} €
Steuern und Gebühren: {vehicle_data.get('steuern_gebuehren', 'Unbekannt')} €
Nettogewinn: {vehicle_data.get('nettogewinn', 'Unbekannt')} €
Gewinnmarge: {vehicle_data.get('gewinnmarge_prozent', 'Unbekannt')}%
ROI: {vehicle_data.get('roi', 'Unbekannt')}%
Profitabilitätskategorie: {vehicle_data.get('profitabilitaet', 'Unbekannt')}"""

        # Erstelle den Prompt
        prompt = self.config.VEHICLE_PROMPT_TEMPLATE.format(
            vehicle_data=vehicle_info,
            market_data=market_info,
            profitability_data=profitability_info
        )

        # Generiere die Antwort
        logger.info(f"Analysiere Fahrzeug: {vehicle_data.get('marke', '')} {vehicle_data.get('modell', '')}")
        response = self._generate_response(prompt, self.config.SYSTEM_PROMPT_TEMPLATE)

        if response:
            logger.info(f"Fahrzeuganalyse für {vehicle_data.get('marke', '')} {vehicle_data.get('modell', '')} erfolgreich generiert.")
            return response
        else:
            logger.error("Fehler bei der Fahrzeuganalyse.")
            return "Fehler: Die Analyse konnte nicht erstellt werden."

    def generate_summary_report(self, summary_data, profitability_distribution):
        """
        Generiert einen zusammenfassenden Bericht für mehrere Fahrzeuge.

        Args:
            summary_data (dict): Zusammenfassungsdaten
            profitability_distribution (dict): Verteilung der Profitabilitätskategorien

        Returns:
            str: Der generierte Bericht
        """
        # Formatiere die Zusammenfassungsdaten für den Prompt
        summary_info = f"""Gesamtanzahl Fahrzeuge: {summary_data.get('gesamtanzahl_fahrzeuge', 0)}
Profitable Fahrzeuge: {summary_data.get('profitable_fahrzeuge', 0)} ({summary_data.get('profitable_prozent', 0):.1f}%)
Sehr profitable Fahrzeuge: {summary_data.get('sehr_profitable_fahrzeuge', 0)} ({summary_data.get('sehr_profitable_prozent', 0):.1f}%)
Unprofitable Fahrzeuge: {summary_data.get('unprofitable_fahrzeuge', 0)} ({summary_data.get('unprofitable_prozent', 0):.1f}%)
Durchschnittliche Gewinnmarge: {summary_data.get('durchschnittliche_gewinnmarge', 0):.2f}%
Durchschnittlicher ROI: {summary_data.get('durchschnittlicher_roi', 0):.2f}%
Gesamtgewinn: {summary_data.get('gesamtgewinn', 0):.2f} €
Gesamtinvestition: {summary_data.get('gesamtinvestition', 0):.2f} €
Gesamt-ROI: {summary_data.get('gesamt_roi', 0):.2f}%

Bestes Fahrzeug:
Marke: {summary_data.get('bestes_fahrzeug', {}).get('marke', 'Unbekannt')}
Modell: {summary_data.get('bestes_fahrzeug', {}).get('modell', 'Unbekannt')}
Baujahr: {summary_data.get('bestes_fahrzeug', {}).get('baujahr', 'Unbekannt')}
Gewinnmarge: {summary_data.get('bestes_fahrzeug', {}).get('gewinnmarge', 0):.2f}%
Nettogewinn: {summary_data.get('bestes_fahrzeug', {}).get('nettogewinn', 0):.2f} €

Schlechtestes Fahrzeug:
Marke: {summary_data.get('schlechtestes_fahrzeug', {}).get('marke', 'Unbekannt')}
Modell: {summary_data.get('schlechtestes_fahrzeug', {}).get('modell', 'Unbekannt')}
Baujahr: {summary_data.get('schlechtestes_fahrzeug', {}).get('baujahr', 'Unbekannt')}
Gewinnmarge: {summary_data.get('schlechtestes_fahrzeug', {}).get('gewinnmarge', 0):.2f}%
Nettogewinn: {summary_data.get('schlechtestes_fahrzeug', {}).get('nettogewinn', 0):.2f} €

Saisonale Empfehlung: {summary_data.get('saisonale_empfehlung', 'Keine Information')}"""

        # Formatiere die Verteilung der Profitabilitätskategorien
        distribution_lines = []
        for category, count in profitability_distribution.items():
            distribution_lines.append(f"{category}: {count} Fahrzeuge")

        distribution_info = "\n".join(distribution_lines)

        # Erstelle den Prompt
        prompt = self.config.SUMMARY_PROMPT_TEMPLATE.format(
            summary_data=summary_info,
            profitability_distribution=distribution_info
        )

        # Generiere die Antwort
        logger.info("Generiere zusammenfassenden Bericht...")
        response = self._generate_response(prompt, self.config.SYSTEM_PROMPT_TEMPLATE)

        if response:
            logger.info("Zusammenfassender Bericht erfolgreich generiert.")
            return response
        else:
            logger.error("Fehler bei der Generierung des zusammenfassenden Berichts.")
            return "Fehler: Der Bericht konnte nicht erstellt werden."

    def analyze_all_vehicles(self, vehicles_df):
        """
        Analysiert alle Fahrzeuge in einem DataFrame.

        Args:
            vehicles_df (pd.DataFrame): DataFrame mit Fahrzeugdaten

        Returns:
            dict: Dictionary mit Fahrzeug-IDs als Schlüssel und Analysen als Werte
        """
        if vehicles_df is None or vehicles_df.empty:
            logger.warning("Keine Fahrzeugdaten für die Analyse vorhanden.")
            return {}

        results = {}

        for idx, row in vehicles_df.iterrows():
            vehicle_id = f"{row.get('marke', 'Unbekannt')}_{row.get('modell', 'Unbekannt')}_{idx}"
            vehicle_data = row.to_dict()

            analysis = self.analyze_vehicle(vehicle_data)
            results[vehicle_id] = {
                'vehicle_data': vehicle_data,
                'analysis': analysis
            }

        return results

    def check_system_requirements(self):
        """
        Überprüft, ob das System die Anforderungen für das DeepSeek-R1 Modell erfüllt.

        Returns:
            dict: Dictionary mit Informationen zu den Systemanforderungen
        """
        try:
            # RAM überprüfen
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            ram_sufficient = ram_gb >= self.config.REQUIRED_RAM_GB

            # GPU überprüfen
            gpu_info = []
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    for gpu in gpus:
                        gpu_info.append({
                            'name': gpu.name,
                            'memory_total_gb': round(gpu.memoryTotal / 1024, 2),
                            'sufficient': gpu.memoryTotal / 1024 >= self.config.REQUIRED_GPU_VRAM_GB
                        })
                gpu_sufficient = any(info['sufficient'] for info in gpu_info) if gpu_info else False
            except:
                gpu_info = [{'name': 'Keine GPU erkannt', 'memory_total_gb': 0, 'sufficient': False}]
                gpu_sufficient = False

            return {
                'ram_gb': ram_gb,
                'ram_sufficient': ram_sufficient,
                'required_ram_gb': self.config.REQUIRED_RAM_GB,
                'gpu_info': gpu_info,
                'gpu_sufficient': gpu_sufficient,
                'required_gpu_vram_gb': self.config.REQUIRED_GPU_VRAM_GB,
                'overall_sufficient': ram_sufficient and gpu_sufficient
            }

        except Exception as e:
            logger.error(f"Fehler bei der Überprüfung der Systemanforderungen: {str(e)}")
            return {
                'error': str(e),
                'overall_sufficient': False
            }

# Beispielverwendung
if __name__ == "__main__":
    # Importiere die erforderlichen Module für den Test
    import json
    from pathlib import Path

    # Einfacher, schneller Test des Clients
    client = DeepSeekClient(model_name="deepseek-r1:14b")  # Wähle ein kleineres Modell für schnelleren Test

    # Überprüfe Systemanforderungen
    system_req = client.check_system_requirements()
    print(f"Systemanforderungen erfüllt: {system_req.get('overall_sufficient', False)}")
    print(f"RAM: {system_req.get('ram_gb', 0)} GB (erforderlich: {system_req.get('required_ram_gb', 0)} GB)")

    # Einen Test-Prompt senden
    test_response = client._generate_response("Gib eine kurze Übersicht über den deutschen Automarkt.")

    if test_response:
        print("\nTest erfolgreich. Antwort:")
        print(test_response[:200] + "..." if len(test_response) > 200 else test_response)
    else:
        print("\nTest fehlgeschlagen.")