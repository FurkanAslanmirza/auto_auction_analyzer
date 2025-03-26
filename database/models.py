# auto_auction_analyzer/database/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import datetime
import json
import logging
import os
from pathlib import Path

Base = declarative_base()
logger = logging.getLogger(__name__)

class Vehicle(Base):
    """Repräsentiert ein Fahrzeug aus einer Auktion"""
    __tablename__ = 'vehicles'

    id = Column(Integer, primary_key=True)

    # Basisdaten
    marke = Column(String(50), nullable=False, index=True)
    modell = Column(String(100), nullable=False, index=True)
    variante = Column(String(100))
    baujahr = Column(Integer, index=True)
    leistung = Column(Integer)  # PS
    leistung_kw = Column(Integer)  # kW
    kilometerstand = Column(Integer)
    kraftstoff = Column(String(20))
    getriebe = Column(String(20))

    # Auktionsdaten
    auktionspreis = Column(Float)
    auktionsdatum = Column(DateTime)
    auktionsquelle = Column(String(100))
    losnummer = Column(String(20))
    fahrgestellnummer = Column(String(50), unique=True)
    auktionskatalog = Column(String(200))

    # Zusatzdaten
    farbe = Column(String(50))
    ausstattung = Column(Text)
    zustand = Column(String(20))
    bemerkungen = Column(Text)

    # Metadaten
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Beziehungen
    market_data = relationship("MarketData", back_populates="vehicle", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="vehicle", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Vehicle(id={self.id}, marke='{self.marke}', modell='{self.modell}', baujahr={self.baujahr})>"

    def to_dict(self):
        """Konvertiert das Fahrzeug in ein Dictionary"""
        return {
            'id': self.id,
            'marke': self.marke,
            'modell': self.modell,
            'variante': self.variante,
            'baujahr': self.baujahr,
            'leistung': self.leistung,
            'leistung_kw': self.leistung_kw,
            'kilometerstand': self.kilometerstand,
            'kraftstoff': self.kraftstoff,
            'getriebe': self.getriebe,
            'auktionspreis': self.auktionspreis,
            'auktionsdatum': self.auktionsdatum.isoformat() if self.auktionsdatum else None,
            'auktionsquelle': self.auktionsquelle,
            'losnummer': self.losnummer,
            'fahrgestellnummer': self.fahrgestellnummer,
            'auktionskatalog': self.auktionskatalog,
            'farbe': self.farbe,
            'ausstattung': self.ausstattung,
            'zustand': self.zustand,
            'bemerkungen': self.bemerkungen,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class MarketData(Base):
    """Marktdaten für ein Fahrzeug von mobile.de oder einer anderen Quelle"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=False, index=True)

    # Marktdaten
    quelle = Column(String(50), nullable=False)  # mobile.de, autoscout24, etc.
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    anzahl_angebote = Column(Integer)

    # Preisstatistiken
    marktpreis_min = Column(Float)
    marktpreis_max = Column(Float)
    marktpreis_mean = Column(Float)
    marktpreis_median = Column(Float)

    # Rohdaten
    raw_data = Column(Text)  # JSON-String mit allen Rohdaten

    # Beziehung
    vehicle = relationship("Vehicle", back_populates="market_data")
    listings = relationship("MarketListing", back_populates="market_data", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<MarketData(id={self.id}, vehicle_id={self.vehicle_id}, quelle='{self.quelle}')>"

    def to_dict(self):
        """Konvertiert die Marktdaten in ein Dictionary"""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'quelle': self.quelle,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'anzahl_angebote': self.anzahl_angebote,
            'marktpreis_min': self.marktpreis_min,
            'marktpreis_max': self.marktpreis_max,
            'marktpreis_mean': self.marktpreis_mean,
            'marktpreis_median': self.marktpreis_median
        }

    def get_raw_data(self):
        """Gibt die Rohdaten als Python-Objekt zurück"""
        if self.raw_data:
            try:
                return json.loads(self.raw_data)
            except json.JSONDecodeError:
                return {}
        return {}


class MarketListing(Base):
    """Einzelnes Fahrzeugangebot aus einer Marktdatenquelle"""
    __tablename__ = 'market_listings'

    id = Column(Integer, primary_key=True)
    market_data_id = Column(Integer, ForeignKey('market_data.id'), nullable=False, index=True)

    # Listing-Daten
    titel = Column(String(200))
    preis = Column(Float)
    baujahr = Column(Integer)
    kilometerstand = Column(Integer)
    kraftstoff = Column(String(20))
    leistung_kw = Column(Integer)

    # Link und Metadaten
    url = Column(String(500))
    haendler = Column(Boolean, default=False)  # True für Händler, False für Privat
    standort = Column(String(100))
    datum_erstellt = Column(DateTime)

    # Beziehung
    market_data = relationship("MarketData", back_populates="listings")

    def __repr__(self):
        return f"<MarketListing(id={self.id}, titel='{self.titel}', preis={self.preis})>"

    def to_dict(self):
        """Konvertiert das Listing in ein Dictionary"""
        return {
            'id': self.id,
            'market_data_id': self.market_data_id,
            'titel': self.titel,
            'preis': self.preis,
            'baujahr': self.baujahr,
            'kilometerstand': self.kilometerstand,
            'kraftstoff': self.kraftstoff,
            'leistung_kw': self.leistung_kw,
            'url': self.url,
            'haendler': self.haendler,
            'standort': self.standort,
            'datum_erstellt': self.datum_erstellt.isoformat() if self.datum_erstellt else None
        }


class AnalysisResult(Base):
    """Ergebnisse der Profitabilitätsanalyse für ein Fahrzeug"""
    __tablename__ = 'analysis_results'

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=False, index=True)

    # Zeitstempel
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Analyseparameter
    min_profit_margin = Column(Float)
    min_profit_amount = Column(Float)

    # Analyseergebnisse
    renovierungskosten = Column(Float)
    steuern_gebuehren = Column(Float)
    gesamtkosten = Column(Float)
    nettogewinn = Column(Float)
    gewinnmarge_prozent = Column(Float)
    roi = Column(Float)
    profitabilitaet = Column(String(20))  # z.B. "Sehr profitabel", "Profitabel", "Geringer Gewinn", "Verlustgeschäft"

    # KI-Analyse
    ai_analysis = Column(Text)

    # Beziehung
    vehicle = relationship("Vehicle", back_populates="analysis_results")

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, vehicle_id={self.vehicle_id}, profitabilitaet='{self.profitabilitaet}')>"

    def to_dict(self):
        """Konvertiert das Analyseergebnis in ein Dictionary"""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'min_profit_margin': self.min_profit_margin,
            'min_profit_amount': self.min_profit_amount,
            'renovierungskosten': self.renovierungskosten,
            'steuern_gebuehren': self.steuern_gebuehren,
            'gesamtkosten': self.gesamtkosten,
            'nettogewinn': self.nettogewinn,
            'gewinnmarge_prozent': self.gewinnmarge_prozent,
            'roi': self.roi,
            'profitabilitaet': self.profitabilitaet,
            'ai_analysis': self.ai_analysis
        }


# auto_auction_analyzer/database/db_manager.py
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta

from .models import Base, Vehicle, MarketData, MarketListing, AnalysisResult

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manager für Datenbank-Operationen"""

    def __init__(self, db_path=None, echo=False):
        """
        Initialisiert den DatabaseManager.

        Args:
            db_path (str, optional): Pfad zur Datenbankdatei (SQLite) oder Verbindungsstring
            echo (bool): SQL-Logging aktivieren
        """
        self.db_path = db_path or os.environ.get('DB_CONNECTION', 'sqlite:///auto_auction.db')
        self.engine = create_engine(self.db_path, echo=echo)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

        # Stelle sicher, dass das Verzeichnis existiert (für SQLite)
        if self.db_path.startswith('sqlite:///'):
            db_file = self.db_path.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(db_file), exist_ok=True)

    def init_db(self):
        """Initialisiert die Datenbankstruktur"""
        Base.metadata.create_all(self.engine)
        logger.info("Datenbankstruktur initialisiert")

    def drop_all(self):
        """Löscht alle Tabellen (Vorsicht!)"""
        Base.metadata.drop_all(self.engine)
        logger.info("Alle Tabellen gelöscht")

    def save_vehicle(self, vehicle_data):
        """
        Speichert ein Fahrzeug in der Datenbank.

        Args:
            vehicle_data (dict): Fahrzeugdaten

        Returns:
            Vehicle: Das gespeicherte Fahrzeugobjekt
        """
        session = self.Session()
        try:
            # Prüfe, ob das Fahrzeug bereits existiert (anhand der Fahrgestellnummer)
            existing_vehicle = None
            if 'fahrgestellnummer' in vehicle_data and vehicle_data['fahrgestellnummer']:
                existing_vehicle = session.query(Vehicle).filter_by(
                    fahrgestellnummer=vehicle_data['fahrgestellnummer']
                ).first()

            # Wenn kein Treffer über Fahrgestellnummer, versuche über Marke, Modell, Baujahr, Kilometerstand
            if not existing_vehicle and all(k in vehicle_data for k in ['marke', 'modell', 'baujahr', 'kilometerstand']):
                existing_vehicle = session.query(Vehicle).filter_by(
                    marke=vehicle_data['marke'],
                    modell=vehicle_data['modell'],
                    baujahr=vehicle_data['baujahr'],
                    kilometerstand=vehicle_data['kilometerstand']
                ).first()

            if existing_vehicle:
                # Aktualisiere vorhandenes Fahrzeug
                for key, value in vehicle_data.items():
                    if hasattr(existing_vehicle, key):
                        setattr(existing_vehicle, key, value)

                vehicle = existing_vehicle
                logger.info(f"Fahrzeug aktualisiert: {vehicle.marke} {vehicle.modell} ({vehicle.id})")
            else:
                # Erstelle neues Fahrzeug
                vehicle = Vehicle(**vehicle_data)
                session.add(vehicle)
                logger.info(f"Neues Fahrzeug erstellt: {vehicle.marke} {vehicle.modell}")

            session.commit()
            return vehicle

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Fehler beim Speichern des Fahrzeugs: {str(e)}")
            raise

        finally:
            session.close()

    def save_vehicles_from_dataframe(self, df):
        """
        Speichert Fahrzeuge aus einem DataFrame in der Datenbank.

        Args:
            df (pd.DataFrame): DataFrame mit Fahrzeugdaten

        Returns:
            list: Liste der gespeicherten Fahrzeug-IDs
        """
        if df is None or df.empty:
            logger.warning("Leerer DataFrame, keine Fahrzeuge zu speichern")
            return []

        vehicle_ids = []

        for _, row in df.iterrows():
            try:
                # Konvertiere Pandas-Zeile in Dictionary
                vehicle_data = row.to_dict()

                # Bereinige NaN-Werte
                for key, value in vehicle_data.items():
                    if isinstance(value, float) and np.isnan(value):
                        vehicle_data[key] = None

                # Speichere Fahrzeug
                vehicle = self.save_vehicle(vehicle_data)
                vehicle_ids.append(vehicle.id)

            except Exception as e:
                logger.error(f"Fehler beim Speichern eines Fahrzeugs aus DataFrame: {str(e)}")

        logger.info(f"{len(vehicle_ids)} Fahrzeuge aus DataFrame gespeichert")
        return vehicle_ids

    def get_vehicle(self, vehicle_id):
        """
        Holt ein Fahrzeug aus der Datenbank.

        Args:
            vehicle_id (int): ID des Fahrzeugs

        Returns:
            Vehicle: Das Fahrzeugobjekt oder None, wenn nicht gefunden
        """
        session = self.Session()
        try:
            return session.query(Vehicle).filter_by(id=vehicle_id).first()
        finally:
            session.close()

    def get_all_vehicles(self, limit=None, with_analysis=False):
        """
        Holt alle Fahrzeuge aus der Datenbank.

        Args:
            limit (int, optional): Maximale Anzahl der zurückgegebenen Fahrzeuge
            with_analysis (bool): Ob Analyseresultate geladen werden sollen

        Returns:
            list: Liste von Fahrzeugobjekten
        """
        session = self.Session()
        try:
            query = session.query(Vehicle)

            if with_analysis:
                # Lade auch das letzte Analyseergebnis für jedes Fahrzeug
                query = query.outerjoin(AnalysisResult)

            if limit:
                query = query.limit(limit)

            return query.all()
        finally:
            session.close()

    def save_market_data(self, vehicle_id, market_data):
        """
        Speichert Marktdaten für ein Fahrzeug.

        Args:
            vehicle_id (int): ID des Fahrzeugs
            market_data (dict): Marktdaten mit Statistiken und Listings

        Returns:
            MarketData: Das gespeicherte Marktdatenobjekt
        """
        session = self.Session()
        try:
            # Prüfe, ob das Fahrzeug existiert
            vehicle = session.query(Vehicle).filter_by(id=vehicle_id).first()
            if not vehicle:
                raise ValueError(f"Fahrzeug mit ID {vehicle_id} nicht gefunden")

            # Erstelle MarketData-Objekt
            listings_data = market_data.pop('listings', [])

            # Wenn raw_data als Dictionary vorliegt, konvertiere zu JSON-String
            if 'raw_data' in market_data and isinstance(market_data['raw_data'], dict):
                market_data['raw_data'] = json.dumps(market_data['raw_data'])

            market_data_obj = MarketData(vehicle_id=vehicle_id, **market_data)
            session.add(market_data_obj)
            session.flush()  # Um die ID zu erhalten

            # Füge Listings hinzu
            for listing in listings_data:
                listing_obj = MarketListing(market_data_id=market_data_obj.id, **listing)
                session.add(listing_obj)

            session.commit()
            logger.info(f"Marktdaten für Fahrzeug {vehicle_id} gespeichert (Listings: {len(listings_data)})")
            return market_data_obj

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Fehler beim Speichern der Marktdaten: {str(e)}")
            raise

        finally:
            session.close()

    def save_market_data_from_dataframe(self, df, vehicle_id_column='vehicle_id'):
        """
        Speichert Marktdaten aus einem DataFrame.

        Args:
            df (pd.DataFrame): DataFrame mit Marktdaten
            vehicle_id_column (str): Name der Spalte mit Fahrzeug-IDs

        Returns:
            int: Anzahl der gespeicherten Marktdatensätze
        """
        if df is None or df.empty:
            logger.warning("Leerer DataFrame, keine Marktdaten zu speichern")
            return 0

        count = 0

        # Gruppiere nach Fahrzeug-ID
        grouped = df.groupby(vehicle_id_column)

        for vehicle_id, group in grouped:
            try:
                # Erstelle Marktdaten für dieses Fahrzeug
                listings = []

                for _, row in group.iterrows():
                    listing = {
                        'titel': row.get('title', ''),
                        'preis': row.get('marktpreis', 0),
                        'baujahr': row.get('baujahr', None),
                        'kilometerstand': row.get('kilometerstand', None),
                        'kraftstoff': row.get('kraftstoff', None),
                        'leistung_kw': row.get('leistung_kw', None),
                        'url': row.get('url', None),
                        'haendler': row.get('haendler', False),
                        'standort': row.get('standort', None)
                    }
                    listings.append(listing)

                # Berechne Statistiken
                prices = group['marktpreis'].dropna().tolist()

                market_data = {
                    'quelle': 'mobile.de',
                    'timestamp': datetime.utcnow(),
                    'anzahl_angebote': len(group),
                    'marktpreis_min': min(prices) if prices else None,
                    'marktpreis_max': max(prices) if prices else None,
                    'marktpreis_mean': np.mean(prices) if prices else None,
                    'marktpreis_median': np.median(prices) if prices else None,
                    'raw_data': None,
                    'listings': listings
                }

                self.save_market_data(vehicle_id, market_data)
                count += 1

            except Exception as e:
                logger.error(f"Fehler beim Speichern von Marktdaten für Fahrzeug {vehicle_id}: {str(e)}")

        logger.info(f"{count} Marktdatensätze aus DataFrame gespeichert")
        return count

    def get_market_data(self, vehicle_id, days=30):
        """
        Holt Marktdaten für ein Fahrzeug aus den letzten X Tagen.

        Args:
            vehicle_id (int): ID des Fahrzeugs
            days (int): Maximales Alter der Daten in Tagen

        Returns:
            MarketData: Das neueste Marktdatenobjekt oder None
        """
        session = self.Session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            return session.query(MarketData) \
                .filter(MarketData.vehicle_id == vehicle_id) \
                .filter(MarketData.timestamp >= cutoff_date) \
                .order_by(desc(MarketData.timestamp)) \
                .first()
        finally:
            session.close()

    def save_analysis_result(self, vehicle_id, analysis_data):
        """
        Speichert ein Analyseergebnis für ein Fahrzeug.

        Args:
            vehicle_id (int): ID des Fahrzeugs
            analysis_data (dict): Analysedaten

        Returns:
            AnalysisResult: Das gespeicherte Analyseobjekt
        """
        session = self.Session()
        try:
            # Prüfe, ob das Fahrzeug existiert
            vehicle = session.query(Vehicle).filter_by(id=vehicle_id).first()
            if not vehicle:
                raise ValueError(f"Fahrzeug mit ID {vehicle_id} nicht gefunden")

            # Erstelle Analyseobjekt
            analysis = AnalysisResult(vehicle_id=vehicle_id, **analysis_data)
            session.add(analysis)
            session.commit()

            logger.info(f"Analyseergebnis für Fahrzeug {vehicle_id} gespeichert")
            return analysis

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Fehler beim Speichern des Analyseergebnisses: {str(e)}")
            raise

        finally:
            session.close()

    def save_analysis_from_dataframe(self, df, vehicle_id_column='id'):
        """
        Speichert Analyseergebnisse aus einem DataFrame.

        Args:
            df (pd.DataFrame): DataFrame mit Analyseergebnissen
            vehicle_id_column (str): Name der Spalte mit Fahrzeug-IDs

        Returns:
            int: Anzahl der gespeicherten Analyseergebnisse
        """
        if df is None or df.empty:
            logger.warning("Leerer DataFrame, keine Analyseergebnisse zu speichern")
            return 0

        count = 0

        for _, row in df.iterrows():
            try:
                vehicle_id = row[vehicle_id_column]

                # Erstelle Analysedaten
                analysis_data = {
                    'timestamp': datetime.utcnow(),
                    'min_profit_margin': row.get('min_profit_margin', 15.0),
                    'min_profit_amount': row.get('min_profit_amount', 2000.0),
                    'renovierungskosten': row.get('renovierungskosten', 0.0),
                    'steuern_gebuehren': row.get('steuern_gebuehren', 0.0),
                    'gesamtkosten': row.get('gesamtkosten', 0.0),
                    'nettogewinn': row.get('nettogewinn', 0.0),
                    'gewinnmarge_prozent': row.get('gewinnmarge_prozent', 0.0),
                    'roi': row.get('roi', 0.0),
                    'profitabilitaet': row.get('profitabilitaet', 'Unbekannt'),
                    'ai_analysis': row.get('ai_analysis', None)
                }

                # Bereinige NaN-Werte
                for key, value in analysis_data.items():
                    if isinstance(value, float) and np.isnan(value):
                        analysis_data[key] = None

                self.save_analysis_result(vehicle_id, analysis_data)
                count += 1

            except Exception as e:
                logger.error(f"Fehler beim Speichern des Analyseergebnisses: {str(e)}")

        logger.info(f"{count} Analyseergebnisse aus DataFrame gespeichert")
        return count

    def get_analysis_results(self, vehicle_id=None, newest_only=True):
        """
        Holt Analyseergebnisse für ein Fahrzeug oder alle Fahrzeuge.

        Args:
            vehicle_id (int, optional): ID des Fahrzeugs oder None für alle
            newest_only (bool): Nur das neueste Ergebnis pro Fahrzeug zurückgeben

        Returns:
            list: Liste von AnalysisResult-Objekten
        """
        session = self.Session()
        try:
            query = session.query(AnalysisResult)

            if vehicle_id:
                query = query.filter_by(vehicle_id=vehicle_id)

            if newest_only:
                if vehicle_id:
                    # Nur das neueste für ein bestimmtes Fahrzeug
                    query = query.order_by(desc(AnalysisResult.timestamp)).limit(1)
                else:
                    # Für alle Fahrzeuge: Subquery mit Group By
                    subquery = session.query(
                        AnalysisResult.vehicle_id,
                        func.max(AnalysisResult.timestamp).label('max_timestamp')
                    ).group_by(AnalysisResult.vehicle_id).subquery()

                    query = session.query(AnalysisResult).join(
                        subquery,
                        (AnalysisResult.vehicle_id == subquery.c.vehicle_id) &
                        (AnalysisResult.timestamp == subquery.c.max_timestamp)
                    )

            return query.all()
        finally:
            session.close()

    def get_summary_statistics(self):
        """
        Holt zusammenfassende Statistiken aus der Datenbank.

        Returns:
            dict: Statistiken über die gespeicherten Daten
        """
        session = self.Session()
        try:
            stats = {
                'total_vehicles': session.query(func.count(Vehicle.id)).scalar() or 0,
                'vehicles_with_market_data': session.query(func.count(func.distinct(MarketData.vehicle_id))).scalar() or 0,
                'vehicles_with_analysis': session.query(func.count(func.distinct(AnalysisResult.vehicle_id))).scalar() or 0,
                'total_market_listings': session.query(func.count(MarketListing.id)).scalar() or 0,
                'profitable_vehicles': session.query(func.count(AnalysisResult.id))
                                       .filter(AnalysisResult.profitabilitaet.in_(['Profitabel', 'Sehr profitabel']))
                                       .scalar() or 0,
                'avg_profit_margin': session.query(func.avg(AnalysisResult.gewinnmarge_prozent)).scalar() or 0,
                'latest_update': session.query(func.max(MarketData.timestamp)).scalar()
            }

            return stats
        finally:
            session.close()

    def export_to_dataframe(self, table_name, query_filter=None):
        """
        Exportiert Daten aus einer Tabelle in einen DataFrame.

        Args:
            table_name (str): Name der Tabelle ('vehicles', 'market_data', 'analysis_results')
            query_filter (callable, optional): Filterfunktion für die Abfrage

        Returns:
            pd.DataFrame: DataFrame mit den Daten
        """
        session = self.Session()
        try:
            if table_name == 'vehicles':
                query = session.query(Vehicle)
            elif table_name == 'market_data':
                query = session.query(MarketData)
            elif table_name == 'market_listings':
                query = session.query(MarketListing)
            elif table_name == 'analysis_results':
                query = session.query(AnalysisResult)
            else:
                raise ValueError(f"Unbekannte Tabelle: {table_name}")

            if query_filter:
                query = query_filter(query)

            results = query.all()

            # Konvertiere zu DataFrame
            data = [item.to_dict() for item in results]
            return pd.DataFrame(data)

        finally:
            session.close()

    def close(self):
        """Schließt alle Verbindungen zur Datenbank"""
        self.Session.remove()
        self.engine.dispose()
        logger.info("Datenbankverbindungen geschlossen")


# auto_auction_analyzer/database/migration.py
from alembic.config import Config
from alembic import command
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DBMigrationManager:
    """Verwaltet Datenbankmigrationen mit Alembic"""

    def __init__(self, db_path=None, migrations_path=None):
        """
        Initialisiert den Migrationsmanager.

        Args:
            db_path (str, optional): Pfad zur Datenbankdatei oder Verbindungsstring
            migrations_path (str, optional): Pfad zum Verzeichnis mit Migrationsskripten
        """
        self.db_path = db_path or os.environ.get('DB_CONNECTION', 'sqlite:///auto_auction.db')
        self.migrations_path = migrations_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'migrations'
        )

        # Stelle sicher, dass das Migrationsverzeichnis existiert
        os.makedirs(self.migrations_path, exist_ok=True)

        # Alembic-Konfiguration
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", self.migrations_path)
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.db_path)

    def init_migrations(self):
        """Initialisiert das Migrationsverzeichnis"""
        try:
            command.init(self.alembic_cfg, self.migrations_path)
            logger.info(f"Migrationsverzeichnis initialisiert: {self.migrations_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Migrationsverzeichnisses: {str(e)}")
            return False

    def create_migration(self, message):
        """
        Erstellt eine neue Migration.

        Args:
            message (str): Beschreibung der Migration

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            logger.info(f"Migration erstellt: {message}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Migration: {str(e)}")
            return False

    def upgrade_db(self, revision='head'):
        """
        Führt Migrationen bis zur angegebenen Revision durch.

        Args:
            revision (str): Zielrevision ('head' für die neueste)

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"Datenbank auf Revision {revision} aktualisiert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Datenbank: {str(e)}")
            return False

    def downgrade_db(self, revision):
        """
        Setzt Migrationen bis zur angegebenen Revision zurück.

        Args:
            revision (str): Zielrevision

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"Datenbank auf Revision {revision} zurückgesetzt")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Zurücksetzen der Datenbank: {str(e)}")
            return False

    def show_migrations(self):
        """
        Zeigt alle verfügbaren Migrationen an.

        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            command.history(self.alembic_cfg)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Anzeigen der Migrationen: {str(e)}")
            return False

    def get_current_revision(self):
        """
        Gibt die aktuelle Revision der Datenbank zurück.

        Returns:
            str: Aktuelle Revision oder None bei Fehler
        """
        try:
            from alembic.runtime.migration import MigrationContext
            from sqlalchemy import create_engine

            engine = create_engine(self.db_path)
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"Fehler beim Ermitteln der aktuellen Revision: {str(e)}")
            return None


# Beispielverwendung
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Datenbank initialisieren
    db_manager = DatabaseManager()
    db_manager.init_db()

    # Beispieldaten einfügen
    vehicle_data = {
        'marke': 'BMW',
        'modell': 'X5',
        'baujahr': 2018,
        'kilometerstand': 75000,
        'auktionspreis': 35000,
        'leistung': 265,
        'kraftstoff': 'Diesel'
    }

    vehicle = db_manager.save_vehicle(vehicle_data)
    print(f"Fahrzeug gespeichert mit ID: {vehicle.id}")

    # Marktdaten hinzufügen
    market_data = {
        'quelle': 'mobile.de',
        'anzahl_angebote': 15,
        'marktpreis_min': 30000,
        'marktpreis_max': 45000,
        'marktpreis_mean': 38000,
        'marktpreis_median': 37500,
        'listings': [
            {
                'titel': 'BMW X5 30d xDrive',
                'preis': 38500,
                'baujahr': 2018,
                'kilometerstand': 68000,
                'kraftstoff': 'Diesel',
                'leistung_kw': 195,
                'url': 'https://mobile.de/example/1'
            },
            {
                'titel': 'BMW X5 30d M-Paket',
                'preis': 42000,
                'baujahr': 2018,
                'kilometerstand': 55000,
                'kraftstoff': 'Diesel',
                'leistung_kw': 195,
                'url': 'https://mobile.de/example/2'
            }
        ]
    }

    db_manager.save_market_data(vehicle.id, market_data)

    # Analyseergebnis hinzufügen
    analysis_data = {
        'min_profit_margin': 15.0,
        'min_profit_amount': 2000.0,
        'renovierungskosten': 1500.0,
        'steuern_gebuehren': 1050.0,
        'gesamtkosten': 37550.0,
        'nettogewinn': 5450.0,
        'gewinnmarge_prozent': 15.57,
        'roi': 14.51,
        'profitabilitaet': 'Profitabel',
        'ai_analysis': 'Dieses Fahrzeug bietet eine solide Gewinnmarge von 15.57%. Der BMW X5 ist in gutem Zustand und hat einen starken Markt mit 15 vergleichbaren Angeboten zu höheren Preisen.'
    }

    db_manager.save_analysis_result(vehicle.id, analysis_data)

    # Statistiken anzeigen
    stats = db_manager.get_summary_statistics()
    print("Datenbankstatistiken:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Datenbankverbindung schließen
    db_manager.close()