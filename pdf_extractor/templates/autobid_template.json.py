{
    "name": "autobid_template",
    "description": "Template für Auktionskataloge",
    "vehicle_pattern": "^\\s*(\\d+)\\s+([A-Za-z\\-]+)\\s+(.+?)\\s+(\\d+)\\s+(\\d+)\\s+(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\\s+(\\d{2})\\s+(\\d+)",
    "fields": {
        "nummer": {"group_index": 1},
        "marke": {"group_index": 2, "transform": {"type": "normalize_marke"}},
        "modell": {"group_index": 3},
        "leistung": {"group_index": 4},
        "leistung_kw": {"group_index": 5},
        "baujahr": {"group_index": 7, "transform": {"type": "extract_year"}},
        "kilometerstand": {"group_index": 8, "transform": {"type": "clean_number"}}
    }
}