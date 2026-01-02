"""Words people use for the schema's concepts. Evidence only: the Bayesian network decides.

Each entry maps a phrase to a *lexeme*. A lexeme can be ambiguous on purpose: "cycle time" is one
lexeme that the network resolves to ``cnc_data.cycle_time`` or ``press_data.cycle_time`` from context.
"""

from __future__ import annotations

METRIC_PHRASES: dict[str, tuple[str, ...]] = {
    "oee": ("oee", "o.e.e", "overall equipment effectiveness", "overall equipment efficiency",
            "equipment effectiveness", "overall effectiveness"),
    "availability": ("availability", "avail", "uptime", "up time", "up-time", "machine availability"),
    "performance": ("performance rate", "speed performance", "performance", "perf"),
    "quality": ("quality rate", "first pass yield", "fpy", "quality", "yield", "good part rate"),
    "scrap_rate": ("scrap rate", "scrap percentage", "scrap %", "scrap pct", "scrap", "scrappage", "scrapped"),
    "downtime_minutes": ("downtime minutes", "minutes of downtime", "downtime", "down time", "down-time",
                         "stoppage minutes", "stoppages", "minutes down", "time down"),
    "spindle_load": ("spindle load", "spindle loads", "spindle loading", "load on the spindle", "spindle %"),
    "spindle_speed": ("spindle speed", "spindle speeds", "spindle rpm", "rpm", "rpms", "rotational speed"),
    "feed_rate": ("feed rate", "feed rates", "feedrate", "feedrates", "feeds", "feed"),
    "tool_wear": ("tool wear", "tooling wear", "wear on the tools", "tool life", "tw"),
    "cycle_time": ("cycle time", "cycle times", "cycletime", "ct", "cycle"),
    "alarm_count": ("alarm count", "alarm counts", "number of alarms", "alarms", "alarm", "faults"),
    "tonnage": ("tonnage", "tonnages", "tonage", "press force", "tons", "force"),
    "stroke_rate": ("stroke rate", "stroke rates", "strokes per minute", "strokes/min", "spm", "strokes",
                    "stroke speed"),
    "die_temperature": ("die temperature", "die temperatures", "die temp", "die temps", "temperature", "temp",
                        "temps"),
    "reject_count": ("reject count", "reject counts", "number of rejects", "rejected parts", "rejects",
                     "rejections", "rejected", "reject"),
}

SUBGRAPH_PHRASES: dict[str, tuple[str, ...]] = {
    "cnc": ("cnc", "c.n.c", "machining", "machine tool", "machine tools", "spindle", "spindles", "milling",
            "mill", "mills", "lathe", "lathes", "turning", "machining center", "machining centre", "vmc", "cutting"),
    "press": ("press", "presses", "stamping", "stamp", "stamped", "die", "dies", "press shop", "pressroom",
              "press room", "stroke"),
    "kpi": ("kpi", "kpis", "k.p.i", "production kpis", "performance metrics", "key performance indicators",
            "production metrics"),
}

RANK_PHRASES: dict[str, tuple[str, ...]] = {
    "top": ("top", "highest", "most", "max", "maximum", "largest", "biggest", "greatest", "peak", "descending",
            "desc", "high to low", "highest to lowest", "heaviest", "longest", "hottest"),
    "bottom": ("bottom", "lowest", "least", "min", "minimum", "smallest", "fewest", "ascending", "asc",
               "low to high", "lowest to highest", "lightest", "shortest", "coolest"),
    # "fastest cycle time" is the lowest one, "fastest stroke rate" the highest: speed words follow polarity
    "best": ("best", "strongest", "top performing", "top-performing", "best performing", "best-performing",
             "fastest", "quickest", "lead on", "leading on", "leads on", "leaders", "leading", "winning"),
    "worst": ("worst", "poorest", "weakest", "worst performing", "worst-performing", "underperforming",
              "slowest"),
    "rank": ("rank", "ranked", "ranking", "rankings", "sort", "sorted", "order by", "ordered by", "leaderboard"),
}

LEVEL_WORDS: dict[str, tuple[str, ...]] = {
    "group": ("group", "groups"),
    "region": ("region", "regions", "regional"),
    "division": ("division", "divisions", "business unit", "business units", "bu", "bus"),
    "plant": ("plant", "plants", "site", "sites", "factory", "factories", "facility", "facilities",
              "location", "locations"),
    "subarea": ("subarea", "subareas", "sub-area", "sub-areas", "sub area", "sub areas", "department",
                "departments", "dept", "depts", "section", "sections", "shop", "shops"),
    "area": ("area", "areas", "zone", "zones", "cell", "cells"),
    "line": ("line", "lines", "production line", "production lines", "ln"),
}
PLURAL_LEVEL_WORDS = {
    "groups": "group", "regions": "region", "divisions": "division", "business units": "division",
    "plants": "plant", "sites": "plant", "factories": "plant", "facilities": "plant", "locations": "plant",
    "subareas": "subarea", "sub-areas": "subarea", "sub areas": "subarea", "departments": "subarea",
    "depts": "subarea", "sections": "subarea", "shops": "subarea",
    "areas": "area", "zones": "area", "cells": "area", "lines": "line", "production lines": "line",
}

RAW_PHRASES = ("records", "record", "recrods", "reocrds", "readings", "reading", "rows", "row", "entries", "entry",
               "logs", "log",
               "raw", "data points", "datapoints", "each shift", "every shift", "per shift", "by shift",
               "shift by shift", "each day", "every day", "per day", "by day", "day by day", "daily", "history",
               "timeline", "all data", "all the data", "full data", "every record", "line items")

# Names people use for hierarchy nodes above plant level (plant names come from the hierarchy itself).
NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "MOB": ("acme mobility", "mobility group", "mobility"),
    "IND": ("acme industrial", "industrial group", "industrial"),
    "AMER": ("the americas", "americas", "america", "north america", "amer"),
    "EUR": ("europe", "european", "eur", "emea"),
    "APAC": ("asia pacific", "asia-pacific", "asia/pacific", "apac", "asia"),
    "BODY": ("body systems", "body system", "body"),
    "PWT": ("powertrain", "power train", "pwt"),
    "CHS": ("chassis",),
    "INT": ("interiors", "interior"),
    "ELX": ("electronics", "elx"),
    "SEAT": ("seating", "seats"),
}
SUBAREA_ALIASES: dict[str, tuple[str, ...]] = {
    "PS": ("press shop", "press shops", "pressshop", "press room", "pressroom", "stamping shop"),
    "MC": ("machining subarea", "machining sub-area", "machining sub area", "machining department",
           "machining dept", "machining shop", "machine shop", "machining section", "machining hall"),
    "AS": ("assembly subarea", "assembly sub-area", "assembly department", "assembly dept", "assembly hall",
           "assembly shop", "assembly section", "assembly"),
}

DOMAIN_WORDS = ("oee", "kpi", "kpis", "metric", "metrics", "production", "line", "lines", "plant", "plants",
                "shift", "press", "cnc", "machine", "machines", "scrap", "downtime", "output")

STOP_BEFORE_PLANT = {
    "the", "a", "an", "our", "your", "my", "this", "that", "each", "every", "which", "what", "one", "any",
    "per", "by", "best", "worst", "top", "bottom", "whole", "entire", "same", "other", "main", "all",
    "highest", "lowest", "most", "least", "whose", "for", "at", "in", "of", "to", "biggest",
    "largest", "smallest", "individual", "specific", "given", "single", "production", "manufacturing",
    "stamping", "assembly", "machining", "press", "cnc", "worst-performing", "best-performing",
    "about", "with", "like", "from", "into", "near", "new", "old", "big", "local", "their", "his", "her", "its",
    "some", "another", "own", "typical", "average", "modern", "smart", "automotive",
}
