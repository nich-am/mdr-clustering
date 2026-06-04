"""
core/preprocess.py
------------------
Text cleaning and feature extraction (location + damage type)
for NRC title strings.
"""

import re
from collections import Counter

# ── Abbreviation expansion ─────────────────────────────────────────────────
ABBREV_MAP = {
    r"\bLH\b": "LEFT HAND", r"\bRH\b": "RIGHT HAND",
    r"\bFWD\b": "FORWARD",  r"\bAFT\b": "AFT",
    r"\bI/B\b": "INBOARD",  r"\bO/B\b": "OUTBOARD",
    r"\bINBD\b": "INBOARD", r"\bOUTBD\b": "OUTBOARD",
    r"\bNLG\b": "NOSE LANDING GEAR",
    r"\bMLG\b": "MAIN LANDING GEAR",
    r"\bENG\b": "ENGINE",   r"\bWNG\b": "WING",
    r"\bFUSEL\b": "FUSELAGE",
    r"\bCOMPT\b": "COMPARTMENT",
    r"\bFTG\b": "FITTING",  r"\bASY\b": "ASSEMBLY",
    r"\bINSTL\b": "INSTALLATION",
    r"\bPNL\b": "PANEL",    r"\bBRKT\b": "BRACKET",
    r"\bSTR\b": "STRINGER", r"\bSTA\b": "STATION",
    r"\bT/R\b": "THRUST REVERSER",
    r"\bHYD\b": "HYDRAULIC",
    r"\bELEC\b": "ELECTRICAL",
    r"\bLAV\b": "LAVATORY", r"\bCKPT\b": "COCKPIT",
    r"\bPAX\b": "PASSENGER",
    r"\b@\b": "AT",
}

STRIP_PATTERNS = [
    r"\bPK-[A-Z]{2,3}\b",
    r"\b\d+[A-Z]{1,3}\b",
    r"\b[A-Z]{1,2}\d+\b",
    r"\bNO\.?\s*\d+\b",
    r"\b#\s*\d+\b",
    r"\b\d+\s*EA\b",
    r"\bPN:\S+",
    r"\b\d{6,}\b",
    r"\bPOS\s+[\d\-]+\b",
    r"\bPRELIM\b",
    r"[^\w\s]",
    r"\b\d+\b",
    r"\s{2,}",
]

# ── Damage type vocabulary ─────────────────────────────────────────────────
DAMAGE_VOCAB = {
    r"\bCRACK(ED|ING)?\b":                                   "cracked",
    r"\bCORROD(ED|ING|E|TION|SION)?\b|CORROTION":           "corroded",
    r"\bEROD(ED|ING|E|TION|SION)?\b|ERROTION|ERROTED":      "eroded",
    r"\bWRINKLE[DS]?\b":                                     "wrinkled",
    r"\bDISBOND(ED)?\b":                                     "disbonded",
    r"\bDENT(ED|S)?\b":                                      "dented",
    r"\bNICK(S|ED)?\b":                                      "nicked",
    r"\bSCRATCH(ED|ES)?\b":                                  "scratched",
    r"\bBURN\s*MARK\b":                                      "burn mark",
    r"\bPAINT\s*PEEL\s*OFF\b|\bPEEL\s*OFF\b|\bPPO\b":      "paint peel off",
    r"\bPAINT\s*DISCOLOU?R(ATION)?\b":                      "paint discolored",
    r"\bPUNCTURE[DS]?\b":                                    "punctured",
    r"\bTEAR\s*OFF\b|\bTORN\b|\bTEAR(ED)?\b":              "torn",
    r"\bOUT\s*OF\s*(LIMIT|TOLERANCE)\b":                    "out of tolerance",
    r"\bCHAF(ING)?\b":                                      "chafing",
    r"\bOVERPLAY\b":                                        "overplay",
    r"\bBROKEN?\b":                                         "broken",
    r"\bLOOSE\b":                                           "loose",
    r"\bMISSING\b":                                         "missing",
    r"\bBENT\b":                                            "bent",
    r"\bDAMAGE[DS]?\b":                                     "damaged",
    r"\bBAD\s*CONDITION\b":                                 "deteriorated",
    r"\bDIRTY\b|\bDUSTY\b|\bNEED\s*CLEAN\b":              "dirty",
    r"\bTAKEN\s*FOR\s*(SERVICE|SUPPORT)\b|ROBBING\s*ITEM": "robbed",
    r"\bNOT\s*(PROPER|AVAIL|BRIGHT|COMPLETE)\b|\bN\s*/\s*A\b": "improper/N.A.",
    r"\bFRAY(ED|ING)?\b":                                   "fraying",
    r"\bFUNGUS\b":                                          "fungus",
}

# ── Location vocabulary ────────────────────────────────────────────────────
LOCATION_VOCAB = [
    (r"\bSCUFF\s*PLATE\b",                                    "door scuff plate"),
    (r"\bAFT\s*CARGO\s*DOOR\b",                              "aft cargo door"),
    (r"\bFWD\s*CARGO\s*DOOR\b",                              "fwd cargo door"),
    (r"\bCARGO\s*DOOR\b",                                     "cargo door"),
    (r"\bAFT\s*CARGO\s*FLOOR\b",                             "aft cargo floor"),
    (r"\bFWD\s*CARGO\s*FLOOR\b",                             "fwd cargo floor"),
    (r"\bCARGO\s*FLOOR\b",                                    "cargo floor"),
    (r"\bCARGO\s*COMP\b|\bCARGO\s*COMPARTMENT\b",            "cargo compartment"),
    (r"\bAFT\s*CARGO\b",                                      "aft cargo"),
    (r"\bFWD\s*CARGO\b",                                      "fwd cargo"),
    (r"\bBULK\s*CARGO\b",                                     "bulk cargo"),
    (r"\bBELLY\s*FAIRING\b",                                  "belly fairing"),
    (r"\bHORSTAB\s*LEADING\s*EDGE\b|\bL\s*/\s*E\s*HORSTAB\b","horstab leading edge"),
    (r"\bHORSTAB\b|\bTHS\b",                                  "horizontal stabilizer"),
    (r"\bELEVATOR\b",                                         "elevator"),
    (r"\bRUDDER\b",                                           "rudder"),
    (r"\bVERSTAB\b",                                          "vertical stabilizer"),
    (r"\bWINGLET\b",                                          "winglet"),
    (r"\bWING\s*TIP\b",                                       "wing tip"),
    (r"\bWING\s*SLAT\b|\bSLAT\b",                            "wing slat"),
    (r"\bWING\s*TANK\s*PANEL\b",                             "wing tank panel"),
    (r"\bFIX\s*FAIRING\b",                                    "fix fairing"),
    (r"\bFLAP\s*FAIRING\b",                                   "flap fairing"),
    (r"\bFLAP\b",                                             "flap"),
    (r"\bSPOILER\b",                                          "spoiler"),
    (r"\bAILERON\b",                                          "aileron"),
    (r"\bWING\b",                                             "wing"),
    (r"\bPYLON\s*PANEL\b",                                   "pylon panel"),
    (r"\bPYLON\b",                                            "pylon"),
    (r"\bMLG\s*DOOR\b",                                       "MLG door"),
    (r"\bMLG\b|\bMAIN\s*LANDING\s*GEAR\b",                   "main landing gear"),
    (r"\bINTAKE\s*COWL\b|\bINLET\s*COWL\b|\bNOSE\s*COWL\b", "engine inlet cowl"),
    (r"\bFAN\s*COWL\b",                                       "engine fan cowl"),
    (r"\bAPU\b",                                              "APU"),
    (r"\bENG(INE)?\b",                                        "engine"),
    (r"\bOUTERPANE\b|\bWINDOW\s*PANE\b",                    "cabin window pane"),
    (r"\bSEAT\s*BELT\b",                                      "seat belt"),
    (r"\bSEAT\s*TRACK\b",                                     "seat track"),
    (r"\bPAX\s*SEAT\b|\bPASSENGER\s*SEAT\b",                "passenger seat"),
    (r"\bPAX\s*DOOR\b",                                       "passenger door"),
    (r"\bDOORWAY\b",                                          "doorway"),
    (r"\bSTATIC\s*DISCHARGE\b",                               "static discharger"),
    (r"\bBONDING\s*(CABLE)?\b|\bLEAD\s*BOND\b",             "bonding cable"),
    (r"\bOHSC\b",                                             "overhead bin"),
    (r"\bSHROUD\b",                                           "lav shroud"),
    (r"\bANTI\s*SLIP\b|\bFLOOR\s*(COVER|PATH)?\b",          "floor"),
    (r"\bCEILING\b",                                          "ceiling panel"),
    (r"\bSIDEWALL\b",                                         "sidewall panel"),
    (r"\bPARTITION\b",                                        "cabin partition"),
    (r"\bLAVATORY\b|\bLAV\b",                                "lavatory"),
    (r"\bGALLEY\b",                                           "galley"),
    (r"\bCOCKPIT\b|\bCKPT\b",                                "cockpit"),
    (r"\bSEAT\b",                                             "seat"),
    (r"\bFUSELAGE\b",                                         "fuselage"),
    (r"\bFLOOR\s*BEAM\b",                                     "floor beam"),
    (r"\bSILL\s*BEAM\b|\bSHAPE\s*SILL\b",                   "sill beam"),
    (r"\bBULKHEAD\b",                                         "bulkhead"),
    (r"\bRADOME\b",                                           "nose radome"),
    (r"\bFASTENER\b|\bSCREW\b|\bBOLT\b|\bRIVET\b",          "fastener"),
    (r"\bSEAL\b",                                             "seal"),
    (r"\bCABIN\b",                                            "cabin"),
]


def clean_text(text: str) -> str:
    text = str(text).upper().strip()
    for pattern, replacement in ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text)
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def get_damage(text: str) -> str:
    u = str(text).upper()
    for pattern, label in DAMAGE_VOCAB.items():
        if re.search(pattern, u):
            return label
    return "other"


def get_location(text: str) -> str:
    u = str(text).upper()
    for pattern, label in LOCATION_VOCAB:
        if re.search(pattern, u):
            return label
    return "unclassified"


def build_cluster_label(titles: list) -> str:
    damages   = [get_damage(t)   for t in titles if get_damage(t)   != "other"]
    locations = [get_location(t) for t in titles if get_location(t) != "unclassified"]
    top_dmg = Counter(damages).most_common(1)[0][0]   if damages   else "damaged"
    top_loc = Counter(locations).most_common(1)[0][0] if locations else "component"
    return f"{top_loc} {top_dmg}"
