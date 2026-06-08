"""
core/preprocess.py
------------------
Text cleaning and feature extraction (location + sub-component + damage type)
for NRC title strings.

FIX v2:
  1. Sub-component vocabulary added — label now = location + sub-component + damage
     e.g. "wing lead bonding broken" instead of "wing broken"
  2. Structural noise patterns expanded — FR xx, STR xx, RIB xx-xx stripped
     so the same defect in different frame positions lands in the same cluster
"""

import re
from collections import Counter

# ── Abbreviation expansion ─────────────────────────────────────────────────
ABBREV_MAP = {
    r"\bLH\b": "LEFT",    r"\bRH\b": "RIGHT",
    r"\bFWD\b": "FORWARD", r"\bAFT\b": "AFT",
    r"\bI/B\b": "INBOARD", r"\bO/B\b": "OUTBOARD",
    r"\bINBD\b": "INBOARD", r"\bOUTBD\b": "OUTBOARD",
    r"\bNLG\b": "NOSE LANDING GEAR",
    r"\bMLG\b": "MAIN LANDING GEAR",
    r"\bENG\b": "ENGINE",  r"\bWNG\b": "WING",
    r"\bFUSEL\b": "FUSELAGE",
    r"\bCOMPT\b": "COMPARTMENT",
    r"\bFTG\b": "FITTING",  r"\bASY\b": "ASSEMBLY",
    r"\bINSTL\b": "INSTALLATION", r"\bINSTAL\b": "INSTALLATION",
    r"\bPNL\b": "PANEL",   r"\bBRKT\b": "BRACKET",
    r"\bSTRINGER\b": "STRINGER",
    r"\bT/R\b": "THRUST REVERSER",
    r"\bHYD\b": "HYDRAULIC",
    r"\bELEC\b": "ELECTRICAL",
    r"\bLAV\b": "LAVATORY", r"\bCKPT\b": "COCKPIT",
    r"\bPAX\b": "PASSENGER",
    r"\b@\b": "AT",
    r"\bBTW\b": "BETWEEN",
    r"\bW/\b": "WITH",
}

STRIP_PATTERNS = [
    # Aircraft tail
    r"\bPK-[A-Z]{2,3}\b",
    # Structural position codes — these are the main cause of over-splitting
    # FR xx, FR xx-xx  (Frame)
    r"\bFR\s*\d+[\-–]\d+\b",
    r"\bFR\s*\d+\b",
    # STR xx, STR xxL/R  (Stringer)
    r"\bSTR\s*\d+[LR]?\b",
    r"\bSTRINGER\s*\d+[LR]?\b",
    # RIB xx-xx
    r"\bRIB\s*\d+[\-–]\d+\b",
    r"\bRIB\s*\d+\b",
    # STA xx
    r"\bSTA\s*\d+\b",
    # Panel codes: 540AB, 640CB, 221FF, etc.
    r"\b[3-9]\d{2}[A-Z]{1,3}\b",
    # Zone/position: #1, #2, POS 3
    r"\b#\s*\d+\b",
    r"\bPOS\s+[\d\-]+\b",
    r"\bNO\.?\s*\d+\b",
    # Quantities: 1 EA, 2EA
    r"\b\d+\s*EA\b",
    r"\b\d+\s*QM\b",
    # Part numbers
    r"\bPN:\S+",
    r"\b[A-Z]{1,4}\d{4,}[:\-]\S*",   # e.g. CR3523-5-3:51563
    # Long order numbers
    r"\b\d{6,}\b",
    # PRELIM, REF
    r"\bPRELIM\b",
    r"\bREF\b",
    # Customer finding refs: CUST 19, FINDING CUST
    r"\bFINDING\s+CUST\b",
    r"\bCUST\s*\d+\b",
    # Carriage returns embedded in cells
    r"_x000D_",
    # Punctuation (keep spaces)
    r"[^\w\s]",
    # Standalone numbers after above cleaning
    r"\b\d+\b",
    # Extra whitespace
    r"\s{2,}",
]

# ── Damage type vocabulary ─────────────────────────────────────────────────
DAMAGE_VOCAB = {
    r"\bCRACK(ED|ING)?\b":                                   "cracked",
    r"\bCORROD(ED|ING|E|TION|SION)?\b|CORROTION|CORR\b":   "corroded",
    r"\bEROD(ED|ING|E|TION|SION)?\b|ERROTION|ERROTED":      "eroded",
    r"\bWRINKLE[DS]?\b|WRINGKLE":                           "wrinkled",
    r"\bDISBOND(ED)?\b|DISBONDING":                         "disbonded",
    r"\bDENT(ED|S)?\b":                                     "dented",
    r"\bNICK(S|ED)?\b":                                     "nicked",
    r"\bSCRATCH(ED|ES)?\b|SCRATCH":                         "scratched",
    r"\bBURN\s*MARK\b":                                     "burn mark",
    r"\bPAINT\s*PEEL\s*OFF\b|\bPEEL\s*OFF\b|\bPPO\b":     "paint peel off",
    r"\bPAINT\s*DISCOLOU?R(ATION)?\b":                     "paint discolored",
    r"\bPUNCTURE[DS]?\b|PUNCTURE":                         "punctured",
    r"\bTEAR\s*OFF\b|\bTORN\b|\bTEAR(ED)?\b":             "torn",
    r"\bOUT\s*OF\s*(LIMIT|TOLERANCE)\b":                   "out of tolerance",
    r"\bCHAF(ING)?\b":                                     "chafing",
    r"\bOVERPLAY\b":                                       "overplay",
    r"\bBROKEN?\b":                                        "broken",
    r"\bLOOSE\b":                                          "loose",
    r"\bMISSING\b":                                        "missing",
    r"\bBENT\b":                                           "bent",
    r"\bELONGAT(ED)?\b":                                   "elongated hole",
    r"\bWEAR\s*DAMAGE\b":                                  "worn/damaged",
    r"\bDAMAGE[DS]?\b":                                    "damaged",
    r"\bBAD\s*CONDITION\b":                                "deteriorated",
    r"\bDIRTY\b|\bDUSTY\b|\bNEED\s*CLEAN\b":             "dirty",
    r"\bTAKEN\s*FOR\s*(SERVICE|SUPPORT)\b|ROBBING":       "robbed",
    r"\bNOT\s*(PROPER|AVAIL|BRIGHT|COMPLETE)\b|\bN\s*/\s*A\b": "improper/N.A.",
    r"\bFRAY(ED|ING)?\b":                                  "fraying",
    r"\bFUNGUS\b":                                         "fungus",
    r"\bBLOWOUT\b":                                        "blowout",
    r"\bMULTIPLE\b":                                       "multiple damage",
}

# ── Sub-component vocabulary (NEW) ────────────────────────────────────────
# Sits between location and damage in the label
# Order matters: more specific patterns first
SUB_COMPONENT_VOCAB = [
    # Wing internals
    (r"\bLEAD\s*BOND(ING)?\b",               "lead bonding"),
    (r"\bBOND(ING)?\s*CABLE\b",              "bonding cable"),
    (r"\bBOND(ING)?\b",                      "bonding"),
    (r"\bPIPING\s*INSTL\b|\bPIPING\b",       "piping"),
    (r"\bFUEL\s*RECIRCULATION\b",            "fuel recirculation"),
    (r"\bWING\s*TANK\s*SCREW\b|\bSCREW\b",  "screw"),
    # Cargo internals
    (r"\bPROFIL?\s*CORNER\b",               "profile corner"),
    (r"\bPLATE\s*(FLOOR\s*)?SUPPORT\b",     "plate support"),
    (r"\bINTERNAL\s*SKIN\b",                "internal skin"),
    (r"\bMIDDLE\s*BEAM\b",                  "middle beam"),
    (r"\bVERTICAL\s*MEMBER\b",              "vertical member"),
    (r"\bSIDE\s*FLOOR\b",                   "side floor"),
    (r"\bCEIL(ING|LING)?\b",               "ceiling"),
    (r"\bFLOOR\s*BEAM\b",                   "floor beam"),
    (r"\bSILL\s*BEAM\b|\bSHAPE\s*SILL\b",  "sill beam"),
    (r"\bBULKHEAD\b",                        "bulkhead"),
    # Fuselage
    (r"\bEXT(ERIOR)?\s*SKIN\b",             "exterior skin"),
    (r"\bSKIN\s*PANEL\b|\bSKIN\b",          "skin"),
    # Control surfaces
    (r"\bLEADING\s*EDGE\b|\bL\s*/\s*E\b",  "leading edge"),
    (r"\bTRAILING\s*EDGE\b",                "trailing edge"),
    # Lavatory / cabin
    (r"\bSHROUD\b",                          "shroud"),
    (r"\bMIRROR\b",                          "mirror"),
    (r"\bSINK\b|\bWASH\s*BASIN\b",          "sink"),
    (r"\bFLUSH\b",                           "flush mechanism"),
    (r"\bSEAT\s*BELT\b",                     "seat belt"),
    (r"\bSEAT\s*TRACK\b",                    "seat track"),
    (r"\bARMREST\b",                         "armrest"),
    # Engine / pylon
    (r"\bFAN\s*BLADE\b",                     "fan blade"),
    (r"\bCOWL\s*LATCH\b",                    "cowl latch"),
    (r"\bHINGE\b",                           "hinge"),
    (r"\bSEAL\b",                            "seal"),
    (r"\bBRACKET\b|\bBRKT\b",               "bracket"),
    # Generic structural
    (r"\bFASTENER\b|\bSCREW\b|\bBOLT\b|\bRIVET\b|\bNUT\b", "fastener"),
    (r"\bCLEAT\b",                           "cleat"),
    (r"\bFITTING\b|\bFTG\b",                "fitting"),
    (r"\bGUSS?ET\b",                         "gusset"),
    (r"\bWEB\b",                             "web"),
    (r"\bFLANGE\b",                          "flange"),
    (r"\bDOME\s*NUT\b",                      "dome nut"),
    (r"\bPLATE\b",                            "plate"),
    (r"\bPANEL\b",                            "panel"),
]

# ── Location vocabulary ────────────────────────────────────────────────────
LOCATION_VOCAB = [
    (r"\bSCUFF\s*PLATE\b",                                     "door scuff plate"),
    (r"\bAFT\s*CARGO\s*DOOR\b",                               "aft cargo door"),
    (r"\bFWD\s*CARGO\s*DOOR\b|\bFORWARD\s*CARGO\s*DOOR\b",   "fwd cargo door"),
    (r"\bCARGO\s*DOOR\b",                                      "cargo door"),
    (r"\bAFT\s*CARGO\s*FLOOR\b",                              "aft cargo floor"),
    (r"\bFWD\s*CARGO\s*FLOOR\b|\bFORWARD\s*CARGO\s*FLOOR\b", "fwd cargo floor"),
    (r"\bCARGO\s*FLOOR\b",                                     "cargo floor"),
    (r"\bCARGO\s*COMP\b|\bCARGO\s*COMPARTMENT\b",             "cargo compartment"),
    (r"\bAFT\s*CARGO\b",                                       "aft cargo"),
    (r"\bFWD\s*CARGO\b|\bFORWARD\s*CARGO\b",                 "fwd cargo"),
    (r"\bBULK\s*CARGO\b",                                      "bulk cargo"),
    (r"\bBELLY\s*FAIRING\b",                                   "belly fairing"),
    (r"\bHORSTAB\s*LEADING\s*EDGE\b|\bL\s*/\s*E\s*HORSTAB\b","horstab leading edge"),
    (r"\bHORSTAB\b|\bTHS\b",                                   "horizontal stabilizer"),
    (r"\bELEVATOR\b",                                          "elevator"),
    (r"\bRUDDER\b",                                            "rudder"),
    (r"\bVERSTAB\b|\bVERTICAL\s*STAB",                       "vertical stabilizer"),
    (r"\bWINGLET\b",                                           "winglet"),
    (r"\bWING\s*TIP\b",                                        "wing tip"),
    (r"\bWING\s*SLAT\b|\bSLAT\b",                             "wing slat"),
    (r"\bWING\s*TANK\s*PANEL\b|\bTANK\s*PANEL\b",            "wing tank panel"),
    (r"\bFIX\s*FAIRING\b",                                     "fix fairing"),
    (r"\bFLAP\s*FAIRING\b",                                    "flap fairing"),
    (r"\bFLAP\b",                                              "flap"),
    (r"\bSPOILER\b",                                           "spoiler"),
    (r"\bAILERON\b",                                           "aileron"),
    (r"\bWING\b",                                              "wing"),
    (r"\bPYLON\s*PANEL\b",                                    "pylon panel"),
    (r"\bPYLON\b",                                             "pylon"),
    (r"\bMLG\s*DOOR\b",                                        "MLG door"),
    (r"\bMLG\b|\bMAIN\s*LANDING\s*GEAR\b",                    "main landing gear"),
    (r"\bINTAKE\s*COWL\b|\bINLET\s*COWL\b|\bNOSE\s*COWL\b", "engine inlet cowl"),
    (r"\bFAN\s*COWL\b",                                        "engine fan cowl"),
    (r"\bAPU\b",                                               "APU"),
    (r"\bENG(INE)?\b",                                         "engine"),
    (r"\bOUTERPANE\b|\bWINDOW\s*PANE\b",                     "cabin window pane"),
    (r"\bSEAT\s*BELT\b",                                       "seat belt"),
    (r"\bSEAT\s*TRACK\b",                                      "seat track"),
    (r"\bPAX\s*SEAT\b|\bPASSENGER\s*SEAT\b",                 "passenger seat"),
    (r"\bPAX\s*DOOR\b|\bPASSENGER\s*DOOR\b",                 "passenger door"),
    (r"\bDOORWAY\b",                                           "doorway"),
    (r"\bSTATIC\s*DISCHARGE\b",                                "static discharger"),
    (r"\bBONDING\s*(CABLE)?\b|\bLEAD\s*BOND\b",              "bonding cable"),
    (r"\bOHSC\b",                                              "overhead bin"),
    (r"\bSHROUD\b",                                            "lav shroud"),
    (r"\bANTI\s*SLIP\b|\bFLOOR\s*(COVER|PATH)?\b",           "floor"),
    (r"\bCEILING\b",                                           "ceiling panel"),
    (r"\bSIDEWALL\b",                                          "sidewall panel"),
    (r"\bPARTITION\b",                                         "cabin partition"),
    (r"\bLAVATORY\b|\bLAV\b",                                 "lavatory"),
    (r"\bGALLEY\b",                                            "galley"),
    (r"\bCOCKPIT\b|\bCKPT\b",                                 "cockpit"),
    (r"\bSEAT\b",                                              "seat"),
    (r"\bFUSELAGE\b",                                          "fuselage"),
    (r"\bFLOOR\s*BEAM\b",                                      "floor beam"),
    (r"\bSILL\s*BEAM\b|\bSHAPE\s*SILL\b",                    "sill beam"),
    (r"\bBULKHEAD\b",                                          "bulkhead"),
    (r"\bRADOME\b",                                            "nose radome"),
    (r"\bFASTENER\b|\bSCREW\b|\bBOLT\b|\bRIVET\b",           "fastener"),
    (r"\bSEAL\b",                                              "seal"),
    (r"\bCABIN\b",                                             "cabin"),
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


def get_sub_component(text: str) -> str:
    """Extract the most specific sub-component from the NRC title."""
    u = str(text).upper()
    for pattern, label in SUB_COMPONENT_VOCAB:
        if re.search(pattern, u):
            return label
    return ""


def build_cluster_label(titles: list) -> str:
    """
    Build label as: location [+ sub-component] + damage
    Sub-component is included when it adds specificity beyond location.
    """
    damages      = [get_damage(t)        for t in titles if get_damage(t)        != "other"]
    locations    = [get_location(t)      for t in titles if get_location(t)      != "unclassified"]
    sub_comps    = [get_sub_component(t) for t in titles if get_sub_component(t) != ""]

    top_dmg = Counter(damages).most_common(1)[0][0]      if damages   else "damaged"
    top_loc = Counter(locations).most_common(1)[0][0]    if locations else "component"
    top_sub = Counter(sub_comps).most_common(1)[0][0]    if sub_comps else ""

    # Only include sub-component if:
    # (a) it's present in >30% of titles (not just noise from one NRC)
    # (b) it's different from what the location already says
    if top_sub and sub_comps:
        prevalence = sub_comps.count(top_sub) / len(titles)
        # Avoid redundancy e.g. location="bonding cable", sub="bonding"
        if prevalence >= 0.3 and top_sub not in top_loc:
            return f"{top_loc} {top_sub} {top_dmg}"

    return f"{top_loc} {top_dmg}"
