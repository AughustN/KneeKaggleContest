"""Negation-aware, multilingual keyword extractor for knee MRI reports. v2.

Key v2 changes (driven by gold-58 error analysis):
- sentence-level scoping: polarity decided within the sentence only, so a
  "no X" never bleeds into the next sentence
- anchor-based rules: anatomical anchor (e.g. ACL, medial meniscus) found
  first; injury keywords are matched in the SAME sentence as the anchor --
  laterality context INCLUDES the anchor span (v1 excluded it, killing recall)
- Cyrillic is transliterated (not deleted) so Bulgarian/Russian reports work
- clinical-history/indication sections are stripped before matching
- "mild/small effusion" counts as positive (matches gold convention)
- subchondral/degenerative/insufficiency-context edema does NOT fire Contusion

Scores: 1.0 asserted, 0.5 hedged, 0.0 negated/normal, -1 no mention.
"""

import re
import unicodedata
from typing import Dict, List, Optional

from .constants import LABELS

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

_CYR_TO_LAT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sht", "ъ": "a",
    "ы": "i", "ь": "", "э": "e", "ю": "iu", "я": "ia",
})

# Greek -> Latin (beta-code-ish, lowercased)
_GRK_TO_LAT = str.maketrans({
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i",
    "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y",
    "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o",
    "ά": "a", "έ": "e", "ή": "i", "ί": "i", "ό": "o", "ύ": "y",
    "ώ": "o", "ϊ": "i", "ϋ": "y", "ΐ": "i", "ΰ": "y",
})

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = text.translate(_CYR_TO_LAT)          # Cyrillic -> latin
    t = t.translate(_GRK_TO_LAT)             # Greek -> latin
    t = unicodedata.normalize("NFKD", t)
    t = t.encode("ascii", "ignore").decode()  # strip accents/Turkish tails
    t = t.lower()
    t = _WS.sub(" ", t)
    return t.strip()


# sections that mention findings speculatively (clinical request), not results
_SECTION_STRIP = re.compile(
    r"(clinical history|clinical information|klinische inlichtingen|"
    r"indication|indicacion|indicatie|indikation|indication clinique|"
    r"anamnese|anamnesis|anamnez|vorgeschichte|diagnostische vraagstell\w*|"
    r"diagnostic question|verzoek|fragestellung|anfrage)\s*:.*?(?=findings|"
    r"bevindingen|hallazgos|resultados|beschreibung|technique|tecnica|"
    r"techniek|examen|exam type|impresion|impression|befund|proc\w*|"
    r"procedure|bildgebung|\Z)",
    re.DOTALL,
)

_SENT_SPLIT = re.compile(r"[.;>\n*|]| (?=conclusion:|impression:|impresion:)")


def sentences(norm: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(norm) if s.strip()]


# ---------------------------------------------------------------------------
# polarity markers (matched within a sentence)
# ---------------------------------------------------------------------------

NEGATORS = [
    "no ", "not ", "without", "absence", "absent", "free of", "no evidence",
    "negative for", "rule out", "r/o", "ruled out", "excluded", "excluye",
    "nor ", "none", "none.", "nincs",
    "no hay", "sin ", "ausencia de", "no se", "no existe", "descarta",
    "sin signos de", "sin evidencia", "sin criterios", "sin alteracion",
    "negativo",
    "keine ", "kein ", "nicht", "ohne", "frei von", "kein hinweis",
    "unauffa", "ausschluss",
    "geen ", "zonder",
    "pas de", "aucun", "aucune", "sans ", "absence de",
    "yok", "yoktur", "izlenmedi", "izlenmeyen", "gorulmedi", "gorulmeyen",
    "gorulmez", "mevcut degil", "saptanmadi", "saptanmayan", "eslik etmiyor",
    "eslik etmedigi", "saglam", "korunmu", "korunmus",
    "non si", "assenza", "senza", "nao ha", "sem ",
    "nema ", "nije ", "ne postoji", "odsutan", "alichie", "bez ",
    # greek (transliterated)
    "den paratir", "den simeion", "den paratiro", "xoris ", "chos ",
    "aparagkaste", "den apokatalyp",
]

HEDGERS = [
    "suspected", "suspect", "suspicious", "possible", "possibility",
    "cannot be excluded", "cannot exclude", "not excluded", "cannot be ruled",
    "cannot totally excluded", "likely", "may be", "questionable",
    "probable", "versus", "differential", "verdacht", "verdachtig",
    "kann nicht ausgeschlossen", "nicht sicher", "moeglicherweise",
    "mogelijk", "mogelijks", "vermoedelijk",
    "muhtemel", "olasi", "dusunul", "supheli", "olasilikla",
    "moguce", "moze", "moze odgovarati", "vjerojatno", "pretpostavka",
    "in favor of", "favor of", "suggest", "suggestive", "suspicion",
    "compatible with", "in keeping with", "in keeping", "favor ",
    "unlikely", "diferencial", "impression",
]

NORMALITY = [
    "intact", "normal", "preserved", "unremarkable", "within normal", "wnl",
    "regular", "regelrecht", "regelrechte", "allseits", "physiologisch",
    "unauffa", "normaal", "normale", "normaldir", "normalde", "intacte",
    "conservada", "sin alteraciones", "conserve", "mantenido",
    "dans les limites", "binnen normale", "grenzen", "grenzen der norm",
    "saglam", "korunmus", "odrzanog kontinuiteta", "primjeren",
    "patolojik",  # careful: "patolojik yok" = negated already
]


def _compile_words(words: List[str]) -> re.Pattern:
    parts = []
    for w in sorted(words, key=len, reverse=True):
        if not w:
            continue
        if w.endswith(" "):
            # trailing space = word must END here; add lookbehind so 'interno '
            # does not match negator 'no '
            parts.append(r"(?<![a-z])" + re.escape(w))
        else:
            parts.append(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])")
    return re.compile("|".join(parts))


NEG_RE = _compile_words(NEGATORS)
HEDGE_RE = _compile_words(HEDGERS)
NORM_RE = _compile_words(NORMALITY)


def sentence_polarity(sent: str) -> float:
    """1.0 positive / 0.5 hedged / 0.0 negated-or-normal."""
    has_neg = bool(NEG_RE.search(sent))
    has_norm = bool(NORM_RE.search(sent))
    has_hedge = bool(HEDGE_RE.search(sent))
    if has_neg or has_norm:
        return 0.0
    if has_hedge:
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# label rules: anchor -> same-sentence injury evidence
# ---------------------------------------------------------------------------

_MED = (r"(?<![a-z])(?:medial|mediyal|mediaal|mediale|medialde|medijaln\w*|"
        r"medialni\w*|medialen\b|medialn\w*|innen|inneren|interior|interno|"
        r"interna|ic|ici|eso)(?![a-z])|innenmeniskus")
_LAT = (r"(?<![a-z])(?:lateral|lateraal|laterale|lateraly|lateralde|"
        r"externo|externa|exterior|dis|disir|fibular|peroneal|exo|"
        r"lateralni\w*|lateraln\w*|laterani\w*|lateralen\b)(?![a-z])|"
        r"aussen|auszen|auseren|auszeren|auenmeniskus")
_PF = (r"(?<![a-z])(?:patell|patello|patelofemoral|rotuli|rotula|trochle|"
       r"troclear|retropatell|kniescheibe|femoropatell|diz kapa|epigonatid|"
       r"patele|patelarn|fasete)")

_MED_RE = re.compile(_MED)
_LAT_RE = re.compile(_LAT)
_PF_RE = re.compile(_PF)
MENISCUS_ANCHOR = re.compile(
    r"menisc|menisko|menisk|inisk|minisk"  # greek inisk/minisk after iota-loss
)


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts))


# --- ACL ---
ACL_ANCHOR = _rx(
    r"\bacl\b", r"a\.c\.l", r"anterior cruciate", r"vordere\w* kreuzband",
    r"\bvkb\b", r"on capraz bag", r"one capraz bag", r"anterior capraz",
    r"cruzado anterior", r"\blca\b", r"croise anterieur", r"fore korsband",
    r"prednji krizni ligament",
    r"prosthioy chiastoy syndes", r"prosthios chiastoy",  # greek anterior cruciate
)
ACL_INJURY = _rx(
    r"tear", r"ruptur", r"rupture", r"rotura", r"disrupt", r"avuls",
    r"laxit", r"insufficien", r"sprain", r"injur", r"lesion", r"midsubstance",
    r"reiss", r"riss", r"scheur", r"breech", r"defekt", r"yrtk", r"delik",
    r"kirilma", r"devams", r"bütünl", r"butunluk kayb",
    r"rixi", r"rima\b", r"asynecheia",  # greek tear/discontinuity
)
PCL_ANCHOR = _rx(
    r"\bpcl\b", r"p\.c\.l", r"posterior cruciate", r"hintere\w* kreuzband",
    r"\bhkb\b", r"arka capraz", r"posterior capraz", r"cruzado posterior",
    r"\blcp\b", r"croise posterieur", r"straznji krizni",
    r"opisthi\w* chiast", r"opisthioy chiast",  # greek posterior cruciate
)

# --- MCL ---
MCL_ANCHOR = _rx(
    r"\bmcl\b", r"medial collateral", r"medial kollateral", r"medial collateraal",
    r"colateral medial", r"collateral medial", r"medial capsular",
    r"medial kollateral ligaman", r"medijalni kolateral", r"tibial collateral",
    r"inneren seitenband", r"inneres seitenband",
    r"eso plagioy syndes", r"eso plagios syndes",  # greek medial collateral
)
MCL_INJURY = _rx(
    r"tear", r"ruptur", r"rupture", r"rotura", r"sprain", r"injur", r"lesion",
    r"disrupt", r"laxit", r"thicken", r"verdik", r"odemat", r"odematoz",
    r"oedema", r"edema", r"swell", r"riss", r"scheur", r"periligament",
    r"grade i\b", r"grade 1", r"grade ii", r"grade 2", r"grade iii", r"grade 3",
    r"distraction", r"avuls", r"yrtk", r"kirilma", r"devams",
)
LCL_ANCHOR = _rx(
    r"\blcl\b", r"\bfcl\b", r"lateral collateral", r"fibular collateral",
    r"laterale collateraal", r"laterales kollateral", r"colateral lateral",
    r"popliteus", r"biceps femoris", r"posterolateral corner",
    r"auseren seitenband", r"auszeren seitenband", r"it band", r"iliotibial",
    r"exo plagioy syndes",  # greek lateral collateral
)

MCL_GRADE1 = re.compile(  # grade 1 / low-grade findings -> NOT labeled MCL injury
    r"grade i\b|grade 1|low-grade|low grade|hafif odematoz|"
    r"incelme|periligamentoz odem$"
)
MCL_HIGH = re.compile(  # grade 2+ / complete -> labeled positive
    r"grade ii|grade 2|grade iii|grade 3|complete|high-grade|high grade|"
    r"complet|total|ruptura completa|perdida (parcial )?de la continuidad"
)

# --- menisci ---
MEN_INJURY = _rx(
    r"tear", r"ruptur", r"rupture", r"rotura", r"riss", r"scheur", r"yrtk",
    r"breuk", r"disrupt", r"lesion", r"delik", r"kirilma", r"extrus",
    r"amputacion", r"amput", r"flap", r"bucket", r"kova sap", r"buckenhandle",
    r"displaced", r"horizontal", r"radial", r"vertical", r"oblique",
    r"complex", r"degenerative signal", r"grade ii", r"grade iii", r"grade 2",
    r"grade 3",
    r"rixi", r"rima\b", r"ekfylis",  # greek tear/rupture
    r"ris", r"ruptura",
)
MEN_DEGEN = _rx(  # degeneration without tear: NOT a positive by itself
    r"mucoid", r"myxoid", r"miksoid", r"degenerat", r"mukoid",
    r"ekfylistik", r"ekfylis",  # greek degenerative
)
MEN_NORMAL = _rx(r"normal", r"intact", r"conservada", r"preserved", r"saglam",
                 r"korunmus", r"afwijking")

# --- OA / cartilage ---
OA_TERMS = _rx(
    r"osteoarthrit", r"arthrose", r"artrose", r"artrosis", r"arthritis",
    r"gonarthr", r"artros", r"chondropath", r"condropat", r"chondropatia",
    r"chondrosis", r"chondrosis", r"chondromalac", r"hondromalac",
    r"hondromalacij",  # serbian/croatian
    r"cartilage loss", r"kraakbeenlijden", r"kraakbeenverlies",
    r"chondral defect", r"chondral ulcer", r"ulceras condral",
    r"ulcera condral", r"cartilage defect", r"cartilage thinning",
    r"cartilage irregular", r"denudation", r"denudacija",
    r"full thickness", r"full-thickness", r"espesor total",
    r"grade iii", r"grade iv", r"icrs grade iii", r"icrs grade iv",
    r"ostechondral", r"osteochondral",
    r"femorotibiaal kraakbeen", r"osteofit", r"osteophyt",
    r"osteoarthritida", r"arthritida",  # greek
    r"chondroy", r"chondro\b", r"anoalia", r"paryfis",  # greek cartilage irregularity
    r"diavrosi",  # greek erosion
)
# OA requires cartilage/compartment context (not bare 'medial' from other
# structures, and not meniscus-degeneration grades)
OA_CART_CONTEXT = _rx(
    r"cartilag|condral|condro|chondro|hrskavic|cartilage|compartiment|"
    r"compartimento|compartment|femorotibial|femorotibiaal|tibiofemoral|"
    r"trochle|patell|rotuli|kompartment|gelenkkompartiment|kraakbeen|"
    r"surface|faceta|facet|plato|plateau|condyle|condil|cond|"
    r"esarthrio|arthrikoy chondroy|arthriko"  # greek compartment/cartilage
)
OA_PF_ONLY = re.compile(_PF)  # reuse the full PF pattern (patella/trochlea/...)
OA_SOFT = _rx(  # early/minimal OA -> hedged
    r"minimal", r"mild", r"incipient", r"early", r"gering", r"kleine",
    r"leichte", r" discrete", r"discreet",
)

# --- effusion ---
EFFUSION = _rx(
    r"effusion", r"efuzyon", r"derrame", r"erguss", r"gelenkerguss",
    r"vocht", r"epanchement", r"sivi", r"sv ", r"izliv", r"izljev", r"izlejev",
    r"liquid", r"liquido", r"hemart", r"hemarth", r"haemarth", r"hemorr",
    r"fluid", r"vochtophoping", r"opzetting", r"recessus", r"burs",
    r"artikul\w* sivi", r"sivi miktar", r"sivi birik", r"sivi art",
    r"sv art", r"sv arts", r"sv miktar", r"eklem sivisi", r"suprapatell",
    r"suprapatel", r"retropatell", r"bursa",
    r"ygroy", r"ygro",  # greek fluid (ygroy endarthrika = intra-articular)
    r"hydrops",
)
EFFUSION_NEG_ONLY = _rx(  # presence words that are NOT findings alone
    r"free fluid",
)
EFF_MIN = re.compile(r"minimal|trace|small|mild|gering|leichte|hafif|pequeno")


# --- synovitis ---
SYNOVITIS = _rx(
    r"synovitis", r"synoviitis", r"sinovitis", r"sinovit", r"synovialitis",
    r"reizsynovial", r"synovial thicken", r"thickened synovium",
    r"thickening of the synovium", r"synovial proliferation",
    r"synovial hypertroph", r"hypertrophy of the synovium", r"verdik\w* synovium",
    r"synovi\w* verdik", r"proliferation synov", r"proliferaciju sinovije",
    r"synovial enhancement", r"villonodular", r"pvns", r"synoviale membraan",
    r"inflam\w* de la sinovial", r"sinovyal", r"bursitis", r"hoffa", r"hoffitis",
    r"synovial",  # bare synovial only counts with thicken/prolifer context
    r"pachynsi toy yena", r"synoviak",  # greek synovial membrane thickening
    r"flegmonod",  # greek inflammatory
)

# --- baker's ---
BAKER = _rx(
    r"baker", r"popliteal cyst", r"poplitea\w* cyst", r"popliteuszyste",
    r"kyste poplit", r"quiste poplit", r"popliteal kist", r"poliklitik",
    r"kist poplit", r"bakerseyste", r"baker-zyste", r"bakers cyste",
    r"zhardeni", r"cyst poplit", r"popliteal burs",
    r"synoviaki kysti baker",  # greek
)

# --- contusion (bone bruise) ---
CONTUSION = _rx(
    r"contusion", r"kontuzyon", r"bruise", r"bone bruise", r"bone-bruise",
    r"bone marrow edema", r"bone marrow oedema", r"marrow edema",
    r"marrow oedema", r"knochenodem", r"knochenodem", r"botoedeem",
    r"kemik iligi odemi", r"kemik odemi", r"kostani edem", r"koštani edem",
    r"kostani edem", r"edema oseo", r"oseo edema", r"bone edema",
    r"edem oseo", r"kemik ilig", r"subchondral", r"sufusion", r"impaction",
    r"impaction injury", r"impresijska", r"impresijske",
    r"osteoyeliko oidia", r"osteoyeliko", r"olopa",  # greek marrow edema/bruise
    r"ontuzionen",  # bulgarian translit variant
    r"kontuzyonel",
)
CONTUSION_BONE = re.compile(  # the edema must be in BONE, not soft tissue
    r"bone|marrow|oseo|osto|ostik|osteoyeliko|kemik|knoch|botoed|kostani|"
    r"kostani|kostno|olopa|condyle|condil|plato|plateau|tibia|tibial|femur|"
    r"femoral|fibula|patell|rotul|molupe|medull"
)
CONTUSION_SOFT = re.compile(  # soft-tissue edema -> NOT contusion
    r"partes blandas|subcutan|periligament|perimenisc|tendon|tendonu|"
    r"kas doku|musk|muscle|weichgewebe|zachte weefsel|soft tissue|"
    r"tejidos blandos|bursa|cyst|zyste|ganglion|retinakul|infrapatell\w* fat|"
    r"hoffa|fettgewebe|subkutan"
)
CONTUSION_EXCL = re.compile(  # these edemas are degenerative, not trauma
    r"degenerative|subchondral|insufficiency|chronic|sequel|"
    r"secuela|chronischem|cystic change|zystoid|"
    r"rekonverz|postoperat|nach postop|arthritis|arthrose|"
    r"artrose|artrosis|osteoarthrit|ueberreizung"
)
CONTUSION_CONFIRM = re.compile(
    r"contusion|kontuzyon|bruise|impaction|impresij|trauma|stress|pivot|"
    r"olopa|molo|ontuzion|kakos"  # greek/bulgarian bruise/trauma markers
)

# --- fracture ---
FRACTURE = _rx(
    r"fractur", r"fraktur", r"fractura", r"frattura", r"kirik", r"\bkirik\b",
    r"fissur", r"frakture", r"break", r"avulsion", r"ausriss", r"trummer",
    r"trümmer", r"impression fracture", r"depression fracture", r"split",
    r"osteochondral defect", r"flake", r"ossikel", r"insufficiency fracture",
    r"stress fracture", r"trabecular", r"subcortical fracture",
    r"frakt", r"lfrakture", r"haarriss", r"hairline",
)
FRACTURE_SOFT = re.compile(
    r"osteochondral|chondral|flake|ossikel|insufficiency"
)

# ligament bruise (MCL edema) does not mean fracture; but avulsion of bone does


def _hits(sent: str, rx: re.Pattern) -> bool:
    return bool(rx.search(sent))


def extract_labels(text: str) -> Dict[str, float]:
    norm = normalize(text)
    if not norm:
        return {c: -1.0 for c in LABELS}

    body = _SECTION_STRIP.sub(" ", norm)
    sents = sentences(body)

    out: Dict[str, float] = {c: -1.0 for c in LABELS}

    def consider(label: str, value: float):
        if value > out[label]:
            out[label] = value

    # tricompartmental OA
    TRICOMP = re.compile(
        r"tricompartiment|tricompartment|tri-compartment|three compartment|"
        r"todos los compartimentos|alle kompartiment|panarthrose|"
        r"compartmens|compartments\b"
    )

    for sent in sents:
        if not sent:
            continue
        pol = sentence_polarity(sent)
        if pol == 0.0:
            # negated/normal sentence still records 0.0 evidence
            record_zero = True
        else:
            record_zero = False

        # ---- ACL: anchor + injury in same sentence, PCL must NOT be the anchor
        if ACL_ANCHOR.search(sent) and not PCL_ANCHOR.search(sent):
            if ACL_INJURY.search(sent):
                consider("ACL", pol)
            elif record_zero:
                consider("ACL", 0.0)
        elif ACL_ANCHOR.search(sent) and PCL_ANCHOR.search(sent):
            # ambiguous sentence: require acl-specific phrasing
            if re.search(r"acl|vkb|anterior cruciate|cruzado anterior|lca",
                         sent) and ACL_INJURY.search(sent):
                consider("ACL", pol)

        # ---- MCL (severity graded: grade-1/low-grade != injury per gold)
        if MCL_ANCHOR.search(sent) and not LCL_ANCHOR.search(sent):
            if MCL_INJURY.search(sent):
                if MCL_GRADE1.search(sent) and not MCL_HIGH.search(sent):
                    consider("MCL", min(pol, 0.25))
                else:
                    consider("MCL", pol)
            elif record_zero:
                consider("MCL", 0.0)
        elif MCL_ANCHOR.search(sent) and LCL_ANCHOR.search(sent):
            if re.search(r"mcl|medial collateral|medial kollateral", sent) \
                    and MCL_INJURY.search(sent):
                if MCL_GRADE1.search(sent) and not MCL_HIGH.search(sent):
                    consider("MCL", min(pol, 0.25))
                else:
                    consider("MCL", pol)

        # ---- menisci (laterality from sentence context)
        if MENISCUS_ANCHOR.search(sent):
            is_med = re.search(_MED, sent)
            is_lat = re.search(_LAT, sent)
            injured = MEN_INJURY.search(sent)
            degen = MEN_DEGEN.search(sent)
            if injured and not MEN_NORMAL.search(sent):
                if is_med:
                    consider("Medial Meniscus", pol)
                if is_lat:
                    consider("Lateral Meniscus", pol)
            elif degen and not MEN_NORMAL.search(sent):
                # degeneration alone -> hedged (many gold labels count it)
                if is_med:
                    consider("Medial Meniscus", min(pol, 0.5))
                if is_lat:
                    consider("Lateral Meniscus", min(pol, 0.5))
            elif record_zero:
                if is_med:
                    consider("Medial Meniscus", 0.0)
                if is_lat:
                    consider("Lateral Meniscus", 0.0)

        # ---- OA per compartment (needs cartilage context; PF anatomy
        # mentioned -> patellofemoral, never medial/lateral OA)
        if OA_TERMS.search(sent):
            soft = bool(OA_SOFT.search(sent))
            val = 0.5 if soft else pol
            has_pf = bool(OA_PF_ONLY.search(sent))
            if OA_CART_CONTEXT.search(sent):
                med_ctx = bool(_MED_RE.search(sent))
                lat_ctx = bool(_LAT_RE.search(sent))
                if med_ctx and not has_pf:
                    consider("Medial OA", val)
                if lat_ctx and not has_pf:
                    consider("Lateral OA", val)
                if has_pf:
                    consider("PF OA", val)
            if TRICOMP.search(sent):
                consider("Medial OA", val)
                consider("Lateral OA", val)
                consider("PF OA", val)

        # ---- effusion
        if EFFUSION.search(sent):
            # 'mild/small/minimal' effusions are graded inconsistently in gold
            # (text alone can't decide) -> soft score; bare/marked stay strong
            if EFF_MIN.search(sent):
                consider("Effusion", min(pol, 0.5))
            elif re.search(r"bursa|bursit|recessus|suprapatell", sent) and \
                    not re.search(r"effusion|erguss|derrame|izliv|izljev|"
                                  r"sivi|sv |efuzyon|vocht|fluid", sent):
                # fluid-ish signal but only in bursa context, not joint
                consider("Effusion", min(pol, 0.5))
            else:
                consider("Effusion", pol)

        # ---- synovitis: bare 'synovial' needs thickening context
        if SYNOVITIS.search(sent):
            if re.search(r"synovial(?! ?membr)", sent) and not re.search(
                r"thicken|verdik|prolifer|hypertroph|reiz|synovitis|"
                r"sinovitis|synoviitis|bursitis|hoffa|hoffitis|enhancement",
                sent,
            ):
                pass  # bare 'synovial membrane' - no finding
            else:
                consider("Synovitis", pol)

        # ---- baker's
        if BAKER.search(sent):
            consider("Baker's", pol)

        # ---- contusion: needs BONE edema + trauma context; degenerative
        # subchondral edema and soft-tissue edema are excluded
        if CONTUSION.search(sent):
            if CONTUSION_SOFT.search(sent) and not CONTUSION_CONFIRM.search(sent):
                consider("Contusion", 0.0)  # soft-tissue edema
            elif CONTUSION_EXCL.search(sent) and not CONTUSION_CONFIRM.search(sent):
                consider("Contusion", 0.0)  # degenerative edema != contusion
            elif CONTUSION_BONE.search(sent):
                consider("Contusion", pol)
            else:
                consider("Contusion", min(pol, 0.5))

        # ---- fracture
        if FRACTURE.search(sent):
            if FRACTURE_SOFT.search(sent) and not re.search(
                r"fractur|fraktur|kirik|fractura|avulsion|ausriss", sent
            ):
                consider("Fracture", min(pol, 0.5))
            else:
                consider("Fracture", pol)

    return out


def extract_frame(df: "pandas.DataFrame", report_col: str = "Report") -> "pandas.DataFrame":
    import pandas as pd

    rows = [extract_labels(str(t)) for t in df[report_col].fillna("")]
    return pd.DataFrame(rows, index=df.index)
