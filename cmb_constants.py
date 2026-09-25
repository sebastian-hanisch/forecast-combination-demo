"""Konstanten der Kombinations-Demo: Vehikel "Tagesaufträge mehrerer Depots" (Stück 9 der Zeitreihen-Prognose-Linie), Mitglieder des Pools, Kombinationsverfahren, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095
FIRST_TEST = 730
FIT_END = 610                 # die Mitglieder schätzen ihre Parameter auf den Tagen vor FIT_END; die Tage danach sind das Kalibrierfenster für die Gewichte
CAL_MAX = FIRST_TEST - FIT_END
LEVEL = 100.0
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
HOLIDAY_DROP = 0.5
HOLIDAY_REBOUND = 0.15
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3
SEASON_PERIOD = 7
AR_PHI = 0.98                 # Rückkehr der depoteigenen Niveauschwankung zum Mittel

# --- Glättung aus Stück 2 ----------------------------------------------------------------------------------------------------------------------------
ETS_MODELS = {"hw_mult": ("add", "mul")}
INIT_DAYS = 28
INIT_WEEKS = 8
FIT_STAGE1 = 3000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98

# --- Mitglieder des Pools ------------------------------------------------------------------------------------------------------------------------------
MEMBERS = ("naive", "snaive1", "wm", "hw", "reg", "gbm")
MEMBER_NAMES = {"naive": "Naiv (letzter Wert)", "snaive1": "Letzte Woche (Stück 1)", "wm": "Wochenmittel (Stück 1)", "hw": "Holt-Winters (Stück 2)", "reg": "Regression (Stück 4)", "gbm": "Boosting, global (Stück 6)"}
MEMBER_SHORT = {"naive": "Naiv", "snaive1": "Letzte Woche", "wm": "Wochenmittel", "hw": "Holt-Winters", "reg": "Regression", "gbm": "Boosting"}
DEFAULT_MEMBERS = ["wm", "hw", "reg", "gbm"]
GBM_ROUNDS = 80

# --- Kombinationsverfahren (Reihenfolge = Anzeige) ------------------------------------------------------------------------------------------------------
METHODS = ("mean", "median", "trim", "invmse", "ols", "simplex", "ewa", "select")
METHOD_NAMES = {"mean": "Einfacher Mittelwert", "median": "Median", "trim": "Getrimmter Mittelwert", "invmse": "Inverse-MSE-Gewichte", "ols": "Kleinste Quadrate (frei)", "simplex": "Kleinste Quadrate (nicht negativ, Summe 1)",
                "ewa": "Exponentiell gewichtet (adaptiv)", "select": "Bestes Mitglied (Auswahl)"}
METHOD_SHORT = {"mean": "Mittelwert", "median": "Median", "trim": "Getrimmt", "invmse": "Inverse MSE", "ols": "LS frei", "simplex": "LS Simplex", "ewa": "Adaptiv", "select": "Auswahl"}

# --- Regler und Voreinstellungen ------------------------------------------------------------------------------------------------------------------------
DEPOTS_MIN, DEPOTS_MAX, DEPOTS_STEP, DEFAULT_DEPOTS = 10, 100, 10, 30
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.04, 0.4, 0.02, 0.14
TREND_MIN, TREND_MAX, TREND_STEP, DEFAULT_TREND = -20, 40, 5, 10
SWING_MIN, SWING_MAX, SWING_STEP, DEFAULT_SWING = 0.0, 0.2, 0.02, 0.06
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 14
WINDOW_MIN, WINDOW_MAX, WINDOW_STEP, DEFAULT_WINDOW = 30, CAL_MAX, 15, 90
ETA_MIN, ETA_MAX, ETA_STEP, DEFAULT_ETA = 0.0, 20.0, 1.0, 5.0
TRIM_SHARE = 0.25             # getrimmter Mittelwert: je Seite der kleinste / größte Anteil der Mitglieder (mindestens eines bei mindestens vier Mitgliedern)

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(3))
EXP_DEPOTS = 20
WINDOW_LEVELS = (30, 60, 120)
POOL_SIZES = (2, 3, 4, 5, 6)
SWING_LEVELS = (0.0, 0.06, 0.12)
