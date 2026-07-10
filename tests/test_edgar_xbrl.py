"""Unit tests for the XBRL fundamentals extraction layer — no DB access.

Exercises the alias-resolution + imputation + accounting-identity checks
against synthetic facts. The DB-facing pieces (connect/get_contexts/fetch_queue)
need a live Postgres and aren't covered here.
"""

from stock_research import edgar_xbrl


class _FakeCursor:
    """Minimal cursor stub: records the query and returns canned rows."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params

    def fetchall(self):
        return self._rows


def _resolver_with(values: dict, context_id: int = 1) -> edgar_xbrl._FactResolver:
    """A resolver pre-loaded with {local_name: value} at ``context_id``, no DB call."""
    r = edgar_xbrl._FactResolver.__new__(edgar_xbrl._FactResolver)
    r._cache = {(k, context_id): v for k, v in values.items()}
    return r


def test_fact_resolver_prefers_first_alias_with_data():
    cur = _FakeCursor([("Assets", 1, 100.0), ("AssetsCurrent", 1, 40.0)])
    resolver = edgar_xbrl._FactResolver(cur, accession_id=1, context_ids=[1])
    assert resolver.resolve(["Assets", "AssetsCurrent"], 1) == 100.0
    assert resolver.resolve(["DoesNotExist"], 1) == 0.0
    val, name = resolver.resolve_named(["DoesNotExist", "AssetsCurrent"], 1)
    assert val == 40.0 and name == "AssetsCurrent"


def test_fact_resolver_skips_query_when_no_contexts():
    cur = _FakeCursor([])
    resolver = edgar_xbrl._FactResolver(cur, accession_id=1, context_ids=[None])
    assert not hasattr(cur, "last_sql")            # execute() never called
    assert resolver.resolve(["Assets"], None) == 0.0
    assert resolver.get("Assets", None) is None


def test_check_or_na_all_zero_is_na():
    assert edgar_xbrl.check_or_na(5.0, 0, 0, 0) is None
    assert edgar_xbrl.check_or_na(0.0, 100, 100) == 0.0


def test_extract_balance_sheet_identity_holds_with_imputation():
    # NoncurrentAssets/NoncurrentLiabilities are never tagged directly -> imputed.
    resolver = _resolver_with({
        "Assets": 1000.0,
        "AssetsCurrent": 400.0,
        "LiabilitiesAndStockholdersEquity": 1000.0,
        "Liabilities": 600.0,
        "LiabilitiesCurrent": 250.0,
        "StockholdersEquity": 400.0,
    })
    bs, checks = edgar_xbrl.extract_balance_sheet(resolver, 1)

    assert bs["NoncurrentAssets"] == 600.0          # BS-Impute-106: Assets - CurrentAssets
    assert bs["NoncurrentLiabilities"] == 350.0     # BS-Impute-113: Liabilities - CurrentLiabilities
    assert bs["Equity"] == 400.0                    # BS-Impute-109: no NCI tagged -> = parent
    assert checks["BS2"] == 0.0                     # Assets == LiabilitiesAndEquity
    assert checks["BS3"] == 0.0                     # Assets == Current + Noncurrent
    assert checks["BS4"] == 0.0                     # Liabilities == Current + Noncurrent
    assert checks["BS1"] == 0.0                     # Equity imputed == Parent (NCI imputed to 0)


def test_extract_income_statement_multistep_identity():
    resolver = _resolver_with({
        "Revenues": 1000.0,
        "CostOfRevenue": 600.0,
        "GrossProfit": 400.0,
        "OperatingExpenses": 250.0,
        "OperatingIncomeLoss": 150.0,
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": 150.0,
        "IncomeTaxExpenseBenefit": 30.0,
        "ProfitLoss": 120.0,
        "NetIncomeLoss": 120.0,
    })
    is_, checks = edgar_xbrl.extract_income_statement(resolver, 1)

    assert checks["IS1"] == 0.0     # GrossProfit == Revenues - CostOfRevenue
    assert checks["IS2"] == 0.0     # OperatingIncomeLoss == GrossProfit - OpEx (+OOI=0)
    assert checks["IS5"] == 0.0     # AfterTax == BeforeTax - Tax (imputed)
    assert checks["IS6"] == 0.0     # NetIncomeLoss == AfterTax (no disc ops/extraordinary)
    assert is_["IncomeLossAfterTax"] == 120.0


def test_extract_income_statement_single_step_gates_gross_profit_check():
    resolver = _resolver_with({
        "Revenues": 1000.0,
        "CostsAndExpenses": 850.0,
        "OperatingIncomeLoss": 150.0,
        "ProfitLoss": 150.0,
    })
    _, checks = edgar_xbrl.extract_income_statement(resolver, 1)
    assert checks["IS1"] is None    # single-step: no GrossProfit/CostOfRevenue reported


def test_extract_cash_flow_identity_holds_with_imputation():
    resolver = _resolver_with({
        "NetCashProvidedByUsedInOperatingActivities": 100.0,
        "NetCashProvidedByUsedInInvestingActivities": -40.0,
        "NetCashProvidedByUsedInFinancingActivities": -20.0,
    })
    cf, checks = edgar_xbrl.extract_cash_flow(resolver, 1)

    assert cf["NetCashFlow"] == 40.0     # CF-Impute: total = sum of the three sections
    assert checks["CF1"] == 0.0


def test_db_config_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("XBRL_DB_HOST", "otherhost")
    monkeypatch.setenv("XBRL_DB_PORT", "6543")
    cfg = edgar_xbrl._db_config()
    assert cfg["host"] == "otherhost"
    assert cfg["port"] == 6543
