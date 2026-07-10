"""Point-in-time fundamentals from the local XBRL Postgres DB.

:mod:`edgar` pulls a curated set of concepts per-ticker from SEC's companyfacts
API — fine for a watchlist, but no way to build a real small-cap *universe*
(one company at a time, rate-limited). The EDGAR-pipeline-share notebooks
(1Discovery -> 2Download -> 3ArelleLoad) instead discover every 10-K filer,
download the filings, and load their XBRL facts into a local Postgres DB via
Arelle's ``xbrlDB`` plugin (``accession``/``fact``/``element``/``qname``/
``context``/``document`` — the XBRL-US public schema) plus a ``filing_manifest``
ledger table (SEC metadata: cik, company_name, form_type, period, filed_date,
zip_path).

This module extracts a full balance-sheet/income-statement/cash-flow row per
filing from that DB, using Hoffman's Fundamental Accounting Concepts method:
alias lists (companies tag the same idea with different XBRL elements),
imputation rules to fill gaps from related concepts, and cross-checks (the
accounting identities each statement must satisfy) so bad extractions are
visible rather than silent. The extraction logic mirrors
``4Fundamentals_formulas.ipynb`` field-for-field; the only change is threading
an explicit per-accession fact cache (:class:`_FactResolver`) instead of the
notebook's module-level global, so this is safe to call from anywhere.

Restricted by default to filings filed on/after 2019-01-01: inline XBRL became
mandatory for all filers by then, which keeps the corpus in one document
format and means the filing's narrative text (MD&A, risk factors) lives in the
very same primary document Arelle already parsed — useful once LLM text
scoring is added on top of this.

Requires the ``xbrl`` extra (``pip install -e ".[xbrl]"``) and a local Postgres
DB populated by the pipeline notebooks. Connection defaults match the
notebooks (``localhost``); override with ``XBRL_DB_HOST``/``PORT``/``NAME``/
``USER``/``PASSWORD`` env vars.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT

XBRL_DIR = REPO_ROOT / "data" / "xbrl_fundamentals"
DEFAULT_SINCE = "2019-01-01"  # inline-XBRL cutover: uniform format + co-located filing text


def _db_config() -> dict:
    return dict(
        host=os.environ.get("XBRL_DB_HOST", "localhost"),
        port=int(os.environ.get("XBRL_DB_PORT", "5432")),
        database=os.environ.get("XBRL_DB_NAME", "postgres"),
        user=os.environ.get("XBRL_DB_USER", "postgres"),
        password=os.environ.get("XBRL_DB_PASSWORD", "password"),
    )


def connect():
    """Open a connection to the local XBRL Postgres DB (needs the ``xbrl`` extra)."""
    import psycopg2
    return psycopg2.connect(**_db_config())


# --------------------------------------------------------------------------- #
# Context discovery + fact resolution (mirrors 4Fundamentals_formulas Cells 2-3).
# --------------------------------------------------------------------------- #
def get_contexts(cur, accession_id: int):
    """Return (doc_period_end_date, instant_ctx_id, duration_ctx_id) for a filing.

    Anchored to DocumentPeriodEndDate: instant_ctx is the balance-sheet-date
    context (no dimensions), duration_ctx is the longest YTD context ending on
    the same date (no dimensions).
    """
    cur.execute("""
        SELECT c.period_end::date
        FROM fact f
        JOIN element e  ON f.element_id  = e.element_id
        JOIN qname  q   ON e.qname_id    = q.qname_id
        JOIN context c  ON f.context_id  = c.context_id
        WHERE f.accession_id          = %s
          AND q.local_name            = 'DocumentPeriodEndDate'
          AND c.specifies_dimensions  = false
        ORDER BY c.period_end DESC
        LIMIT 1
    """, (accession_id,))
    row = cur.fetchone()
    if not row:
        return None, None, None
    doc_end = row[0]

    cur.execute("""
        SELECT context_id FROM context
        WHERE accession_id         = %s
          AND period_instant::date = %s
          AND specifies_dimensions = false
        LIMIT 1
    """, (accession_id, doc_end))
    row = cur.fetchone()
    instant_ctx = row[0] if row else None

    cur.execute("""
        SELECT context_id FROM context
        WHERE accession_id         = %s
          AND period_end::date     = %s
          AND period_start IS NOT NULL
          AND specifies_dimensions = false
        ORDER BY (period_end::date - period_start::date) DESC
        LIMIT 1
    """, (accession_id, doc_end))
    row = cur.fetchone()
    duration_ctx = row[0] if row else None

    return doc_end, instant_ctx, duration_ctx


class _FactResolver:
    """Per-accession fact cache + alias resolution.

    Replaces the notebook's module-level ``FACT_CACHE`` global with an
    instance so extraction is safe to call concurrently / from anywhere.
    """

    def __init__(self, cur, accession_id: int, context_ids: list) -> None:
        self._cache: dict[tuple[str, int], float] = {}
        ctxs = [c for c in context_ids if c is not None]
        if not ctxs:
            return
        cur.execute("""
            SELECT q.local_name, f.context_id, f.effective_value
            FROM fact    f
            JOIN element e ON f.element_id = e.element_id
            JOIN qname   q ON e.qname_id   = q.qname_id
            WHERE f.accession_id = %s
              AND f.context_id   = ANY(%s)
              AND f.effective_value IS NOT NULL
            ORDER BY f.ultimus_index ASC NULLS LAST
        """, (accession_id, ctxs))
        for name, ctx, val in cur.fetchall():
            key = (name, ctx)
            if key not in self._cache:
                self._cache[key] = float(val)

    def get(self, local_name: str, context_id) -> float | None:
        if context_id is None:
            return None
        return self._cache.get((local_name, context_id))

    def resolve(self, aliases: list[str], context_id) -> float:
        """Try aliases in order, return the first non-None value or 0."""
        for name in aliases:
            val = self.get(name, context_id)
            if val is not None:
                return val
        return 0.0

    def resolve_named(self, aliases: list[str], context_id):
        """Like resolve(), but also returns which alias matched (or None)."""
        for name in aliases:
            val = self.get(name, context_id)
            if val is not None:
                return val, name
        return 0.0, None


def check_or_na(residual, *terms):
    """Return the check residual, or None ("N/A") when every input term is zero.

    A check computed entirely from absent concepts is inapplicable, not passing.
    """
    if all(t == 0 for t in terms):
        return None
    return residual


# --------------------------------------------------------------------------- #
# Balance sheet (mirrors 4Fundamentals_formulas Cell 4). Checks BS1-BS5.
# --------------------------------------------------------------------------- #
def extract_balance_sheet(resolver: _FactResolver, ctx):
    r = {}

    r['Assets'] = resolver.resolve(['Assets', 'AssetsCurrent'], ctx)
    r['CurrentAssets'] = resolver.resolve(['AssetsCurrent'], ctx)
    r['NoncurrentAssets'] = resolver.resolve(['AssetsNoncurrent'], ctx)

    r['LiabilitiesAndEquity'] = resolver.resolve([
        'LiabilitiesAndStockholdersEquity',
        'LiabilitiesAndPartnersCapital',
        'LiabilitiesAndMembersEquity'
    ], ctx)
    r['Liabilities'] = resolver.resolve(['Liabilities'], ctx)
    r['CurrentLiabilities'] = resolver.resolve(['LiabilitiesCurrent'], ctx)
    r['NoncurrentLiabilities'] = resolver.resolve(['LiabilitiesNoncurrent'], ctx)
    r['CommitmentsAndContingencies'] = resolver.resolve(['CommitmentsAndContingencies'], ctx)

    r['Equity'] = resolver.resolve([
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest',
        'LimitedLiabilityCompanyLlcMembersEquityIncludingPortionAttributableToNoncontrollingInterest'
    ], ctx)
    r['EquityAttributableToParent'] = resolver.resolve([
        'StockholdersEquity', 'PartnersCapital', 'MembersEquity'
    ], ctx)
    r['EquityAttributableToNCI'] = resolver.resolve([
        'MinorityInterest',
        'PartnersCapitalAttributableToNoncontrollingInterest',
        'MembersEquityAttributableToNoncontrollingInterest'
    ], ctx)

    # Temp variables used only for TemporaryEquity imputation.
    _redeemable_nci_common = resolver.resolve(['RedeemableNoncontrollingInterestEquityCommonCarryingAmount'], ctx)
    _redeemable_nci_preferred = resolver.resolve(['RedeemableNoncontrollingInterestEquityPreferredCarryingAmount'], ctx)
    _redeemable_nci_other = resolver.resolve(['RedeemableNoncontrollingInterestEquityOtherCarryingAmount'], ctx)
    _redeemable_nci = resolver.resolve([
        'RedeemableNoncontrollingInterestEquityCarryingAmount',
        'RedeemableNoncontrollingInterestEquityFairValue',
        'RedeemableNoncontrollingInterestEquityOtherFairValue'
    ], ctx)
    _temp_equity_parent = resolver.resolve(['TemporaryEquityCarryingAmountAttributableToParent'], ctx)

    # BS-Impute-101
    if _redeemable_nci == 0:
        _redeemable_nci = _redeemable_nci_common + _redeemable_nci_preferred + _redeemable_nci_other

    r['TemporaryEquity'] = resolver.resolve([
        'TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests',
        'RedeemablePreferredStockCarryingAmount',
        'TemporaryEquityValueExcludingAdditionalPaidInCapital'
    ], ctx)
    # BS-Impute-102
    if r['TemporaryEquity'] == 0:
        r['TemporaryEquity'] = _temp_equity_parent + _redeemable_nci

    r = _impute_bs(r)

    checks = {
        'BS1': check_or_na(r['Equity'] - (r['EquityAttributableToParent'] + r['EquityAttributableToNCI']),
                           r['Equity'], r['EquityAttributableToParent'], r['EquityAttributableToNCI']),
        'BS2': check_or_na(r['Assets'] - r['LiabilitiesAndEquity'],
                           r['Assets'], r['LiabilitiesAndEquity']),
        'BS3': check_or_na(r['Assets'] - (r['CurrentAssets'] + r['NoncurrentAssets']),
                           r['Assets'], r['CurrentAssets'], r['NoncurrentAssets']),
        'BS4': check_or_na(r['Liabilities'] - (r['CurrentLiabilities'] + r['NoncurrentLiabilities']),
                           r['Liabilities'], r['CurrentLiabilities'], r['NoncurrentLiabilities']),
        'BS5': check_or_na(r['LiabilitiesAndEquity'] - (
                   r['Liabilities'] + r['CommitmentsAndContingencies']
                   + r['TemporaryEquity'] + r['Equity']),
                           r['LiabilitiesAndEquity'], r['Liabilities'], r['CommitmentsAndContingencies'],
                           r['TemporaryEquity'], r['Equity']),
    }
    return r, checks


def _impute_bs(r):
    if r['Assets'] == 0 and r['NoncurrentAssets'] == 0 \
            and r['Assets'] != r['LiabilitiesAndEquity'] \
            and r['CurrentAssets'] == r['LiabilitiesAndEquity']:
        r['Assets'] = r['CurrentAssets']
    if r['Assets'] == 0 and r['LiabilitiesAndEquity'] != 0 \
            and r['CurrentAssets'] == r['LiabilitiesAndEquity']:
        r['Assets'] = r['CurrentAssets']
    if r['Assets'] == 0 and r['NoncurrentAssets'] == 0 \
            and r['LiabilitiesAndEquity'] != 0 \
            and r['LiabilitiesAndEquity'] == (r['Liabilities'] + r['Equity']):
        r['Assets'] = r['CurrentAssets']
    if r['NoncurrentAssets'] == 0 and r['Assets'] != 0 and r['CurrentAssets'] != 0:
        r['NoncurrentAssets'] = r['Assets'] - r['CurrentAssets']
    if r['LiabilitiesAndEquity'] == 0 and r['Assets'] != 0:
        r['LiabilitiesAndEquity'] = r['Assets']
    if r['Equity'] == 0 and r['EquityAttributableToNCI'] != 0 \
            and r['EquityAttributableToParent'] != 0:
        r['Equity'] = r['EquityAttributableToParent'] + r['EquityAttributableToNCI']
    if r['Equity'] == 0 and r['EquityAttributableToNCI'] == 0 \
            and r['EquityAttributableToParent'] != 0:
        r['Equity'] = r['EquityAttributableToParent']
    if r['Equity'] == 0:
        r['Equity'] = r['EquityAttributableToParent'] + r['EquityAttributableToNCI']
    if r['EquityAttributableToParent'] == 0 and r['Equity'] != 0 \
            and r['EquityAttributableToNCI'] != 0:
        r['EquityAttributableToParent'] = r['Equity'] - r['EquityAttributableToNCI']
    if r['EquityAttributableToParent'] == 0 and r['Equity'] != 0 \
            and r['EquityAttributableToNCI'] == 0:
        r['EquityAttributableToParent'] = r['Equity']
    if r['NoncurrentLiabilities'] == 0 and r['CurrentLiabilities'] != 0 \
            and r['Liabilities'] != 0:
        r['NoncurrentLiabilities'] = r['Liabilities'] - r['CurrentLiabilities']
    if r['Liabilities'] == 0 and r['CurrentLiabilities'] != 0 \
            and r['NoncurrentLiabilities'] != 0:
        r['Liabilities'] = r['CurrentLiabilities'] + r['NoncurrentLiabilities']
    if r['Liabilities'] == 0 and r['Equity'] != 0:
        r['Liabilities'] = r['LiabilitiesAndEquity'] - (
            r['CommitmentsAndContingencies'] + r['TemporaryEquity'] + r['Equity'])
    if r['NoncurrentLiabilities'] == 0 and r['CurrentLiabilities'] != 0 \
            and r['Liabilities'] != 0:
        r['NoncurrentLiabilities'] = r['Liabilities'] - r['CurrentLiabilities']
    if r['Liabilities'] == 0 and r['CurrentLiabilities'] != 0 \
            and r['NoncurrentLiabilities'] == 0:
        r['Liabilities'] = r['CurrentLiabilities']
    if r['EquityAttributableToParent'] != 0 and r['Equity'] != 0 \
            and r['EquityAttributableToNCI'] != 0 \
            and r['EquityAttributableToParent'] == r['Equity']:
        r['EquityAttributableToParent'] = r['Equity'] - r['EquityAttributableToNCI']
    if r['CurrentLiabilities'] == 0 and r['NoncurrentLiabilities'] == 0 \
            and r['Liabilities'] != 0:
        r['CurrentLiabilities'] = r['Liabilities']
    if r['EquityAttributableToNCI'] == 0 and r['Equity'] != 0 \
            and r['EquityAttributableToParent'] != 0:
        r['EquityAttributableToNCI'] = r['Equity'] - r['EquityAttributableToParent']
    if r['CurrentAssets'] == 0 and r['Assets'] != 0 \
            and r['LiabilitiesAndEquity'] != 0 \
            and r['LiabilitiesAndEquity'] == r['Assets']:
        r['CurrentAssets'] = r['Assets']
    if (r['TemporaryEquity'] == 0 and r['Liabilities'] != 0 and r['Equity'] != 0
            and r['LiabilitiesAndEquity'] != 0
            and r['Liabilities'] == (r['CurrentLiabilities'] + r['NoncurrentLiabilities'])
            and r['Equity'] == (r['EquityAttributableToParent'] + r['EquityAttributableToNCI'])):
        r['TemporaryEquity'] = r['LiabilitiesAndEquity'] - (
            (r['Liabilities'] + r['Equity']) - r['CommitmentsAndContingencies'])
    return r


# --------------------------------------------------------------------------- #
# Income statement (mirrors 4Fundamentals_formulas Cell 5). Checks IS1-IS10.
# --------------------------------------------------------------------------- #
def extract_income_statement(resolver: _FactResolver, ctx):
    r = {}

    r['Revenues'] = resolver.resolve([
        'Revenues', 'SalesRevenueNet', 'SalesRevenueServicesNet', 'SalesRevenueGoodsNet',
        'RevenuesNetOfInterestExpense', 'RealEstateRevenueNet',
        'InterestAndDividendIncomeOperating', 'RevenueMineralSales', 'OilAndGasRevenue',
        'FinancialServicesRevenue', 'RegulatedAndUnregulatedOperatingRevenue',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'HealthCareOrganizationRevenue',
        'HealthCareOrganizationRevenueNetOfPatientServiceRevenueProvisions',
        'InterestIncomeExpenseNet', 'RevenuesExcludingInterestAndDividends',
        'InvestmentBankingRevenue', 'NetInvestmentIncome',
        'FeesAndCommissions', 'OtherSalesRevenueNet'
    ], ctx)
    r['CostOfRevenue'] = resolver.resolve([
        'CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfServices', 'CostOfGoodsSold',
        'CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization',
        'DirectOperatingCosts', 'CostOfGoodsSoldOilAndGas', 'FinancialServicesCosts',
        'ContractRevenueCost', 'CostOfRealEstateRevenue'
    ], ctx)

    # Temp vars for imputation only.
    _tax_current = resolver.resolve(['CurrentIncomeTaxExpenseBenefit'], ctx)
    _tax_deferred = resolver.resolve(['DeferredIncomeTaxExpenseBenefit'], ctx)
    _disc_phase = resolver.resolve(['DiscontinuedOperationIncomeLossFromDiscontinuedOperationDuringPhaseOutPeriodNetOfTax'], ctx)
    _disc_disposal = resolver.resolve(['DiscontinuedOperationGainLossOnDisposalOfDiscontinuedOperationNetOfTax'], ctx)
    _disc_prov = resolver.resolve(['DiscontinuedOperationProvisionForLossGainOnDisposalNetOfTax'], ctx)
    _disc_adj = resolver.resolve(['DiscontinuedOperationAmountOfAdjustmentToPriorPeriodGainLossOnDisposalNetOfTax'], ctx)
    _nci_nonredeem = resolver.resolve([
        'NetIncomeLossAttributableToNonredeemableNoncontrollingInterest',
        'NoncontrollingInterestInNetIncomeLossPreferredUnitHoldersNonredeemable'
    ], ctx)
    _nci_redeem = resolver.resolve([
        'NetIncomeLossAttributableToRedeemableNoncontrollingInterest',
        'NoncontrollingInterestInNetIncomeLossOperatingPartnershipsRedeemable'
    ], ctx)
    _pref_undist = resolver.resolve(['UndistributedEarningsLossAllocatedToParticipatingSecuritiesBasic'], ctx)

    r['GrossProfit'] = resolver.resolve(['GrossProfit'], ctx)
    r['OperatingExpenses'] = resolver.resolve(['OperatingExpenses', 'UtilitiesOperatingExpense'], ctx)
    # Many filers tag only a total-cost subtotal (CostsAndExpenses /
    # OperatingCostsAndExpenses) instead of an OperatingExpenses subtotal.
    # Derive OpEx by removing cost of revenue.
    if r['OperatingExpenses'] == 0:
        _total_costs = resolver.resolve(['CostsAndExpenses', 'OperatingCostsAndExpenses'], ctx)
        if _total_costs != 0 and r['CostOfRevenue'] != 0:
            r['OperatingExpenses'] = _total_costs - r['CostOfRevenue']
    r['OtherOperatingIncome'] = resolver.resolve(['OtherOperatingIncome'], ctx)
    r['OperatingIncomeLoss'] = resolver.resolve([
        'OperatingIncomeLoss',
        'IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet'
    ], ctx)
    r['NonoperatingIncomeLoss'] = 0.0  # disabled -- taxonomy error per Hoffman
    r['IncomeLossBeforeTax'], _bt_element = resolver.resolve_named([
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'
    ], ctx)
    # The second element EXCLUDES equity-method income by definition. When the
    # filer also tags equity-method income, add it back so the IS5/IS6 chain
    # (BeforeTax - Tax -> AfterTax -> NetIncome) is on a consistent basis.
    _equity_method = resolver.resolve(['IncomeLossFromEquityMethodInvestments'], ctx)
    if _bt_element == 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments' \
            and _equity_method != 0:
        r['IncomeLossBeforeTax'] = r['IncomeLossBeforeTax'] + _equity_method
    r['IncomeTaxExpense'] = resolver.resolve([
        'IncomeTaxExpenseBenefit', 'IncomeTaxExpenseBenefitContinuingOperations',
        'FederalHomeLoanBankAssessments',
        'FederalIncomeTaxExpenseBenefitContinuingOperations',
        'StateAndLocalIncomeTaxExpenseBenefitContinuingOperations'
    ], ctx)
    r['IncomeLossAfterTax'], _at_element = resolver.resolve_named([
        'IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest',
        'IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple',
        'IncomeLossFromContinuingOperations'
    ], ctx)
    r['DiscontinuedOps'] = resolver.resolve([
        'IncomeLossFromDiscontinuedOperationsNetOfTax',
        'IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity'
    ], ctx)
    r['ExtraordinaryItems'] = resolver.resolve(['ExtraordinaryItemNetOfTax'], ctx)
    r['NetIncomeLoss'] = resolver.resolve([
        'ProfitLoss', 'IncomeLossIncludingPortionAttributableToNoncontrollingInterest'
    ], ctx)
    r['NetIncomeLossToParent'] = resolver.resolve([
        'NetIncomeLoss', 'IncomeLossAttributableToParent'
    ], ctx)
    r['NetIncomeLossToNCI'] = resolver.resolve([
        'NetIncomeLossAttributableToNoncontrollingInterest',
        'IncomeLossAttributableToNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity'
    ], ctx)
    r['PreferredDividends'] = resolver.resolve(['PreferredStockDividendsAndOtherAdjustments'], ctx)
    r['NetIncomeLossToCommon'] = resolver.resolve(['NetIncomeLossAvailableToCommonStockholdersBasic'], ctx)
    r['OtherComprehensiveIncome'] = resolver.resolve(['OtherComprehensiveIncomeLossNetOfTax'], ctx)
    r['ComprehensiveIncomeLoss'] = resolver.resolve([
        'ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest'
    ], ctx)
    r['ComprehensiveIncomeLossToParent'] = resolver.resolve(['ComprehensiveIncomeNetOfTax'], ctx)
    r['ComprehensiveIncomeLossToNCI'] = resolver.resolve([
        'ComprehensiveIncomeNetOfTaxAttributableToNoncontrollingInterest'
    ], ctx)

    # NCI sign-flip repair: some filers tag NetIncomeLossAttributableTo-
    # NoncontrollingInterest using the deduction convention (negative when it
    # is income). If flipping the sign closes NI = Parent + NCI exactly while
    # the unflipped form does not, flip it.
    if r['NetIncomeLossToNCI'] != 0 \
            and abs(r['NetIncomeLoss'] - (r['NetIncomeLossToParent'] + r['NetIncomeLossToNCI'])) > 1 \
            and abs(r['NetIncomeLoss'] - (r['NetIncomeLossToParent'] - r['NetIncomeLossToNCI'])) <= 1:
        r['NetIncomeLossToNCI'] = -r['NetIncomeLossToNCI']

    # Basis fix-up: 'IncomeLossFromContinuingOperations' is attributable to
    # parent only. When the NCI share is tagged and adding it back closes the
    # net-income rollup exactly, move AfterTax to a consolidated basis.
    if _at_element == 'IncomeLossFromContinuingOperations' and r['NetIncomeLossToNCI'] != 0 \
            and r['IncomeLossAfterTax'] != 0:
        _gap = r['NetIncomeLoss'] - (r['IncomeLossAfterTax'] + r['DiscontinuedOps'] + r['ExtraordinaryItems'])
        if abs(_gap - r['NetIncomeLossToNCI']) <= 1:
            r['IncomeLossAfterTax'] = r['IncomeLossAfterTax'] + r['NetIncomeLossToNCI']

    r = _impute_is(r, _tax_current, _tax_deferred,
                   _disc_phase, _disc_disposal, _disc_prov, _disc_adj,
                   _nci_nonredeem, _nci_redeem, _pref_undist)

    # Fix-up: filer's OperatingExpenses includes cost of revenue (total-cost
    # basis). Detected when OpInc = Revenues - OpEx + OOI holds exactly while
    # the gross-profit form does not; repair by removing CoR from OpEx.
    if r['OperatingIncomeLoss'] != 0 and r['OperatingExpenses'] != 0 and r['CostOfRevenue'] != 0 \
            and abs(r['OperatingIncomeLoss'] - (r['GrossProfit'] - r['OperatingExpenses'] + r['OtherOperatingIncome'])) > 1 \
            and abs(r['OperatingIncomeLoss'] - (r['Revenues'] - r['OperatingExpenses'] + r['OtherOperatingIncome'])) <= 1:
        r['OperatingExpenses'] = r['OperatingExpenses'] - r['CostOfRevenue']

    _single_step = (r['GrossProfit'] == 0 and r['CostOfRevenue'] == 0)
    checks = {
        'IS1': None if _single_step else check_or_na(
                    r['GrossProfit'] - (r['Revenues'] - r['CostOfRevenue']),
                    r['GrossProfit'], r['Revenues'], r['CostOfRevenue']),
        'IS2': None if (_single_step or r['OperatingIncomeLoss'] == 0) else check_or_na(
                    r['OperatingIncomeLoss'] - (r['GrossProfit'] - r['OperatingExpenses'] + r['OtherOperatingIncome']),
                    r['OperatingIncomeLoss'], r['GrossProfit'], r['OperatingExpenses'], r['OtherOperatingIncome']),
        'IS4': None if (r['OperatingIncomeLoss'] == 0 or r['IncomeLossBeforeTax'] == 0) else
                    r['IncomeLossBeforeTax'] - (r['OperatingIncomeLoss'] + r['NonoperatingIncomeLoss']),
        'IS5': check_or_na(r['IncomeLossAfterTax'] - (r['IncomeLossBeforeTax'] - r['IncomeTaxExpense']),
                    r['IncomeLossAfterTax'], r['IncomeLossBeforeTax'], r['IncomeTaxExpense']),
        'IS6': check_or_na(r['NetIncomeLoss'] - (r['IncomeLossAfterTax'] + r['DiscontinuedOps'] + r['ExtraordinaryItems']),
                    r['NetIncomeLoss'], r['IncomeLossAfterTax'], r['DiscontinuedOps'], r['ExtraordinaryItems']),
        'IS7': check_or_na(r['NetIncomeLoss'] - (r['NetIncomeLossToParent'] + r['NetIncomeLossToNCI']),
                    r['NetIncomeLoss'], r['NetIncomeLossToParent'], r['NetIncomeLossToNCI']),
        'IS8': check_or_na(r['NetIncomeLossToCommon'] - (r['NetIncomeLossToParent'] - r['PreferredDividends']),
                    r['NetIncomeLossToCommon'], r['NetIncomeLossToParent'], r['PreferredDividends']),
        'IS9': check_or_na(r['ComprehensiveIncomeLoss'] - (r['ComprehensiveIncomeLossToParent'] + r['ComprehensiveIncomeLossToNCI']),
                    r['ComprehensiveIncomeLoss'], r['ComprehensiveIncomeLossToParent'], r['ComprehensiveIncomeLossToNCI']),
        'IS10': check_or_na(r['ComprehensiveIncomeLoss'] - (r['NetIncomeLoss'] + r['OtherComprehensiveIncome']),
                    r['ComprehensiveIncomeLoss'], r['NetIncomeLoss'], r['OtherComprehensiveIncome']),
    }
    return r, checks


def _impute_is(r, tax_cur, tax_def, disc_phase, disc_disp, disc_prov, disc_adj, nci_nr, nci_r, pref_undist):
    if r['DiscontinuedOps'] == 0:
        r['DiscontinuedOps'] = disc_phase + disc_disp + disc_prov + disc_adj
    if r['NetIncomeLossToNCI'] == 0:
        r['NetIncomeLossToNCI'] = nci_nr + nci_r
    if r['IncomeTaxExpense'] == 0 and (tax_cur != 0 or tax_def != 0):
        r['IncomeTaxExpense'] = tax_cur + tax_def
    if r['NetIncomeLossToParent'] == 0 and r['NetIncomeLossToCommon'] != 0 and r['PreferredDividends'] == 0:
        r['NetIncomeLossToParent'] = r['NetIncomeLossToCommon']
    if r['NetIncomeLossToCommon'] == 0 and r['PreferredDividends'] == 0 and r['NetIncomeLossToParent'] != 0:
        r['NetIncomeLossToCommon'] = r['NetIncomeLossToParent']
    if r['IncomeLossAfterTax'] == 0 and r['IncomeLossBeforeTax'] != 0:
        r['IncomeLossAfterTax'] = r['IncomeLossBeforeTax'] - r['IncomeTaxExpense']
    if r['IncomeLossAfterTax'] == 0 and r['NetIncomeLoss'] != 0 \
            and r['DiscontinuedOps'] == 0 and r['ExtraordinaryItems'] == 0:
        r['IncomeLossAfterTax'] = r['NetIncomeLoss']
    if r['NetIncomeLoss'] == 0 and r['IncomeLossAfterTax'] != 0:
        r['NetIncomeLoss'] = r['IncomeLossAfterTax'] + r['DiscontinuedOps'] + r['ExtraordinaryItems']
    if r['NetIncomeLoss'] == 0 and r['NetIncomeLossToNCI'] == 0 and r['NetIncomeLossToParent'] != 0:
        r['NetIncomeLoss'] = r['NetIncomeLossToParent']
    if r['IncomeLossBeforeTax'] == 0 and r['IncomeLossAfterTax'] != 0 and r['IncomeTaxExpense'] == 0:
        r['IncomeLossBeforeTax'] = r['IncomeLossAfterTax']
    if r['NetIncomeLoss'] == 0 and r['NetIncomeLossToParent'] != 0 and r['NetIncomeLossToNCI'] != 0:
        r['NetIncomeLoss'] = r['NetIncomeLossToParent'] + r['NetIncomeLossToNCI']
    if r['NetIncomeLossToParent'] == 0 and r['NetIncomeLossToNCI'] != 0 and r['NetIncomeLoss'] != 0:
        r['NetIncomeLossToParent'] = r['NetIncomeLoss'] - r['NetIncomeLossToNCI']
    if r['NetIncomeLossToCommon'] == 0 and r['PreferredDividends'] != 0 and r['NetIncomeLossToParent'] != 0:
        r['NetIncomeLossToCommon'] = r['NetIncomeLossToParent'] - r['PreferredDividends']
    if r['NetIncomeLossToParent'] == 0 and r['NetIncomeLossToNCI'] == 0 and r['NetIncomeLoss'] != 0:
        r['NetIncomeLossToParent'] = r['NetIncomeLoss']
    if r['PreferredDividends'] == 0 and r['NetIncomeLossToParent'] != 0 and r['NetIncomeLossToCommon'] != 0:
        r['PreferredDividends'] = r['NetIncomeLossToParent'] - r['NetIncomeLossToCommon']
    if r['NetIncomeLossToCommon'] == 0 and r['PreferredDividends'] == 0 and r['NetIncomeLossToParent'] != 0:
        r['NetIncomeLossToCommon'] = r['NetIncomeLossToParent']
    if r['IncomeLossAfterTax'] == 0 and r['NetIncomeLoss'] != 0:
        r['IncomeLossAfterTax'] = r['NetIncomeLoss'] - r['DiscontinuedOps'] - r['ExtraordinaryItems']
    if r['IncomeLossAfterTax'] == 0 and r['IncomeTaxExpense'] != 0 and r['IncomeLossBeforeTax'] != 0:
        r['IncomeLossAfterTax'] = r['IncomeLossBeforeTax'] - r['IncomeTaxExpense']
    if r['IncomeLossBeforeTax'] == 0 and r['IncomeLossAfterTax'] != 0:
        r['IncomeLossBeforeTax'] = r['IncomeLossAfterTax'] + r['IncomeTaxExpense']
    if r['NonoperatingIncomeLoss'] == 0 and r['IncomeLossBeforeTax'] != 0 and r['OperatingIncomeLoss'] != 0:
        r['NonoperatingIncomeLoss'] = r['IncomeLossBeforeTax'] - r['OperatingIncomeLoss']
    if r['GrossProfit'] == 0 and r['Revenues'] != 0 and r['CostOfRevenue'] != 0:
        r['GrossProfit'] = r['Revenues'] - r['CostOfRevenue']
    # (fixed: include OtherOperatingIncome, omitted in the original VBA --
    # without it every filing with OOI != 0 failed IS2 by exactly OOI)
    if r['OperatingExpenses'] == 0 and r['GrossProfit'] != 0 and r['OperatingIncomeLoss'] != 0:
        r['OperatingExpenses'] = r['GrossProfit'] - r['OperatingIncomeLoss'] + r['OtherOperatingIncome']
    if r['CostOfRevenue'] == 0 and r['GrossProfit'] != 0 and r['Revenues'] != 0:
        r['CostOfRevenue'] = r['Revenues'] - r['GrossProfit']
    if r['NetIncomeLossToNCI'] == 0 and r['NetIncomeLoss'] != 0 and r['NetIncomeLossToParent'] != 0:
        r['NetIncomeLossToNCI'] = r['NetIncomeLoss'] - r['NetIncomeLossToParent']
    # Comprehensive income.
    if r['ComprehensiveIncomeLossToParent'] == 0 and r['ComprehensiveIncomeLossToNCI'] == 0 \
            and r['ComprehensiveIncomeLoss'] != 0:
        r['ComprehensiveIncomeLossToParent'] = r['ComprehensiveIncomeLoss']
    if r['ComprehensiveIncomeLossToParent'] == 0 and r['ComprehensiveIncomeLossToNCI'] != 0 \
            and r['ComprehensiveIncomeLoss'] != 0:
        r['ComprehensiveIncomeLossToParent'] = r['ComprehensiveIncomeLoss'] - r['ComprehensiveIncomeLossToNCI']
    if r['ComprehensiveIncomeLoss'] == 0 and r['ComprehensiveIncomeLossToNCI'] != 0 \
            and r['ComprehensiveIncomeLossToParent'] != 0:
        r['ComprehensiveIncomeLoss'] = r['ComprehensiveIncomeLossToParent'] + r['ComprehensiveIncomeLossToNCI']
    if r['ComprehensiveIncomeLoss'] == 0 and r['ComprehensiveIncomeLossToNCI'] == 0 \
            and r['ComprehensiveIncomeLossToParent'] != 0:
        r['ComprehensiveIncomeLoss'] = r['ComprehensiveIncomeLossToParent']
    if r['ComprehensiveIncomeLossToNCI'] == 0 and r['ComprehensiveIncomeLoss'] != 0 \
            and r['ComprehensiveIncomeLossToParent'] != 0:
        r['ComprehensiveIncomeLossToNCI'] = r['ComprehensiveIncomeLoss'] - r['ComprehensiveIncomeLossToParent']
    if r['ComprehensiveIncomeLoss'] == 0 and r['ComprehensiveIncomeLossToParent'] == 0 \
            and r['ComprehensiveIncomeLossToNCI'] == 0 and r['OtherComprehensiveIncome'] == 0:
        r['ComprehensiveIncomeLoss'] = r['NetIncomeLoss']
    if r['OtherComprehensiveIncome'] == 0 and r['ComprehensiveIncomeLoss'] != 0:
        r['OtherComprehensiveIncome'] = r['ComprehensiveIncomeLoss'] - r['NetIncomeLoss']
    if r['ComprehensiveIncomeLossToParent'] == 0 and r['ComprehensiveIncomeLossToNCI'] == 0 \
            and r['ComprehensiveIncomeLoss'] != 0:
        r['ComprehensiveIncomeLossToParent'] = r['ComprehensiveIncomeLoss']
    if r['ComprehensiveIncomeLossToNCI'] == 0 and r['ComprehensiveIncomeLossToParent'] != 0 \
            and r['ComprehensiveIncomeLoss'] != 0:
        r['ComprehensiveIncomeLossToNCI'] = r['ComprehensiveIncomeLoss'] - r['ComprehensiveIncomeLossToParent']
    return r


# --------------------------------------------------------------------------- #
# Cash flow statement (mirrors 4Fundamentals_formulas Cell 6). Checks CF1-CF6.
# --------------------------------------------------------------------------- #
def extract_cash_flow(resolver: _FactResolver, ctx):
    r = {}

    # Totals that INCLUDE the exchange-rate effect come first; the Excluding
    # variants are a last resort and are flagged so FX can be added back below.
    r['NetCashFlow'], _ncf_element = resolver.resolve_named([
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
        'CashAndCashEquivalentsPeriodIncreaseDecrease',
        'CashPeriodIncreaseDecrease',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect',
        'CashAndCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect'
    ], ctx)
    _ncf_excludes_fx = _ncf_element is not None and 'ExcludingExchangeRateEffect' in _ncf_element
    r['NetCashFlowsOperating'] = resolver.resolve(['NetCashProvidedByUsedInOperatingActivities'], ctx)
    r['NetCashFlowsInvesting'] = resolver.resolve(['NetCashProvidedByUsedInInvestingActivities'], ctx)
    r['NetCashFlowsFinancing'] = resolver.resolve(['NetCashProvidedByUsedInFinancingActivities'], ctx)
    r['NetCashFlowsOperatingContinuing'] = resolver.resolve(['NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'], ctx)
    r['NetCashFlowsInvestingContinuing'] = resolver.resolve(['NetCashProvidedByUsedInInvestingActivitiesContinuingOperations'], ctx)
    r['NetCashFlowsFinancingContinuing'] = resolver.resolve(['NetCashProvidedByUsedInFinancingActivitiesContinuingOperations'], ctx)
    r['NetCashFlowsOperatingDiscontinued'] = resolver.resolve(['CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations'], ctx)
    r['NetCashFlowsInvestingDiscontinued'] = resolver.resolve(['CashProvidedByUsedInInvestingActivitiesDiscontinuedOperations'], ctx)
    r['NetCashFlowsFinancingDiscontinued'] = resolver.resolve(['CashProvidedByUsedInFinancingActivitiesDiscontinuedOperations'], ctx)
    r['NetCashFlowsDiscontinued'] = resolver.resolve(['NetCashProvidedByUsedInDiscontinuedOperations'], ctx)
    r['NetCashFlowsContinuing'] = resolver.resolve(['NetCashProvidedByUsedInContinuingOperations'], ctx)
    r['ExchangeGainsLosses'] = resolver.resolve([
        'EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations',
        'EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
        'EffectOfExchangeRateOnCashAndCashEquivalents',
        'EffectOfExchangeRateOnCash',
        'EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsContinuingOperations',
        'EffectOfExchangeRateOnCashAndCashEquivalentsContinuingOperations',
        'EffectOfExchangeRateOnCashContinuingOperations',
        'EffectOfExchangeRateOnCashAndCashEquivalentsDiscontinuedOperations'
    ], ctx)

    # If the total came from an Excluding-exchange-rate element, add FX back so
    # NetCashFlow is on the same basis as the CF1/CF2 identities.
    if _ncf_excludes_fx:
        r['NetCashFlow'] = r['NetCashFlow'] + r['ExchangeGainsLosses']

    r = _impute_cf(r)

    checks = {
        'CF1': check_or_na(r['NetCashFlow'] - (r['NetCashFlowsOperating'] + r['NetCashFlowsInvesting'] + r['NetCashFlowsFinancing'] + r['ExchangeGainsLosses']),
                   r['NetCashFlow'], r['NetCashFlowsOperating'], r['NetCashFlowsInvesting'], r['NetCashFlowsFinancing'], r['ExchangeGainsLosses']),
        'CF2': check_or_na(r['NetCashFlow'] - (r['NetCashFlowsContinuing'] + r['NetCashFlowsDiscontinued'] + r['ExchangeGainsLosses']),
                   r['NetCashFlow'], r['NetCashFlowsContinuing'], r['NetCashFlowsDiscontinued'], r['ExchangeGainsLosses']),
        'CF3': check_or_na(r['NetCashFlowsDiscontinued'] - (r['NetCashFlowsOperatingDiscontinued'] + r['NetCashFlowsInvestingDiscontinued'] + r['NetCashFlowsFinancingDiscontinued']),
                   r['NetCashFlowsDiscontinued'], r['NetCashFlowsOperatingDiscontinued'], r['NetCashFlowsInvestingDiscontinued'], r['NetCashFlowsFinancingDiscontinued']),
        'CF4': check_or_na(r['NetCashFlowsOperating'] - (r['NetCashFlowsOperatingContinuing'] + r['NetCashFlowsOperatingDiscontinued']),
                   r['NetCashFlowsOperating'], r['NetCashFlowsOperatingContinuing'], r['NetCashFlowsOperatingDiscontinued']),
        'CF5': check_or_na(r['NetCashFlowsInvesting'] - (r['NetCashFlowsInvestingContinuing'] + r['NetCashFlowsInvestingDiscontinued']),
                   r['NetCashFlowsInvesting'], r['NetCashFlowsInvestingContinuing'], r['NetCashFlowsInvestingDiscontinued']),
        'CF6': check_or_na(r['NetCashFlowsFinancing'] - (r['NetCashFlowsFinancingContinuing'] + r['NetCashFlowsFinancingDiscontinued']),
                   r['NetCashFlowsFinancing'], r['NetCashFlowsFinancingContinuing'], r['NetCashFlowsFinancingDiscontinued']),
    }
    return r, checks


def _impute_cf(r):
    if r['NetCashFlowsOperatingDiscontinued'] == 0 and r['NetCashFlowsInvestingDiscontinued'] == 0 \
            and r['NetCashFlowsFinancingDiscontinued'] == 0 and r['NetCashFlowsDiscontinued'] != 0:
        r['NetCashFlowsOperatingDiscontinued'] = r['NetCashFlowsDiscontinued']
    if r['NetCashFlowsOperatingContinuing'] == 0 and r['NetCashFlowsOperating'] != 0:
        r['NetCashFlowsOperatingContinuing'] = r['NetCashFlowsOperating'] - r['NetCashFlowsOperatingDiscontinued']
    if r['NetCashFlowsInvestingContinuing'] == 0 and r['NetCashFlowsInvesting'] != 0:
        r['NetCashFlowsInvestingContinuing'] = r['NetCashFlowsInvesting'] - r['NetCashFlowsInvestingDiscontinued']
    if r['NetCashFlowsFinancingContinuing'] == 0 and r['NetCashFlowsFinancing'] != 0:
        r['NetCashFlowsFinancingContinuing'] = r['NetCashFlowsFinancing'] - r['NetCashFlowsFinancingDiscontinued']
    if r['NetCashFlowsOperating'] == 0:
        r['NetCashFlowsOperating'] = r['NetCashFlowsOperatingContinuing'] + r['NetCashFlowsOperatingDiscontinued']
    if r['NetCashFlowsInvesting'] == 0:
        r['NetCashFlowsInvesting'] = r['NetCashFlowsInvestingContinuing'] + r['NetCashFlowsInvestingDiscontinued']
    if r['NetCashFlowsFinancing'] == 0:
        r['NetCashFlowsFinancing'] = r['NetCashFlowsFinancingContinuing'] + r['NetCashFlowsFinancingDiscontinued']
    if r['NetCashFlowsDiscontinued'] == 0:
        r['NetCashFlowsDiscontinued'] = (r['NetCashFlowsOperatingDiscontinued']
                                         + r['NetCashFlowsInvestingDiscontinued']
                                         + r['NetCashFlowsFinancingDiscontinued'])
    if r['NetCashFlowsContinuing'] == 0 and r['NetCashFlow'] != 0:
        r['NetCashFlowsContinuing'] = r['NetCashFlow'] - r['NetCashFlowsDiscontinued'] - r['ExchangeGainsLosses']
    if r['NetCashFlow'] == 0 and r['NetCashFlowsContinuing'] != 0:
        r['NetCashFlow'] = r['NetCashFlowsContinuing'] + r['NetCashFlowsDiscontinued'] + r['ExchangeGainsLosses']
    if (r['NetCashFlowsInvestingContinuing'] == 0 and r['NetCashFlowsOperatingContinuing'] != 0
            and r['NetCashFlowsFinancingContinuing'] != 0
            and (r['NetCashFlowsContinuing'] - (r['NetCashFlowsOperatingContinuing']
                 + r['NetCashFlowsInvestingContinuing'] + r['NetCashFlowsFinancingContinuing'])) != 0):
        r['NetCashFlowsInvestingContinuing'] = r['NetCashFlowsContinuing'] - (
            r['NetCashFlowsOperatingContinuing'] + r['NetCashFlowsFinancingContinuing'])
    if (r['NetCashFlowsFinancingContinuing'] == 0 and r['NetCashFlowsOperatingContinuing'] != 0
            and r['NetCashFlowsInvestingContinuing'] != 0
            and (r['NetCashFlowsContinuing'] - (r['NetCashFlowsOperatingContinuing']
                 + r['NetCashFlowsInvestingContinuing'] + r['NetCashFlowsFinancingContinuing'])) != 0):
        r['NetCashFlowsFinancingContinuing'] = r['NetCashFlowsContinuing'] - (
            r['NetCashFlowsOperatingContinuing'] + r['NetCashFlowsInvestingContinuing'])
    if (r['NetCashFlowsInvesting'] == 0 and r['NetCashFlowsOperating'] != 0
            and r['NetCashFlowsFinancing'] != 0
            and (r['NetCashFlow'] - (r['NetCashFlowsOperating'] + r['NetCashFlowsInvesting']
                 + r['NetCashFlowsFinancing'] + r['ExchangeGainsLosses'])) != 0):
        r['NetCashFlowsInvesting'] = r['NetCashFlow'] - (
            r['NetCashFlowsOperating'] + r['NetCashFlowsFinancing'] + r['ExchangeGainsLosses'])
    if (r['NetCashFlowsFinancing'] == 0 and r['NetCashFlowsOperating'] != 0
            and r['NetCashFlowsInvesting'] != 0
            and (r['NetCashFlow'] - (r['NetCashFlowsOperating'] + r['NetCashFlowsInvesting']
                 + r['NetCashFlowsFinancing'] + r['ExchangeGainsLosses'])) != 0):
        r['NetCashFlowsFinancing'] = r['NetCashFlow'] - (
            r['NetCashFlowsOperating'] + r['NetCashFlowsInvesting'] + r['ExchangeGainsLosses'])
    if r['NetCashFlowsContinuing'] == 0:
        r['NetCashFlowsContinuing'] = (r['NetCashFlowsOperatingContinuing']
                                       + r['NetCashFlowsInvestingContinuing']
                                       + r['NetCashFlowsFinancingContinuing'])
    if r['NetCashFlowsOperating'] == 0 and r['NetCashFlowsOperatingContinuing'] != 0 \
            and r['NetCashFlowsOperatingDiscontinued'] == 0:
        r['NetCashFlowsOperating'] = r['NetCashFlowsOperatingContinuing']
    if r['NetCashFlowsInvesting'] == 0 and r['NetCashFlowsInvestingContinuing'] != 0 \
            and r['NetCashFlowsInvestingDiscontinued'] == 0:
        r['NetCashFlowsInvesting'] = r['NetCashFlowsInvestingContinuing']
    if r['NetCashFlowsFinancing'] == 0 and r['NetCashFlowsFinancingContinuing'] != 0 \
            and r['NetCashFlowsFinancingDiscontinued'] == 0:
        r['NetCashFlowsFinancing'] = r['NetCashFlowsFinancingContinuing']
    if r['NetCashFlowsInvestingContinuing'] == 0 and r['NetCashFlowsInvestingDiscontinued'] == 0 \
            and r['NetCashFlowsInvesting'] != 0:
        r['NetCashFlowsInvestingContinuing'] = r['NetCashFlowsInvesting']
    if r['NetCashFlowsFinancingContinuing'] == 0 and r['NetCashFlowsFinancingDiscontinued'] == 0 \
            and r['NetCashFlowsFinancing'] != 0:
        r['NetCashFlowsFinancingContinuing'] = r['NetCashFlowsFinancing']
    if (r['NetCashFlow'] == 0 and r['NetCashFlowsContinuing'] != 0
            and r['NetCashFlowsDiscontinued'] == 0 and r['NetCashFlowsOperatingDiscontinued'] == 0
            and r['NetCashFlowsInvestingDiscontinued'] == 0 and r['NetCashFlowsFinancingDiscontinued'] == 0
            and r['ExchangeGainsLosses'] == 0):
        r['NetCashFlow'] = r['NetCashFlowsContinuing']
    if r['NetCashFlow'] == 0 and r['NetCashFlowsOperating'] != 0 \
            and r['NetCashFlowsInvesting'] != 0 and r['NetCashFlowsFinancing'] != 0:
        r['NetCashFlow'] = (r['NetCashFlowsOperating'] + r['NetCashFlowsInvesting']
                            + r['NetCashFlowsFinancing'] + r['ExchangeGainsLosses'])
    if r['NetCashFlow'] == 0 and r['NetCashFlowsContinuing'] != 0 and r['NetCashFlowsDiscontinued'] != 0:
        r['NetCashFlow'] = r['NetCashFlowsContinuing'] + r['NetCashFlowsDiscontinued'] + r['ExchangeGainsLosses']
    return r


# --------------------------------------------------------------------------- #
# Per-filing extraction + manifest-driven orchestration.
# --------------------------------------------------------------------------- #
def extract_one(cur, accession_id: int) -> dict | None:
    """Extract BS/IS/CF + checks for one accession; None if unparseable (no DPED)."""
    doc_end, instant_ctx, duration_ctx = get_contexts(cur, accession_id)
    if doc_end is None:
        return None
    resolver = _FactResolver(cur, accession_id, [instant_ctx, duration_ctx])
    bs, bs_checks = extract_balance_sheet(resolver, instant_ctx)
    is_, is_checks = extract_income_statement(resolver, duration_ctx)
    cf, cf_checks = extract_cash_flow(resolver, duration_ctx)

    row = {'accession_id': accession_id, 'doc_period_end': doc_end}
    row.update({f'BS_{k}': v for k, v in bs.items()})
    row.update({f'IS_{k}': v for k, v in is_.items()})
    row.update({f'CF_{k}': v for k, v in cf.items()})
    row.update({f'CHK_{k}': v for k, v in bs_checks.items()})
    row.update({f'CHK_{k}': v for k, v in is_checks.items()})
    row.update({f'CHK_{k}': v for k, v in cf_checks.items()})
    return row


_MANIFEST_COLS = ["accession_id", "accession_number", "cik", "company_name",
                  "form_type", "period", "filed_date"]


def fetch_queue(conn, *, since: str = DEFAULT_SINCE, include_partial: bool = False,
                limit: int | None = None) -> list[dict]:
    """Filings ready to extract: current, filed on/after ``since``, cleanly loaded.

    Matches each ``filing_manifest`` row to its ``accession_id`` via the
    accession number embedded in the loaded document's URI (the same join
    ``3ArelleLoad.ipynb`` and ``4Fundamentals_formulas.ipynb`` use, since
    ``accession.filing_accession_number`` holds a load timestamp, not the SEC
    accession number, in this DB). Recomputes the DocumentPeriodEndDate check
    directly rather than trusting ``filing_manifest.load_status`` alone, so a
    stale ledger can't silently admit a partial load.
    """
    load_statuses = ['loaded', 'loaded_partial'] if include_partial else ['loaded']
    limit_clause = "LIMIT %(limit)s" if limit else ""
    sql = r"""
        WITH mapped AS (
            SELECT fm.accession_number, fm.cik, fm.company_name, fm.form_type,
                   fm.period, fm.filed_date, a.accession_id,
                   EXISTS (
                       SELECT 1 FROM fact f
                       JOIN element e ON f.element_id = e.element_id
                       JOIN qname   q ON e.qname_id   = q.qname_id
                       WHERE f.accession_id = a.accession_id
                         AND q.local_name = 'DocumentPeriodEndDate'
                   ) AS has_dped
            FROM filing_manifest fm
            JOIN document d
                ON substring(d.document_uri from '\d{10}-\d{2}-\d{6}') = fm.accession_number
            JOIN accession_document_association ada ON ada.document_id = d.document_id
            JOIN accession a ON a.accession_id = ada.accession_id
            WHERE fm.superseded_by IS NULL
              AND fm.filed_date >= %(since)s
              AND fm.load_status = ANY(%(load_statuses)s)
        )
        SELECT DISTINCT ON (accession_number)
               accession_id, accession_number, cik, company_name, form_type, period, filed_date
        FROM mapped
        ORDER BY accession_number, has_dped DESC, accession_id DESC
    """ + limit_clause
    params: dict = {"since": since, "load_statuses": load_statuses}
    if limit:
        params["limit"] = limit
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(zip(_MANIFEST_COLS, row)) for row in cur.fetchall()]


def _report_check_pass_rates(df: pd.DataFrame, *, tol: float = 1.0, verbose: bool = True) -> None:
    for chk in (c for c in df.columns if c.startswith('CHK_')):
        vals = pd.to_numeric(df[chk], errors='coerce')
        applicable = vals.notna()
        n_app = int(applicable.sum())
        if n_app:
            pct = (vals[applicable].abs() <= tol).mean() * 100
            _log(verbose, f"  {chk}: {pct:5.1f}% pass ({n_app} applicable, "
                          f"{int((~applicable).sum())} N/A)")


def run(*, since: str = DEFAULT_SINCE, include_partial: bool = False,
        limit: int | None = None, out_path: Path | None = None,
        verbose: bool = True) -> pd.DataFrame:
    """Extract point-in-time fundamentals for every matching filing; write a CSV.

    One row per filing: cik/company_name/form_type/accession_number/period/
    filed_date from ``filing_manifest``, plus the BS_*/IS_*/CF_*/CHK_* columns
    from :func:`extract_one`. This is the table :mod:`panel` should join
    against by (cik, filed_date) instead of calling the SEC companyfacts API
    per ticker.
    """
    conn = connect()
    try:
        queue = fetch_queue(conn, since=since, include_partial=include_partial, limit=limit)
        _log(verbose, f"{len(queue)} filings to extract (since {since})")
        rows: list[dict] = []
        skipped: list[tuple] = []
        with conn.cursor() as cur:
            for i, filing in enumerate(queue, 1):
                try:
                    extracted = extract_one(cur, filing["accession_id"])
                except Exception as exc:  # noqa: BLE001 - keep going; one bad filing shouldn't kill the run
                    skipped.append((filing["accession_number"], str(exc)))
                    continue
                if extracted is None:
                    skipped.append((filing["accession_number"], "no DocumentPeriodEndDate"))
                    continue
                row = {
                    "cik": filing["cik"], "company_name": filing["company_name"],
                    "form_type": filing["form_type"], "accession_number": filing["accession_number"],
                    "period": filing["period"], "filed_date": filing["filed_date"],
                }
                row.update(extracted)
                rows.append(row)
                if verbose and i % 500 == 0:
                    print(f"  ...{i}/{len(queue)} extracted")
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    _log(verbose, f"\nExtracted {len(df)} filings, skipped {len(skipped)}")
    if not df.empty:
        _report_check_pass_rates(df, verbose=verbose)

    out_path = Path(out_path) if out_path else (XBRL_DIR / f"fundamentals_{since}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _log(verbose, f"Saved -> {out_path}")
    return df


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
