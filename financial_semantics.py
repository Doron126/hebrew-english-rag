"""Source-local monetary roles and explicitly named document scope.

Financial document identity is a scope constraint. It never supplies an amount,
currency, or payable/subtotal/tax role missing from the bounded source row.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import unicodedata
from typing import Any

from answer_units import GROUPED_DECIMAL_LITERAL_PATTERN

_QUOTE = r'["\u05f4\u201c\u201d\u2033]'
_TOTAL_HE = rf'(?:סה\s*{_QUOTE}?\s*כ|סך\s*(?:הכל|הכול))'
_ROLES = {
    "payable_total": re.compile(
        rf'\b(?:grand\s+total|total\s+(?:amount\s+)?(?:due|payable)|amount\s+due|balance\s+due)\b|'
        rf'(?:{_TOTAL_HE}\s+לתשלום|ה?סכום\s+(?:ה?כולל\s+)?לתשלום)', re.I),
    "subtotal": re.compile(r'\b(?:sub[ -]?total|net\s+amount|total\s+before\s+(?:tax|vat))\b|'
                           rf'(?:{_TOTAL_HE}\s+לפני\s+מע\s*{_QUOTE}?\s*מ|סכום\s+לפני\s+מע\s*{_QUOTE}?\s*מ)', re.I),
    "tax_amount": re.compile(r'\b(?:vat\s+amount|tax\s+amount|amount\s+of\s+(?:vat|tax))\b|'
                             rf'סכום\s+(?:ה?מע\s*{_QUOTE}?\s*מ|ה?מס)', re.I),
    "unit_price": re.compile(r'\bunit\s+price\b|מחיר\s+(?:ליחידה|יחידה)', re.I),
}
_EXCLUDES_TAX = re.compile(r'\b(?:excluding|excl\.?|before|without)\s+(?:vat|tax)\b|'
                           rf'(?:לפני|ללא|לא\s+כולל)\s+מע\s*{_QUOTE}?\s*מ', re.I)
from financial_documents import DOCUMENT_REFERENCE_RE as _DOC, canonical_document_kind
_CURRENCIES = {
    "₪": ("₪", "ILS", "NIS", "shekel", "shekels", "שקל", "שקלים", 'ש"ח', "ש״ח"),
    "$": ("$", "USD", "dollar", "dollars", "דולר", "דולרים"),
    "€": ("€", "EUR", "euro", "euros", "אירו", "יורו"),
    "£": ("£", "GBP", "pound", "pounds"),
}
_CURRENCY = re.compile(r'(?:₪|\$|€|£|\b(?:ILS|NIS|USD|EUR|GBP|shekels?|dollars?|euros?|pounds?)\b|'
                       r'שקלים|שקל|דולרים|דולר|אירו|יורו|ש["״]ח)', re.I)
_MONETARY_NUMBER = re.compile(rf'(?<![\w])(?:{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:[.,]\d+)?)(?![\w])')
_SCOPE_STOP = frozenset({"for", "the", "in", "on", "of", "and", "a", "an", "what", "which", "is", "was", "how", "much",
                         "מה", "מהו", "מהי", "עבור", "של", "את", "בגין"})

# Business dates are independent roles. Filesystem/PDF creation metadata is
# deliberately absent from this parser's interface.
_MONTH_NAMES = (
    ("january", "jan", "ינואר"), ("february", "feb", "פברואר"),
    ("march", "mar", "מרץ", "מארס"), ("april", "apr", "אפריל"),
    ("may", "מאי"), ("june", "jun", "יוני"),
    ("july", "jul", "יולי"), ("august", "aug", "אוגוסט"),
    ("september", "sep", "sept", "ספטמבר"), ("october", "oct", "אוקטובר"),
    ("november", "nov", "נובמבר"), ("december", "dec", "דצמבר"),
)
_MONTH_MAP = {name: number for number, names in enumerate(_MONTH_NAMES, 1) for name in names}
_MONTH_WORD = "|".join(sorted(map(re.escape, _MONTH_MAP), key=len, reverse=True))
_NAMED_MONTH = re.compile(rf'(?<![A-Za-z\u05d0-\u05ea])(?:ב|ל|מ)?(?P<month>{_MONTH_WORD})\.?\s*[,/-]?\s*(?P<year>(?:19|20)\d{{2}})(?!\d)', re.I)
_NUMERIC_MONTH = re.compile(r'(?<![\d/])(?P<month>0?[1-9]|1[0-2])\s*[/.-]\s*(?P<year>(?:19|20)\d{2})(?![\d/])')
_BUSINESS_DATE = re.compile(r'(?<!\d)(?:(?P<day>[0-3]?\d)[/.-](?P<month>[01]?\d)[/.-](?P<year>(?:19|20)\d{2})|(?P<iso_year>(?:19|20)\d{2})-(?P<iso_month>[01]\d)-(?P<iso_day>[0-3]\d))(?!\d)')
_FINANCIAL_TYPE = re.compile(r'(?P<kind>tax\s+invoice\s*/\s*receipt|invoice\s*/\s*receipt|חשבונית\s+מס\s*/?\s*קבלה|\bcredit\s+note\b|חשבונית\s+זיכוי|חשבונית\s+(?:מס|עסקה)|חשבונית|\binvoice\b|\breceipt\b|קבלה)', re.I)
_MONETARY_FINANCIAL_TYPE = re.compile(r'(?P<kind>\b(?:quotation|quote|estimate)\b|הצעת\s+ה?מחיר|tax\s+invoice\s*/\s*receipt|invoice\s*/\s*receipt|חשבונית\s+מס\s*/?\s*קבלה|\bcredit\s+note\b|חשבונית\s+זיכוי|חשבונית\s+(?:מס|עסקה)|חשבונית|\binvoice\b|\breceipt\b|קבלה)', re.I)
_PAYMENT_HEADING = re.compile(r'(?:(?:פרטי|פירוט|פרוט|נתוני)\s+(?:ה?תשלומים|תקבולים|תשלום)|\bpayment\s+(?:details|history|received|date)|\bpayments\b|תאריך\s+(?:ה?תשלום|ה?תקבול)|(?:שולם|התקבל)\s+(?:בתאריך|ביום))', re.I)
_PAYMENT_METHOD = re.compile(r'(?:העברה\s+בנקאית|מזומן|כרטיס\s+אשראי|המחאה|שיק|\bbank\s+transfer\b|\bcash\b|\bcredit\s+card\b|\bche(?:ck|que)\b)', re.I)
_CANCELLED = re.compile(r'(?:\bcancell?ed\b|\bvoid(?:ed)?\b|מבוטל(?:ת)?|בוטל(?:ה)?)', re.I)


@dataclass(frozen=True)
class FinancialQuery:
    operation: str  # payments or documents
    date_basis: str  # payment, issue, service, or service_or_issue
    year: int
    month: int
    document_kind: str = ""

    @property
    def period(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def financial_query(text: str) -> FinancialQuery | None:
    """Interpret monthly financial operations without supplying source facts."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    # A monthly date does not turn a scoped fact or comparison into a ledger
    # operation. Those requests retain the existing fact/comparison path.
    if re.search(r'\b(?:compare|comparison|difference|versus|between)\b|השוו|השווה|השוואה|לעומת|הבדל|בין.+לבין', value, re.I):
        return None
    if re.search(r'\b(?:not\s+(?:received|paid)|unpaid|unreceived)\b|לא\s+(?:התקבל|שולם|שולמ)', value, re.I):
        return None
    if re.search(r'\b(?:vat|tax|subtotal|unit\s+price)\b|מע\s*["״]?\s*מ|לפני\s+מס|מחיר\s+יחידה', value, re.I):
        return None
    if re.search(r'(?:\b(?:invoice|receipt)\b|חשבונית|קבלה)\s*(?:(?:number|no\.?|מספר|מס\.)\s*)?[#:]?\s*\d{2,12}(?!\d)', value, re.I):
        return None
    months = {(int(match.group("year")), _MONTH_MAP[match.group("month").casefold()]) for match in _NAMED_MONTH.finditer(value)}
    months.update((int(match.group("year")), int(match.group("month"))) for match in _NUMERIC_MONTH.finditer(value))
    if len(months) != 1:
        return None
    financial = bool(re.search(r'כסף|כספים|תשלו|תקבול|שולמ|שולם|התקבל|חשבוני|קבלות|\b(?:money|cash|payments?|paid|collected|receiv(?:e|ed)|receipts?|invoices?)\b', value, re.I))
    if not financial:
        return None
    year, month = next(iter(months))
    payment_action = bool(re.search(r'התקבל|שולם|שולמ|נגבה|גבינו|\b(?:receiv(?:e|ed)|paid|collected|came\s+in)\b', value, re.I))
    monetary_measure = bool(re.search(r'כסף|כספים|כמה|סכום|סך|\b(?:how\s+much|money|cash|amount|total|sum)\b', value, re.I))
    explicit_payment_total = bool(re.search(r'(?:סך|סכום|סכם).{0,20}(?:תשלו|תקבול)|\b(?:total|sum)\s+(?:of\s+)?(?:payments?|receipts?)\b', value, re.I))
    payments = payment_action and monetary_measure or explicit_payment_total
    if payments:
        return FinancialQuery("payments", "payment", year, month)
    kind = "invoice" if re.search(r'חשבוני|\binvoices?\b', value) else "receipt" if re.search(r'קבלות|קבלה|\breceipts?\b', value) else ""
    if not kind:
        return None
    enumeration = bool(re.search(r'\b(?:list|show|find|display|enumerate|name|give|which|what\s+(?:are|invoices?|receipts?))\b|(?:הצג|תציג|הראה|מצא|רשום|רשמי|מנה|אילו|איזה|מהן|מהם|תן|תני|מה\s+ה?חשבוניות|מה\s+ה?קבלות)', value, re.I)
                       or re.match(r'\s*(?:invoices?|receipts?|חשבוניות|קבלות)\b', value, re.I))
    scalar_field = bool(re.search(r'\b(?:amount|price|balance|total|how\s+many|date)\b|סכום|מחיר|יתרה|כמה\s+חשבוני|תאריך', value, re.I))
    if not enumeration or scalar_field:
        return None
    basis = ("payment" if re.search(r'שולמ|שולם|התקבל|\b(?:paid|received)\b', value)
             else "issue" if re.search(r'הונפק|הוצא|תאריך\s+ה?חשבונית|\b(?:issued|issuance|dated)\b', value)
             else "service" if re.search(r'פעילות|תקופת|שירות|\b(?:service|activity)\b', value)
             else "service_or_issue")
    return FinancialQuery("documents", basis, year, month, kind)


@dataclass(frozen=True)
class FinancialField:
    value: Any
    quote: str
    span: tuple[int, int]
    page: int | None = None


@dataclass(frozen=True)
class PaymentEvidence:
    amount: Decimal
    currency: str
    date: date
    quote: str
    span: tuple[int, int]
    amount_span: tuple[int, int]
    date_span: tuple[int, int]
    reference: str = ""
    page: int | None = None


@dataclass(frozen=True)
class FinancialRecord:
    source_id: str
    title: str
    tenant_id: str
    content_hash: str
    document_kind: str
    document_number: str
    issuer_identity: str
    issue_date: FinancialField | None
    service_periods: tuple[FinancialField, ...]
    subtotal: FinancialField | None
    vat: FinancialField | None
    total: FinancialField | None
    currency: str
    payments: tuple[PaymentEvidence, ...]
    cancelled: bool = False
    related_identifiers: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


def _date_value(match) -> date | None:
    try:
        return date(int(match.group("year") or match.group("iso_year")), int(match.group("month") or match.group("iso_month")), int(match.group("day") or match.group("iso_day")))
    except (ValueError, TypeError):
        return None


def _decimal_literal(raw: str) -> Decimal | None:
    if re.fullmatch(GROUPED_DECIMAL_LITERAL_PATTERN, raw):
        value = raw.replace(",", "")
    elif re.fullmatch(r'\d+(?:\.\d+)?', raw):
        value = raw
    elif re.fullmatch(r'\d+,\d{1,2}', raw):
        value = raw.replace(",", ".")
    else:
        return None
    try:
        result = Decimal(value)
        return result if result.is_finite() and result >= 0 else None
    except InvalidOperation:
        return None


def _money_matches(text: str):
    output = []
    for match in _MONETARY_NUMBER.finditer(text):
        # Reject a partial match inside an unsupported monetary format rather
        # than silently changing its magnitude.
        if ((match.start() and text[match.start()-1] in ",.")
                or (match.end() < len(text) and text[match.end()] in ",." and match.end()+1 < len(text) and text[match.end()+1].isdigit())
                or re.match(r'\s+\d{3}(?:[.,]\d|(?=\s|$))', text[match.end():])):
            continue
        currency = currency_for_span(text, *match.span())
        amount = _decimal_literal(match.group())
        if currency and amount is not None:
            output.append((match, amount, currency))
    return output


def extract_financial_record(text: str, *, source_id: str, title: str, tenant_id: str,
                             page: int | None = None, content_hash: str = "",
                             include_quotations: bool = False) -> FinancialRecord | None:
    """Read source-local business roles from an authorized, ordered document.

    The caller authorizes and assembles the source. Every extracted field
    retains an exact span in that supplied text. Missing/ambiguous fields stay
    missing; an amount due is never promoted to a received payment.
    """
    value = str(text or "")
    kind_match = (_MONETARY_FINANCIAL_TYPE if include_quotations else _FINANCIAL_TYPE).search(value[:2400])
    if not kind_match:
        return None
    raw_kind = kind_match.group()
    kind = "invoice_receipt" if re.search(r'חשבונית.*קבלה|invoice.*receipt', raw_kind, re.I) else "credit" if re.search(r'זיכוי|credit', raw_kind, re.I) else "invoice" if re.search(r'חשבונית|invoice', raw_kind, re.I) else "quotation" if re.search(r"quote|quotation|estimate|הצעת", raw_kind, re.I) else "receipt"
    if kind == "quotation" and not include_quotations:
        return None
    identifier = re.match(r'\s*(?:(?:no\.?|number|מספר|מס\.)\s*)?[#:]?\s*(\d{2,12})(?!\d)', value[kind_match.end():kind_match.end() + 65], re.I)
    number = identifier.group(1) if identifier else ""
    issuer = re.search(r'(?:עוסק\s+מורשה|מספר\s+עוסק|\b(?:tax|vat)\s*(?:id|number))\s*[:#]?\s*(\d{6,12})', value[:kind_match.start()], re.I)
    issuer_id = issuer.group(1) if issuer else ""
    payment_heading = _PAYMENT_HEADING.search(value)
    prepayment = value[:payment_heading.start()] if payment_heading else value
    dates = [(match, _date_value(match)) for match in _BUSINESS_DATE.finditer(prepayment)]
    dates = [(match, parsed) for match, parsed in dates if parsed is not None]
    labeled = [(match, parsed) for match, parsed in dates
               if re.search(r'(?:issue\s+date|invoice\s+date|date\s+issued|תאריך\s+(?:הנפקה|חשבונית)|^תאריך)\s*[:\s]*$', value[max(0, match.start()-50):match.start()], re.I | re.M)]
    header = [(match, parsed) for match, parsed in dates
              if kind_match.start()-100 <= match.start() <= kind_match.end()+100]
    issue_candidates = labeled or header
    unique_dates = {parsed for _match, parsed in issue_candidates}
    issue = None
    if len(unique_dates) == 1:
        match, parsed = issue_candidates[0]
        issue = FinancialField(parsed, match.group(), match.span(), page)
    service = []
    for match in _NAMED_MONTH.finditer(prepayment):
        line_start = value.rfind("\n", 0, match.start()) + 1
        context = value[max(0, line_start-80):match.end()]
        if re.search(r'פעילות|שירות|תקופ|פירוט|תיאור|עבור|\b(?:service|activity|description|period|for)\b', context, re.I):
            line_end = value.find("\n", match.end())
            line_end = len(value) if line_end < 0 else line_end
            service.append(FinancialField((int(match.group("year")), _MONTH_MAP[match.group("month").casefold()]), value[line_start:line_end], (line_start, line_end), page))
    money = _money_matches(value)
    money_starts = [match.start() for match, _amount, _currency in money]
    fields: dict[str, list[FinancialField]] = {"subtotal": [], "tax_amount": [], "payable_total": []}
    currencies = set()
    for match, amount, currency in money:
        if payment_heading and match.start() >= payment_heading.start():
            continue
        currencies.add(currency)
        left = max(value.rfind("\n", 0, match.start()) + 1, match.start()-70)
        right = value.find("\n", match.end())
        right = min(len(value), match.end()+70) if right == -1 else right
        local = value[left:right]
        role = monetary_role(local)
        if not role and re.search(r'מע\s*["״]?\s*מ|\bVAT\b', local, re.I):
            role = "tax_amount"
        if role in fields:
            fields[role].append(FinancialField(amount, local, (left, right), page))
        elif not local.replace(match.group(), "").strip(" ₪$€£\t"):
            # Positioned PDF extraction may place the VAT label on the line
            # immediately after its amount. Preserve that adjacent source
            # span; never borrow a distant document-wide role.
            following = re.match(r'\s*מע\s*["״]?\s*מ\s*\d+(?:[.,]\d+)?\s*%', value[match.end():match.end()+45], re.I)
            if following:
                fields["tax_amount"].append(FinancialField(amount, value[left:match.end()+following.end()], (left, match.end()+following.end()), page))
        if not role and re.search(rf'{_TOTAL_HE}\s*$', _CURRENCY.sub("", value[left:match.start()]), re.I):
            following = value[match.end():min(len(value), match.end()+100)]
            if re.search(r'מע\s*["״]?\s*מ\s*\d+(?:[.,]\d+)?\s*%', following, re.I):
                fields["subtotal"].append(FinancialField(amount, local, (left, right), page))
    def unambiguous(role):
        candidates = fields[role]
        return candidates[0] if len({item.value for item in candidates}) == 1 else None
    payments = []
    diagnostics = []
    if payment_heading:
        start = payment_heading.end()
        section = value[start:]
        section_dates = list(_BUSINESS_DATE.finditer(section))
        for date_index, match in enumerate(section_dates):
            parsed = _date_value(match)
            if parsed is None:
                continue
            absolute_start, absolute_end = start+match.start(), start+match.end()
            # A payment section owns the row, while a method or explicitly
            # labelled payment date guards against footer/print dates.
            previous_date_end = start+section_dates[date_index-1].end() if date_index else start
            next_date_start = start+section_dates[date_index+1].start() if date_index+1 < len(section_dates) else len(value)
            window_start, window_end = max(start, previous_date_end, absolute_start-240), min(len(value), next_date_start, absolute_end+100)
            window = value[window_start:window_end]
            if not (_PAYMENT_METHOD.search(window) or _PAYMENT_HEADING.search(value[max(start-60, absolute_start-60):absolute_start])):
                continue
            before = [(m,a,c) for m,a,c in money[bisect_left(money_starts, window_start):bisect_left(money_starts, absolute_start)]
                      if not re.search(rf'{_TOTAL_HE}|total|balance|due', value[max(window_start, value.rfind("\n", window_start, m.start())+1):m.start()], re.I)]
            after = [(m,a,c) for m,a,c in money[bisect_left(money_starts, absolute_end):bisect_left(money_starts, window_end)]
                     if not re.search(rf'{_TOTAL_HE}|total|balance|due', value[max(absolute_end, value.rfind("\n", absolute_end, m.start())+1):m.start()], re.I)]
            candidates = before or after
            if len(candidates) > 1:
                diagnostics.append("payment_row_amount_ambiguous")
                continue
            if not candidates:
                diagnostics.append("payment_date_without_supported_amount")
                continue
            amount_match, amount, currency = candidates[0]
            methods = list(_PAYMENT_METHOD.finditer(window))
            method = min(methods, key=lambda found: abs((window_start+found.start())-absolute_start)) if methods else None
            reference_window = window[method.end():].split("\n", 1)[0] if method else window
            reference_match = re.search(r'(?:reference|transaction|אסמכתא)\s*[:#]?\s*([A-Za-z0-9-]{3,})', reference_window, re.I)
            reference = reference_match.group(1) if reference_match else ""
            currency_prefixes = [m for m in _CURRENCY.finditer(value[max(0, amount_match.start()-16):amount_match.start()]) if not value[max(0, amount_match.start()-16)+m.end():amount_match.start()].strip()]
            amount_start = max(0, amount_match.start()-16)+currency_prefixes[-1].start() if currency_prefixes else amount_match.start()
            row_start, row_end = min(amount_start, absolute_start), max(amount_match.end(), absolute_end)
            if method:
                row_end = max(row_end, window_start+method.end())
            quote = value[row_start:row_end]
            payment = PaymentEvidence(amount, currency, parsed, quote, (row_start, row_end), amount_match.span(), (absolute_start, absolute_end), reference, page)
            payments.append(payment)
    if kind in {"receipt", "invoice_receipt"} and not payments:
        diagnostics.append("receipt_payment_fields_incomplete")
    if len(currencies) > 1:
        diagnostics.append("multiple_document_currencies")
    if not issue:
        diagnostics.append("issue_date_unproven")
    related = tuple(dict.fromkeys(re.findall(r'(?:for\s+invoice|invoice\s+reference|עבור\s+חשבונית|מבטל(?:ת)?\s+(?:את\s+)?חשבונית)\s*(?:מס\s*)?[/#:]?\s*(\d{2,12})', value, re.I)))
    return FinancialRecord(source_id, title, tenant_id, content_hash or hashlib.sha256(value.encode()).hexdigest(), kind, number, issuer_id, issue, tuple(service), unambiguous("subtotal"), unambiguous("tax_amount"), unambiguous("payable_total"), next(iter(currencies)) if len(currencies)==1 else "", tuple(payments), bool(_CANCELLED.search(value)), related, tuple(dict.fromkeys(diagnostics)))


@dataclass(frozen=True)
class PaymentAggregation:
    totals: tuple[tuple[str, Decimal], ...]
    included: tuple[tuple[FinancialRecord, PaymentEvidence], ...]
    duplicate_count: int
    unsupported_records: tuple[str, ...]
    excluded_cross_tenant: int


def aggregate_payments(records, query: FinancialQuery, *, tenant_id: str) -> PaymentAggregation:
    """Sum supported payment rows once using Decimal and explicit identities."""
    totals: dict[str, Decimal] = {}
    included = []
    unsupported = []
    seen: dict[tuple, tuple[str, int]] = {}
    duplicates = 0
    cross_tenant = 0
    records = tuple(records)
    aliases = {}
    for record in records:
        if record.tenant_id == tenant_id and record.issuer_identity and record.document_number:
            own = (record.issuer_identity, record.document_number)
            aliases.setdefault(own, set()).add(record.document_number)
            aliases[own].update(record.related_identifiers)
    for record in records:
        if record.tenant_id != tenant_id:
            cross_tenant += 1
            continue
        if record.cancelled or record.document_kind == "credit":
            unsupported.append(record.source_id)
            continue
        if not record.payments:
            unsupported.append(record.source_id)
        for payment_index, payment in enumerate(record.payments):
            if (payment.date.year, payment.date.month) != (query.year, query.month):
                continue
            # A transaction reference is issuer-owned. A byte-identical
            # alternate file is safe to merge even without such a reference.
            if payment.reference and record.issuer_identity:
                identity = ("transaction", record.issuer_identity, payment.reference, payment.date, payment.amount, payment.currency)
            elif record.issuer_identity and record.document_number:
                related = {record.document_number, *record.related_identifiers}
                for (issuer, identifier), values in aliases.items():
                    if issuer == record.issuer_identity and record.document_number in values:
                        related.update(values)
                identity = ("record", record.issuer_identity, min(related), payment_index, payment.date, payment.amount, payment.currency)
            else:
                identity = ("content", record.content_hash, payment_index, payment.date, payment.amount, payment.currency, payment.reference, payment.quote)
            # Repeated equal transactions within one document remain distinct
            # rows. A proven identity may merge an alternate document only.
            if identity in seen and (seen[identity][0] != record.source_id or seen[identity][1] == payment_index):
                duplicates += 1
                continue
            seen[identity] = (record.source_id, payment_index)
            totals[payment.currency] = totals.get(payment.currency, Decimal("0")) + payment.amount
            included.append((record, payment))
    return PaymentAggregation(tuple(sorted(totals.items())), tuple(included), duplicates, tuple(dict.fromkeys(unsupported)), cross_tenant)


def financial_record_matches(record: FinancialRecord, query: FinancialQuery) -> tuple[str, ...]:
    """Return the proven matching temporal roles, preserving ambiguity."""
    if query.document_kind and query.document_kind not in record.document_kind:
        return ()
    wanted = (query.year, query.month)
    roles = []
    if record.issue_date and (record.issue_date.value.year, record.issue_date.value.month) == wanted:
        roles.append("issue")
    if any(field.value == wanted for field in record.service_periods):
        roles.append("service")
    if any((payment.date.year, payment.date.month) == wanted for payment in record.payments):
        roles.append("payment")
    return tuple(role for role in roles if role in ({"service", "issue"} if query.date_basis == "service_or_issue" else {query.date_basis}))


def normalize_currency(unit: str) -> str:
    value = unicodedata.normalize("NFKC", str(unit or "")).strip().casefold()
    return next((canonical for canonical, aliases in _CURRENCIES.items()
                 if value in {alias.casefold() for alias in aliases}), "")


@dataclass(frozen=True)
class FinancialDocumentReference:
    kind: str
    number: str
    span: tuple[int, int]
    number_span: tuple[int, int]


def document_references(text: str) -> tuple[FinancialDocumentReference, ...]:
    """Expose existing document identity without requiring monetary intent.

    Kind uses the financial parser's canonical invoice/quote vocabulary.
    Number and both spans retain the supplied text, including leading zeros.
    """
    return tuple(FinancialDocumentReference(
        canonical_document_kind(match.group("kind")),
        match.group("number"), match.span(), match.span("number"))
        for match in _DOC.finditer(str(text or "")))


def document_reference_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(reference.number_span for reference in document_references(text))


def currency_for_span(text: str, start: int, end: int) -> str:
    """Return only a literal currency immediately adjoining this scalar."""
    before = text[max(0, start - 16):start]
    after = text[end:end + 16]
    preceding = [m for m in _CURRENCY.finditer(before) if not before[m.end():].strip()]
    following = [m for m in _CURRENCY.finditer(after) if not after[:m.start()].strip()]
    currencies = {normalize_currency(m.group()) for m in preceding + following}
    currencies.discard("")
    return next(iter(currencies)) if len(currencies) == 1 else ""


def monetary_role(text: str) -> str:
    """Classify a local label without consulting a question or document title."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    roles = {role for role, pattern in _ROLES.items() if pattern.search(value)}
    if len(roles) != 1:
        return ""
    role = next(iter(roles))
    if role == "payable_total" and _EXCLUDES_TAX.search(value):
        return ""
    return role


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', value)
                 if word.casefold() not in _SCOPE_STOP)


def _variants(word: str) -> set[str]:
    return {word, word[1:]} if len(word) >= 4 and re.fullmatch(r'[\u05d0-\u05ea]+', word) and word[0] in "ובכלמהש" else {word}


_SCOPE_ROLES = {"project": "project", "פרויקט": "project", "customer": "customer",
                "client": "customer", "לקוח": "customer", "לקוחה": "customer"}


def _scope_role(word: str) -> str:
    return next((_SCOPE_ROLES[variant] for variant in _variants(word) if variant in _SCOPE_ROLES), "")


def _scope_groups(value: str):
    """Retain role-owned contiguous names; independent groups may reorder."""
    groups = []
    role, words = "service", []
    def flush():
        if words:
            groups.append((role, tuple(words)))
            words.clear()
            return True
        return role == "service"
    for word in re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', value.casefold()):
        new_role = _scope_role(word)
        if new_role or word in {"for", "עבור", "בגין"}:
            if not flush():
                return None
            role = new_role or "service"
        elif word not in _SCOPE_STOP:
            words.append(word)
    return tuple(groups) if flush() else None


@dataclass(frozen=True)
class FinancialRequest:
    role: str
    document_kind: str
    document_number: str
    scope_terms: tuple[str, ...]
    currency: str = ""
    scope_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def scope_json(self):
        return json.dumps({"kind": self.document_kind, "number": self.document_number,
                           "terms": self.scope_terms, "groups": self.scope_groups}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def financial_request(text: str) -> FinancialRequest | None:
    matches = list(_DOC.finditer(text))
    if len(matches) != 1:
        return None
    document = matches[0]
    # A document type ("price quotation") identifies the document. It must
    # not replace the requested payable/subtotal/tax metric.
    role = monetary_role(text[:document.start()])
    if not role:
        return None
    currencies = {normalize_currency(m.group()) for m in _CURRENCY.finditer(text[:document.start()])}
    currencies.discard("")
    if len(currencies) > 1:
        return None
    kind = canonical_document_kind(document.group("kind"))
    residual_prefix = text[:document.start()]
    for pattern in _ROLES.values():
        residual_prefix = pattern.sub(" ", residual_prefix)
    residual_prefix = re.sub(r'\b(?:including|incl\.?)\s+(?:vat|tax)\b|'
                             rf'כולל\s+מע\s*{_QUOTE}?\s*מ', " ", residual_prefix, flags=re.I)
    residual_prefix = _CURRENCY.sub(" ", residual_prefix)
    # Preserve qualifiers on either side of the document reference. Unknown
    # customer/project/service wording cannot disappear into an answer label.
    scope_terms = tuple(dict.fromkeys(_tokens(residual_prefix) + _tokens(text[document.end():])))
    prefix_groups, suffix_groups = _scope_groups(residual_prefix), _scope_groups(text[document.end():])
    if prefix_groups is None or suffix_groups is None:
        return None
    groups = prefix_groups + suffix_groups
    named_roles = [role for role, _words in groups if role != "service"]
    if len(named_roles) != len(set(named_roles)):
        return None
    return FinancialRequest(role, kind, document.group("number"), scope_terms,
                            next(iter(currencies), ""), groups)



@dataclass(frozen=True)
class MonetaryQuestion:
    """A descriptive price request, distinct from a monthly ledger query.

    These are request roles only. No field supplies financial source facts or
    turns a proposed price into evidence that payment occurred.
    """
    operation: str
    description: str
    terms: tuple[str, ...]
    role_requested: str = ""
    explicit_source: object | None = None
    negative: bool = False
    year: int = 0
    month: int = 0
    date_basis: str = ""
    document_kind: str = ""
    requested_fields: tuple[str, ...] = ()
    payment_method: str = ""
    unsupported_payment_method_filter: bool = False

    @property
    def period(self) -> str:
        return ""


_MONETARY_QUESTION = re.compile(
    r'\b(?:price|cost|amount|fee|payment|paid|received|unpaid|payable|subtotal|vat|tax)\b|'
    r'(?<!\w)(?:ה?(?:סכום|מחיר|עלות|תשלום)|לתשלום|שולם|שולמו|שילם|שילמו|'
    r'שילמנו|שילמתי|שילמתם|התקבל|התקבלו|נגבה|נגבו|תומחר|תומחרה|תומחרו|יתרה|מע\s*["״]?\s*מ)(?!\w)|'
    r'(?<!\w)כמה\s+(?:עלה|עלתה|עלו|עולה|עולות|עולים)(?!\w)|'
    # Singular past-tense ellipsis ("how much was the repair/technician")
    # asks for a price. Plural counts and explicit count labels remain counts.
    r'(?<!\w)כמה\s+(?:היה|הייתה)\s+ה?(?:טכנאי|טכנאית|טיפול|תיקון|התקנה|שירות|עבודה|ביקור)(?!\w)', re.I)
_MONETARY_ACTUAL_PAYMENT = re.compile(
    r'\b(?:paid|received|collected|unpaid)\b|\b(?:actual\s+payment|'
    r'payment\s+(?:actually|occurred|made|received|recorded)|(?:recorded|documented)\s+payment|evidence\s+of\s+payment)\b|'
    r'(?<!\w)(?:ו?ש?)(?:שולם|שולמו|שילם|שילמו|שילמנו|שילמתי|שילמתם|'
    r'התקבל|התקבלו|נגבה|נגבו)(?!\w)|'
    r'(?:האם|כבר|בפועל|באמת|בוצע|בוצעה|התבצע|התבצעה).{0,14}תשלום|'
    r'תשלום.{0,14}(?:בפועל|באמת|בוצע|התבצע)', re.I)
_MONETARY_COMPARISON = re.compile(
    r'\b(?:compare|comparison|versus|difference|cheaper|more\s+expensive)\b|'
    r'(?:השוו|השווה|השוואה|לעומת|הבדל|זול\s+יותר|יקר\s+יותר|בין.+לבין)', re.I)
_MONETARY_PAYMENT_WORDS = frozenset({
    'שולם', 'שולמו', 'שילם', 'שילמו', 'שילמנו', 'שילמתי', 'שילמתם',
    'התקבל', 'התקבלו', 'נגבה', 'נגבו',
})
_MONETARY_GENERIC_WORDS = frozenset({
    'what', 'how', 'much', 'was', 'were', 'is', 'are', 'did', 'does', 'do',
    'has', 'have', 'been', 'there', 'it', 'the', 'a', 'an', 'for', 'of', 'on',
    'at', 'to', 'in', 'from', 'according', 'please', 'actually', 'already',
    'evidence', 'payment', 'made', 'price', 'cost', 'amount', 'fee', 'paid',
    'received', 'collected', 'unpaid', 'not', 'yet', 'confirmed', 'actual', 'recorded', 'documented',
    'מה', 'מהו', 'מהי', 'כמה', 'איך', 'כיצד', 'היה', 'הייתה', 'היו', 'האם', 'יש', 'הייתה',
    'את', 'של', 'על', 'עבור', 'בגין', 'לפי', 'במסמך', 'בחשבונית', 'בהצעה',
    'הסכום', 'סכום', 'כסף', 'כספים', 'הכסף', 'הכספים', 'המחיר', 'מחיר', 'העלות', 'עלות', 'התשלום', 'תשלום',
    'תומחר', 'תומחרה', 'תומחרו',
    'לתשלום', 'הכולל', 'כולל', 'הכוללת', 'הכוללים', 'באמת', 'בפועל', 'כבר',
    'שולם', 'שולמו', 'שילם', 'שילמו', 'שילמנו', 'שילמתי', 'שילמתם',
    'התקבל', 'התקבלו', 'נגבה', 'נגבו', 'בוצע', 'בוצעה', 'התבצע', 'התבצעה',
    'לא', 'שלא', 'טרם', 'נא', 'לי', 'עדות', 'הוכחה', 'ה', 'אותו', 'אותה', 'הזאת',
})
_PAYMENT_ATTRIBUTE_WORDS = frozenset({
    'when','which','date','method','way','this','that',
    'באיזה','באיזו','איזה','איזו','מתי','תאריך','דרך','אמצעי','זה','הזה','זאת','הזאת',
})


def canonical_payment_method(value: str) -> str:
    # Map only complete recognized source method labels to request roles.
    value=' '.join(unicodedata.normalize('NFKC',str(value or '')).casefold().split())
    if not _PAYMENT_METHOD.fullmatch(value):
        return ''
    return {
        'העברה בנקאית':'bank_transfer', 'bank transfer':'bank_transfer',
        'מזומן':'cash', 'cash':'cash', 'כרטיס אשראי':'card', 'credit card':'card',
        'המחאה':'check', 'שיק':'check', 'check':'check', 'cheque':'check',
    }.get(value,'')


_PAYMENT_METHOD_REQUEST = re.compile(
    r'(?<!\w)(?:by|via|through|in)\s+(?P<english>bank\s+transfer|cash|credit\s+card|che(?:ck|que))\b|'
    r'(?<!\w)(?:ב|באמצעות\s+)(?P<hebrew>העברה\s+בנקאית|מזומן|כרטיס\s+אשראי|המחאה|שיק)(?!\w)',re.I)


def monetary_question(text: str, *, operation_override=None, description_override=None) -> MonetaryQuestion | None:
    """Recognize a price/payment question whose owner is described in words.

    Named numeric field lookups and monthly operations retain their existing
    interpreters. A bare payment noun is an ordinary price request; a completed
    payment verb or an explicit occurrence question requires payment evidence.
    """
    value = unicodedata.normalize('NFKC', str(text or '')).strip()
    if not value or (not _MONETARY_QUESTION.search(value) and operation_override is None):
        return None
    if operation_override not in (None, 'payment_status', 'description_price'):
        return None
    if (_MONETARY_COMPARISON.search(value)
            or _NAMED_MONTH.search(value) or _NUMERIC_MONTH.search(value)
            or re.match(r'\s*(?:summari[sz]e|describe|explain|סכם|סכמי|סכמו|תאר|תארי|תארו|הסבר)(?!\w)', value, re.I)):
        return None
    actual = (operation_override == 'payment_status' if operation_override is not None
              else bool(_MONETARY_ACTUAL_PAYMENT.search(value)))
    if not actual and financial_request(value) is not None:
        return None
    references = document_references(value)
    if len(references) > 1:
        return None
    source = references[0] if references else None
    description = description_override if description_override is not None else value
    requested_kind = ''
    if source:
        a, b = source.span
        description = description[:a] + ' ' + description[b:]
    else:
        # A declared filename remains a source selector, never a content term.
        # The ordinary request plan still owns selection and authorization.
        from document_query import named_document_reference
        named = named_document_reference(value)
        if named and named.kind != 'ambiguous':
            title = named.title
            if re.search(r'\.(?:pdf|docx?|pptx?|xlsx?|csv|tsv|txt|md|odt|ods|odp)$', title, re.I):
                literal = re.search(re.escape(title), description, re.I)
                if literal:
                    prefix = description[:literal.start()]
                    prefix = re.sub(r'(?:(?:in|from|according\s+to)\s+)?(?:the\s+)?'
                        r'(?:document|file|presentation|workbook|spreadsheet)\s*["“״\']?\s*$',
                        ' ', prefix, flags=re.I)
                    description = prefix + ' ' + description[literal.end():]
                    source = named
        if source is None:
            # The document type is a constraint on the source, not a word
            # that must appear in the description of the work being priced.
            # Resolve filenames first: a type word in a filename is identity.
            kind_match = _MONETARY_FINANCIAL_TYPE.search(description)
            if kind_match:
                from financial_documents import canonical_document_kind
                requested_kind = canonical_document_kind(kind_match.group('kind'))
                description = description[:kind_match.start()] + ' ' + description[kind_match.end():]
    # A specified payment method constrains the authenticated payment row;
    # it is not part of the described service or project.
    method=''
    unsupported_method=False
    if actual:
        methods=list(_PAYMENT_METHOD_REQUEST.finditer(description))
        all_methods=list(_PAYMENT_METHOD.finditer(description))
        unsupported_method=bool(all_methods and (len(all_methods)!=1 or len(methods)!=1))
        if len(methods)==1:
            match=methods[0]
            before=description[max(0,match.start()-24):match.start()]
            if re.search(r'(?:\b(?:not|without|except)\s*|(?:לא|ללא|שלא|למעט)\s*)$',before,re.I):
                unsupported_method=True
            elif not unsupported_method:
                method=canonical_payment_method(match.group('english') or match.group('hebrew'))
                if method:
                    description=description[:match.start()]+' '+description[match.end():]
    # Requested companion fields are output slots, not work/source discovery
    # anchors. Remove source identities first so a filename is never mistaken
    # for a requested field. Values remain authenticated source facts.
    from monetary_presentation import requested_monetary_fields, without_requested_fields
    requested_fields = requested_monetary_fields(description)
    description = without_requested_fields(description)
    role = monetary_role(value)
    if not role:
        role_patterns = {
            'subtotal': r'\b(?:before|excluding|without)\s+(?:vat|tax)\b|(?:לפני|ללא|לא\s+כולל)\s+מע\s*["״]?\s*מ',
            'payable_total': r'\b(?:including|with)\s+(?:vat|tax)\b|כולל\s+מע\s*["״]?\s*מ',
            'tax_amount': r'\b(?:vat|tax)\b|מע\s*["״]?\s*מ',
        }
        roles = {key for key, pattern in role_patterns.items() if re.search(pattern, value, re.I)}
        # A VAT qualifier determines a base/total role, rather than requesting
        # the tax amount itself. Requests for both retain the full price view.
        if 'subtotal' in roles or 'payable_total' in roles:
            roles.discard('tax_amount')
        if len(roles) == 1:
            role = next(iter(roles))
    # Remove only recognized metric phrases; arbitrary service/project words
    # and numeric product identities remain content anchors.
    phrases = [pattern.pattern for pattern in _ROLES.values()]
    phrases += [
        r'\b(?:including|excluding|before|without|with)\s+(?:vat|tax)\b',
        r'(?:לפני|ללא|לא\s+כולל|כולל)\s+מע\s*["״]?\s*מ',
        r'מע\s*["״]?\s*מ', r'\b(?:vat|tax|subtotal|payable)\b',
        r'(?<!\w)כמה\s+(?:עלה|עלתה|עלו|עולה|עולות|עולים)(?!\w)',
    ]
    for pattern in phrases:
        description = re.sub(pattern, ' ', description, flags=re.I)
    words = re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', description)
    words = [word for word in words
             if word.casefold() not in {'and', 'also'}
             if word.casefold() not in _MONETARY_GENERIC_WORDS
             and not (word.startswith('ו') and word[1:].casefold() in _MONETARY_GENERIC_WORDS)
             and not any(word.startswith(prefix) and word[len(prefix):] in _MONETARY_PAYMENT_WORDS
                         for prefix in ('ש', 'וש'))]
    if actual:
        words = [word for word in words if word.casefold() not in _PAYMENT_ATTRIBUTE_WORDS
                 and not (word.startswith('ו') and word[1:].casefold() in _PAYMENT_ATTRIBUTE_WORDS)]
    if (not words or not any(re.search(r'[A-Za-z\u05d0-\u05ea]', word) for word in words)) and source is None:
        return None
    description = ' '.join(words)
    terms = tuple(dict.fromkeys(word.casefold() for word in words))
    negative = bool(re.search(r'\b(?:not\s+(?:yet\s+)?paid|unpaid|not\s+received)\b|(?:לא|שלא|טרם)\s+(?:שולם|שולמו|התקבל|התקבלו)', value, re.I))
    quote_requested = bool(re.search(
        r'(?i)\b(?:quoted|proposed|offered)\s+(?:price|cost|amount)\b|'
        r'\b(?:price|cost|amount)\s+(?:quoted|proposed|offered)\b|'
        r'(?:ה?מחיר|ה?עלות|ה?סכום)\s+(?:ה?מוצע(?:ת)?|שהוצע(?:ה)?)', value))
    return MonetaryQuestion('payment_status' if actual else 'description_price',
                            description, terms, role, source, negative,
                            document_kind='quotation' if quote_requested else requested_kind,
                            requested_fields=requested_fields, payment_method=method,
                            unsupported_payment_method_filter=unsupported_method)


def validated_semantic_monetary_question(query, operation, subject):
    """Validate a model's non-factual operation and complete user-owned topic."""
    from subject_intent import _has_unrepresented_constraints, _independently_delimited_subject
    if operation not in {'payment_status', 'description_price'} or not isinstance(subject, str):
        return None
    subject = unicodedata.normalize('NFKC', subject).strip(' ?!.;,')
    delimited = _independently_delimited_subject(query)
    if not delimited:
        # An attached preposition may introduce the complete work subject.
        # Only a recognized work noun starts this alternative; all remaining
        # terms stay in the subject and explicit quotation scope remains bound.
        from subject_intent import _WORK_TERM
        value = unicodedata.normalize('NFKC', str(query or '')).strip(' ?!.;,')
        work = _WORK_TERM.search(value)
        if work and work.group().startswith('ל'):
            tail = value[work.start():].strip(' ?!.;,')
            candidates = (tail, tail[1:])
            if any(subject.casefold() == candidate.casefold() for candidate in candidates):
                prior = monetary_question(query)
                from monetary_questions import _variants
                proposed_terms = monetary_question(
                    query, operation_override=operation, description_override=subject)
                if prior and proposed_terms:
                    retained = [_variants(word) for word in proposed_terms.terms]
                    omitted = [word for word in prior.terms if not any(
                        _variants(word) & forms for forms in retained)]
                    if all(re.fullmatch(r'ה?מוצע(?:ת)?|שהוצע(?:ה)?', word) for word in omitted):
                        delimited = subject
    if (not subject or subject.casefold() != delimited.casefold()
            or _has_unrepresented_constraints(query, allow_monetary=True)
            or _MONETARY_COMPARISON.search(query)):
        return None
    prior = monetary_question(query)
    if prior and prior.operation == 'payment_status' and operation != prior.operation:
        return None
    if (prior and prior.operation == 'description_price' and operation == 'payment_status'
            and re.search(r'(?i)\b(?:price|cost|quotation|quote)\b|'
                          r'(?<!\w)(?:ה?מחיר|ה?עלות|הצעת\s+מחיר)(?!\w)', query)):
        return None
    proposed = monetary_question(query, operation_override=operation, description_override=subject)
    if prior and proposed:
        from dataclasses import replace
        # Semantic interpretation can supply the operation and the topic,
        # but must retain all deterministic monetary roles and constraints.
        if replace(proposed, operation=prior.operation,
                   description=prior.description, terms=prior.terms) != prior:
            return None
    return proposed


def contextual_monetary_query(current: str, previous: str) -> str | None:
    """Bind a genuine financial follow-up to the prior user-side owner.

    Called only after conversation resolution has retained the source. Past
    answer values never enter the request, and independent questions do not
    pass this contextual boundary.
    """
    if not _MONETARY_QUESTION.search(current):
        return None
    from financial_documents import explicit_document_request
    reference = explicit_document_request(previous)
    prior = monetary_question(previous)
    if reference is not None:
        owner = previous[reference.span[0]:reference.span[1]]
    elif prior is not None:
        owner = prior.description
        if prior.explicit_source is not None:
            refs = document_references(previous)
            if len(refs) == 1:
                owner += ' ' + previous[refs[0].span[0]:refs[0].span[1]]
    else:
        return None
    query = current + ' ' + owner
    return query if monetary_question(query) is not None else None

def document_scope_matches(required_scope: str, source_label: str) -> bool:
    """Match every explicit document qualifier against its proven source label."""
    try:
        scope = json.loads(required_scope)
    except (TypeError, ValueError):
        return False
    if (not isinstance(scope, dict) or set(scope) != {"kind", "number", "terms", "groups"}
            or scope["kind"] not in {"invoice", "quote", "receipt", "invoice_receipt", "credit"} or not isinstance(scope["number"], str)
            or not isinstance(scope["terms"], list) or not isinstance(scope["groups"], list)):
        return False
    label = unicodedata.normalize("NFKC", str(source_label or ""))
    label = label.replace("\\", "/").rsplit("/", 1)[-1]
    if not re.search(r'(?<![0-9])' + re.escape(scope["number"]) + r'(?![0-9])', label):
        return False
    from financial_documents import role_matches
    if not role_matches(canonical_document_kind(label), scope["kind"]):
        return False
    source_words = set(_tokens(label))
    if not all(isinstance(term, str) and (term in source_words or (
            _scope_role(term) and any(_scope_role(term) == _scope_role(observed) for observed in source_words)))
               for term in scope["terms"]):
        return False
    ordered = _tokens(label)
    for group in scope["groups"]:
        if (not isinstance(group, list) or len(group) != 2 or group[0] not in {"project", "customer", "service"}
                or not isinstance(group[1], list) or not group[1] or not all(isinstance(word, str) for word in group[1])):
            return False
        role, phrase = group
        starts = range(len(ordered)) if role == "service" else [index + 1 for index, word in enumerate(ordered) if _scope_role(word) == role]
        if not any(len(ordered[start:start + len(phrase)]) == len(phrase) and all(
                required == observed for required, observed in zip(phrase, ordered[start:start + len(phrase)]))
                   for start in starts):
            return False
    return True


def source_document_label(fact) -> str:
    """Use only identity-bound governing source metadata, never a local guess."""
    context = getattr(fact, "evidence_context", None)
    if context is None or not context.structural_binding_valid or not context.source_id:
        return ""
    provenance = context.provenance_for("table_chart_title")
    label = str(context.table_chart_title or "")
    if (provenance is None or provenance.source_container not in {"metadata.title", "metadata.table_title"}
            or provenance.value != label or provenance.source_quote != label):
        return ""
    return label


def financial_fact_errors(role: str, scope: str, currency: str, fact) -> tuple[str, ...]:
    """Recompute financial admission from independent source roles/provenance."""
    if not role:
        return ()
    errors = []
    claim = str(getattr(fact, "source_claim", "") or "")
    monetary_literals = [match for match in _MONETARY_NUMBER.finditer(claim)
                         if currency_for_span(claim, match.start(), match.end())]
    if len(monetary_literals) != 1:
        errors.append("financial_source_amount_ambiguous")
    if monetary_role(claim) != role:
        errors.append("financial_row_role_incompatible")
    source_currency = normalize_currency(getattr(fact, "evidence_unit", "") or getattr(fact, "unit", ""))
    literal = str(getattr(fact, "value", "") or "")
    occurrences = list(re.finditer(re.escape(literal), claim)) if literal else []
    if (not source_currency or len(occurrences) != 1
            or currency_for_span(claim, occurrences[0].start(), occurrences[0].end()) != source_currency):
        errors.append("financial_currency_unproven")
    if currency and normalize_currency(currency) != source_currency:
        errors.append("financial_currency_incompatible")
    if not scope or not document_scope_matches(scope, source_document_label(fact)):
        errors.append("financial_document_scope_incompatible")
    return tuple(errors)


def rendered_document_reference_spans(text: str, linked_facts) -> tuple[tuple[int, int], ...]:
    """Locate identifier numbers whose role and source document are proved."""
    output = []
    for match in _DOC.finditer(text):
        kind = canonical_document_kind(match.group("kind"))
        scope = json.dumps({"kind": kind, "number": match.group("number"), "terms": [], "groups": []})
        if any(monetary_role(str(getattr(fact, "source_claim", "") or ""))
               and normalize_currency(getattr(fact, "unit", ""))
               and document_scope_matches(scope, source_document_label(fact)) for fact in linked_facts):
            output.append(match.span("number"))
    return tuple(output)


def rendered_financial_role_errors(text: str, linked_facts) -> tuple[str, ...]:
    source_roles = {monetary_role(str(getattr(fact, "source_claim", "") or ""))
                    for fact in linked_facts if normalize_currency(getattr(fact, "unit", ""))}
    source_roles.discard("")
    if not source_roles:
        return ()
    if any(pattern.search(text) for pattern in _ROLES.values()) or _EXCLUDES_TAX.search(text):
        rendered = monetary_role(text)
        if not rendered or rendered not in source_roles:
            return ("rendered_financial_role_unbound",)
    return ()


def conflicting_monetary_amounts(facts) -> bool:
    amounts = set()
    for fact in facts:
        try:
            amount = Decimal(str(fact.value).replace(",", ""))
        except (InvalidOperation, ValueError):
            return True
        if not amount.is_finite():
            return True
        amounts.add((amount, normalize_currency(fact.unit)))
    return len(amounts) > 1
