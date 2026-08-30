import os
from pathlib import Path

# Create corpus directories
project_root = Path(__file__).resolve().parent.parent
corpus_dir = project_root / "data" / "corpus"
corpus_dir.mkdir(parents=True, exist_ok=True)

# Document details
documents = {
    # 1. Exchange Rulebook (EXCH_RULE)
    "EXCH_RULE_01_TRADING_HOURS.txt": {
        "title": "Exchange Trading Hours",
        "category": "Exchange Rulebook",
        "content": """
Clause 1.1: Normal Market Hours
The equity and derivatives (F&O) segments are open for trading from Monday to Friday. The pre-open session runs from 09:00 hours to 09:15 hours. The normal trading market hours are from 09:15 hours to 15:30 hours.

Clause 1.2: Post-Closing Session
The post-closing trading session is conducted between 15:40 hours and 16:00 hours. Transactions during this session are executed at the closing price of the security.

Clause 1.3: Weekly Holidays
The exchange remains closed on Saturdays and Sundays, and on public holidays declared by the exchange in advance at the start of the calendar year.
"""
    },
    "EXCH_RULE_02_ORDER_TYPES.txt": {
        "title": "Supported Order Types",
        "category": "Exchange Rulebook",
        "content": """
Clause 2.1: Limit and Market Orders
Market orders execute immediately at the best available current price. Limit orders are placed in the order book with a specified maximum purchase price or minimum sales price.

Clause 2.2: Stop-Loss Orders
A stop-loss order becomes a market order once the trigger price is reached. For Stop-Loss Limit orders, once the trigger price is reached, it is placed as a limit order at the specified limit price.

Clause 2.3: Disclosed Quantity Orders
An investor can disclose only a part of the total order quantity to the market. The disclosed quantity must be at least 10% of the total order size and must not be less than the minimum lot size.
"""
    },
    "EXCH_RULE_03_CIRCUIT_BREAKERS.txt": {
        "title": "Market-Wide Circuit Breakers",
        "category": "Exchange Rulebook",
        "content": """
Clause 3.1: Trigger Levels
Market-wide circuit breakers are triggered by movement in either the Nifty 50 or Sensex index. The circuit breakers are set at three levels: 10%, 15%, and 20% index movement.

Clause 3.2: Market Halt Duration
- A 10% trigger before 13:00 hours results in a 45-minute market halt.
- A 10% trigger at or after 13:00 hours and before 14:30 hours results in a 15-minute halt.
- A 10% trigger at or after 14:30 hours results in no halt.
- A 15% trigger before 13:00 hours results in a 1-hour 45-minute halt.
- A 15% trigger at or after 13:00 hours and before 14:30 hours results in a 45-minute halt.
- A 15% trigger at or after 14:30 hours halts trading for the remainder of the day.
- A 20% trigger at any time halts trading for the rest of the day.
"""
    },
    "EXCH_RULE_04_INDEX_CONTRACT_SPECS.txt": {
        "title": "Index Derivatives Specifications",
        "category": "Exchange Rulebook",
        "content": """
Clause 4.1: Contract Expiry
Index futures and options contracts expire on the last Thursday of the expiry month. If the last Thursday is a holiday, the contracts expire on the preceding working day.

Clause 4.2: Lot Sizes
The contract lot size is determined by the exchange periodically. The lot size for Nifty Futures is 75 units, and the lot size for Bank Nifty Futures is 15 units.

Clause 4.3: Tick Size
The minimum price movement (tick size) for index futures and options contracts is set at 0.05 currency units.
"""
    },
    "EXCH_RULE_05_BLOCK_TRADES.txt": {
        "title": "Block Trade Guidelines",
        "category": "Exchange Rulebook",
        "content": """
Clause 5.1: Block Deal Window
Block deals are executed during a special window. The morning window is open from 08:45 hours to 09:00 hours. The afternoon window is open from 14:05 hours to 14:20 hours.

Clause 5.2: Minimum Deal Value
A transaction qualifies as a block deal only if the minimum order value is 100 million currency units.

Clause 5.3: Execution Price
The execution price of a block deal must be within a range of +/- 1% of the current market price or previous closing price.
"""
    },
    "EXCH_RULE_06_STOCK_CONTRACT_SPECS.txt": {
        "title": "Stock Derivatives Specifications",
        "category": "Exchange Rulebook",
        "content": """
Clause 6.1: Eligibility Criteria
Stocks must meet minimum liquidity and market capitalization thresholds for 6 consecutive months to be eligible for the F&O segment. The stock's average daily market capitalization must be at least 5 billion currency units.

Clause 6.2: Delivery Settlement
All open stock futures and option contracts at expiry are settled through physical delivery of shares. Cash settlement is not permitted for stock derivatives at expiry.
"""
    },
    "EXCH_RULE_07_PRE_OPEN_SESSION.txt": {
        "title": "Pre-Open Session Mechanics",
        "category": "Exchange Rulebook",
        "content": """
Clause 7.1: Order Entry Period
The pre-open session order entry, modification, and cancellation period is from 09:00 hours to 09:08 hours.

Clause 7.2: Order Matching Period
The order matching and price discovery period starts at 09:08 hours and completes at 09:12 hours. Trading buffer time is from 09:12 hours to 09:15 hours.
"""
    },
    "EXCH_RULE_08_POST_CLOSE_SESSION.txt": {
        "title": "Post-Close Session Rules",
        "category": "Exchange Rulebook",
        "content": """
Clause 8.1: Eligible Orders
Only market orders are permitted during the post-close session from 15:40 hours to 16:00 hours. Orders are matched against the closing price discovered in the normal market session.
"""
    },
    "EXCH_RULE_09_ALGO_TRADING_RULES.txt": {
        "title": "Algorithmic Trading Regulations",
        "category": "Exchange Rulebook",
        "content": """
Clause 9.1: Prior Approval
All algorithmic trading strategies must receive prior approval from the exchange before deployment. Unapproved algorithms are strictly prohibited and subject to a penalty of 100,000 currency units.

Clause 9.2: Order-to-Trade Ratio
Brokers must maintain an Order-to-Trade Ratio (OTR) of less than 50:1 daily. Exceeding this ratio attracts progressive economic charges.
"""
    },
    "EXCH_RULE_10_ERROR_TRADE_POLICY.txt": {
        "title": "Error Trade Policy",
        "category": "Exchange Rulebook",
        "content": """
Clause 10.1: Cancellation Request
A broker can request the cancellation of an error trade within 30 minutes of trade execution. The exchange reserves the right to accept or reject the trade cancellation request.
"""
    },

    # 2. Brokerage Margin & F&O Policy (MARGIN_POL)
    "MARGIN_POL_01_SPAN_EXPOSURE.txt": {
        "title": "SPAN and Exposure Margins",
        "category": "Margin & F&O Policy",
        "content": """
Clause 1.1: Margin Composition
For trading in the F&O segment, the client must maintain both SPAN Margin and Exposure Margin.
- The SPAN Margin is calculated based on portfolio risk metrics.
- The Exposure Margin is set by the brokerage as a safety buffer, typically 3.5% of the total contract value.

Clause 1.2: Minimum Margin Requirements
The total margin (SPAN + Exposure) must be present in the account in cash or approved collateral before executing a trade. The minimum initial margin requirement for Index Futures is 12% of the contract value.
"""
    },
    "MARGIN_POL_02_PEAK_MARGIN.txt": {
        "title": "Peak Margin Requirements",
        "category": "Margin & F&O Policy",
        "content": """
Clause 2.1: Intraday Margin Audits
The clearing corporation takes 4 random snapshots of client positions during trading hours. The maximum margin requirement among these snapshots is defined as the Peak Margin.

Clause 2.2: Margin Shortfall Penalties
If the client's available margin falls below the Peak Margin requirement during any snapshot, a peak margin shortfall penalty is levied, ranging from 0.5% to 5.0% of the shortfall amount depending on the duration and size.
"""
    },
    "MARGIN_POL_03_MTM_SETTLEMENT.txt": {
        "title": "Mark-to-Market (MTM) Policy",
        "category": "Margin & F&O Policy",
        "content": """
Clause 3.1: Daily MTM Calculation
All open derivative positions are marked to market at the end of every trading day based on the closing price. If the daily MTM loss exceeds 50% of the client's available cash balance, an immediate warning alert is sent.

Clause 3.2: Immediate MTM Cutoff
If the accumulated MTM loss reaches 80% of the client's total ledger margin balance, the brokerage system automatically initiates square-off of open positions.
"""
    },
    "MARGIN_POL_04_MARGIN_CALLS.txt": {
        "title": "Margin Call Guidelines",
        "category": "Margin & F&O Policy",
        "content": """
Clause 4.1: Shortfall Notification
A Margin Call is triggered when the available margin falls below 100% of the required margin. The client is notified via registered email and SMS to deposit funds immediately.

Clause 4.2: Funding Timeline
The client must fulfill the margin call shortfall before 23:59 hours on the day the margin call is made. Failure to deposit funds results in automatic square-off on the next trading day at market open.
"""
    },
    "MARGIN_POL_05_AUTO_SQUARE_OFF.txt": {
        "title": "Auto-Square-Off Criteria",
        "category": "Margin & F&O Policy",
        "content": """
Clause 5.1: Intraday Auto-Square-Off
Intraday (MIS) positions must be closed by the client before 15:15 hours. If the client fails to close MIS positions, the brokerage auto-square-off system executes market exit orders beginning at 15:16 hours.

Clause 5.2: Auto-Square-Off Fee
A charge of 50 currency units per squared-off order is levied for positions closed by the automated square-off system.
"""
    },
    "MARGIN_POL_06_COLLATERAL_HAIRCUTS.txt": {
        "title": "Collateral and Haircuts",
        "category": "Margin & F&O Policy",
        "content": """
Clause 6.1: Approved Collateral List
Clients can pledge approved liquid stocks and mutual funds to get margin benefits. 

Clause 6.2: Haircut Ratios
Pledged collaterals are subject to value haircuts. Group I stocks are subject to a minimum haircut of 15%. Liquid mutual funds are subject to a 10% haircut. Equity mutual funds are subject to a 20% haircut.
"""
    },
    "MARGIN_POL_07_INTRA_DAY_LEVERAGE.txt": {
        "title": "Intraday Leverage Limits",
        "category": "Margin & F&O Policy",
        "content": """
Clause 7.1: Equity Delivery Leverage
No leverage is provided for Equity Delivery trades. 100% of the purchase value must be paid by the client upfront.

Clause 7.2: Equity Intraday (MIS) Leverage
Intraday trades in highly liquid stocks are allowed up to 5x leverage (or 20% margin requirement). F&O intraday trades do not receive extra leverage beyond the exchange prescribed margins.
"""
    },
    "MARGIN_POL_08_OPTION_WRITING_MARGINS.txt": {
        "title": "Option Writing Margins",
        "category": "Margin & F&O Policy",
        "content": """
Clause 8.1: Naked Option Writing
Option writers (sellers) must maintain full SPAN + Exposure margin. Writing naked options is permitted only if the client's account net equity exceeds 50,000 currency units.
"""
    },

    # 3. Settlement and Payout Procedures (SETTLE_PROC)
    "SETTLE_PROC_01_T1_CYCLE.txt": {
        "title": "T+1 Settlement Cycle",
        "category": "Settlement and Payout Procedures",
        "content": """
Clause 1.1: Equity Trades Settlement
All equity trades (delivery) are settled on a T+1 basis (Trade day plus one working day). Stock credits are delivered to the client's depository participant (DP) account by 14:00 hours on the T+1 day.

Clause 1.2: F&O Settlement
All derivative contract trades are settled on a daily mark-to-market basis on T+1. Realized profits are credited, and realized losses are debited, before 08:30 hours on the next working day.
"""
    },
    "SETTLE_PROC_02_PAYIN_PAYOUT.txt": {
        "title": "Pay-In and Pay-Out Timelines",
        "category": "Settlement and Payout Procedures",
        "content": """
Clause 2.1: Fund Pay-In (Deposit)
Deposits made via UPI or Net Banking reflect instantly in the trading ledger, within a maximum processing window of 10 minutes.

Clause 2.2: Fund Pay-Out (Withdrawal)
Withdrawal requests placed before 08:30 hours are processed and credited to the client's bank account on the same day by 14:00 hours. Requests placed after 08:30 hours are processed on the next working day.
"""
    },
    "SETTLE_PROC_03_BANK_CUTOFFS.txt": {
        "title": "Bank Integration Cutoffs",
        "category": "Settlement and Payout Procedures",
        "content": """
Clause 3.1: Net Banking Cutoffs
Net Banking fund transfers are supported 24/7. However, bank maintenance windows between 23:00 hours and 01:30 hours daily may cause transfer delays of up to 2 hours.
"""
    },
    "SETTLE_PROC_04_AUCTION_SETTLEMENT.txt": {
        "title": "Auction Settlement Rules",
        "category": "Settlement and Payout Procedures",
        "content": """
Clause 4.1: Short Delivery Handling
If a seller fails to deliver shares on T+1, the transaction is marked as a short delivery. The exchange conducts a buy-in auction on T+2 to procure the shares.

Clause 4.2: Auction Pricing
The shares are purchased in the auction at the best available price. The seller is charged the difference between the transaction price and the auction price, plus a penalty fee of 2.0% of the closing price.
"""
    },
    "SETTLE_PROC_05_PHYSICAL_DELIVERY.txt": {
        "title": "Physical Delivery Rules",
        "category": "Settlement and Payout Procedures",
        "content": """
Clause 5.1: Delivery Intent
Clients with open stock derivatives positions must declare their delivery intent (willingness to give or take shares) by 11:00 hours on expiry day. Failure to declare intent leads to automated contract square-off.
"""
    },
    "SETTLE_PROC_06_CORPORATE_ACTIONS.txt": {
        "title": "Corporate Action Settlements",
        "category": "Settlement and Payout Procedures",
        "content": """
Clause 6.1: Dividends and Splits
Dividend credits are directly paid by the registrar to the client's bank account. Share splits and bonus shares are credited to the DP account on the ex-date + 2 working days.
"""
    },

    # 4. Account-Opening & Trading Terms (ACCT_TERMS)
    "ACCT_TERMS_01_KYC_REQ.txt": {
        "title": "KYC Documentation Requirements",
        "category": "Account-Opening & Trading Terms",
        "content": """
Clause 1.1: Mandatory KYC Documents
To open a trading account, the applicant must submit a valid Proof of Identity (PAN Card) and Proof of Address (Aadhaar Card, Passport, or Voter ID).

Clause 1.2: Income Proof for F&O Segment
To activate the derivatives (F&O) segment, the client must submit income proof, which must be one of:
- Bank statement for the last 6 months showing average balance of 10,000 currency units.
- Copy of ITR Acknowledgment.
- Salary slip for the last 3 months.
"""
    },
    "ACCT_TERMS_02_INACTIVE_ACCT.txt": {
        "title": "Inactive Account Policy",
        "category": "Account-Opening & Trading Terms",
        "content": """
Clause 2.1: Inactive Classification
An account is classified as inactive (dormant) if no trade has been executed in the account for a consecutive period of 12 months.

Clause 2.2: Reactivation Process
To reactivate a dormant account, the client must undergo the Re-KYC process, which includes submiting updated identity/address proofs and performing an In-Person Verification (IPV) via webcam.
"""
    },
    "ACCT_TERMS_03_POA_LIMITS.txt": {
        "title": "Power of Attorney (POA) Terms",
        "category": "Account-Opening & Trading Terms",
        "content": """
Clause 3.1: POA Purpose
The Power of Attorney (POA) or Demat Debit and Pledge Instruction (DDPI) granted by the client is limited. It allows the broker to debit shares from the client's demat account only to fulfill pay-in obligations for trades executed by the client.

Clause 3.2: Transfer Prohibitions
The broker is strictly prohibited from transferring shares from the client demat account for proprietary trading or to other client accounts.
"""
    },
    "ACCT_TERMS_04_ACCT_CLOSURE.txt": {
        "title": "Account Closure Procedure",
        "category": "Account-Opening & Trading Terms",
        "content": """
Clause 4.1: Closure Requirements
An account closure request can only be processed if there are no open positions in the F&O segment, no debit balance in the ledger, and no holding of securities in the Demat account. All holdings must be sold or transferred.

Clause 4.2: Processing Timeline
The brokerage must process a valid account closure request within 3 working days from the date of submission of the physical or digital request form.
"""
    },
    "ACCT_TERMS_05_NOMINATION_RULES.txt": {
        "title": "Nomination Regulations",
        "category": "Account-Opening & Trading Terms",
        "content": """
Clause 5.1: Mandatory Declaration
It is mandatory for all individual demat and trading accounts to either nominate a beneficiary or explicitly opt out of nomination by submitting the nomination declaration form.
"""
    },

    # 5. Fees & Brokerage Schedule (FEES_SCHED)
    "FEES_SCHED_01_BROKERAGE.txt": {
        "title": "Brokerage Charge Schedule",
        "category": "Fees & Brokerage Schedule",
        "content": """
Clause 1.1: Equity Delivery Brokerage
Equity Delivery trades are charged a brokerage of 0.1% of the trade turnover, or a maximum of 20 currency units per executed order, whichever is lower.

Clause 1.2: Equity Intraday Brokerage
Equity Intraday trades are charged a brokerage of 0.03% of the trade turnover, or 20 currency units per executed order, whichever is lower.

Clause 1.3: F&O Segment Brokerage
- Futures (both Equity and Index) are charged 0.03% of the turnover or 20 currency units per executed order, whichever is lower.
- Options (both Equity and Index) are charged a flat rate of 20 currency units per executed order.
"""
    },
    "FEES_SCHED_02_GOVT_TAXES.txt": {
        "title": "Government Taxes and Levies",
        "category": "Fees & Brokerage Schedule",
        "content": """
Clause 2.1: Securities Transaction Tax (STT)
- Equity Delivery: STT is charged at 0.1% on both buy and sell transactions.
- Equity Intraday: STT is charged at 0.025% on sell transactions only.
- Equity Futures: STT is charged at 0.0125% on sell transactions only.
- Equity Options: STT is charged at 0.0625% on sell transactions only (on option premium).

Clause 2.2: Goods and Services Tax (GST)
GST is charged at a rate of 18% on the sum of brokerage charges, exchange transaction charges, and SEBI turnover fees.
"""
    },
    "FEES_SCHED_03_OTHER_CHARGES.txt": {
        "title": "Other Statutory Charges",
        "category": "Fees & Brokerage Schedule",
        "content": """
Clause 3.1: SEBI Turnover Fee
SEBI turnover fee is charged at a flat rate of 0.0001% (10 currency units per 10 million) of the total transaction turnover across all trading segments.

Clause 3.2: Stamp Duty
Stamp duty is charged at 0.015% (buy transactions only) for Equity Delivery, 0.003% for Equity Intraday, and 0.002% for Futures. Stamp duty for Option contracts is 0.003% on premium value.
"""
    }
}

# Write files to corpus directory
for filename, doc_data in documents.items():
    file_path = corpus_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"DOCUMENT ID: {filename.replace('.txt', '')}\n")
        f.write(f"DOCUMENT TITLE: {doc_data['title']}\n")
        f.write(f"CATEGORY: {doc_data['category']}\n")
        f.write("="*40 + "\n")
        f.write(doc_data['content'].strip() + "\n")

print(f"Generated {len(documents)} synthetic corpus files in {corpus_dir}")
