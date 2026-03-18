# Polymarket Quant Strategy Package
from .lmsr import LMSRMarket, lmsr_price, price_impact, scan_arb_window
from .kelly import kelly_fraction, fractional_kelly, simulate_growth
from .ev_scanner import ev_gap, rank_opportunities, MarketSignal
from .kl_divergence import kl_divergence, symmetric_kl, analyze_pair, scan_pairs, MarketPair
from .bregman import bregman_project, find_mispriced_outcomes, MultiOutcomeMarket
from .bayesian import BayesianModel, Evidence, edge_vs_market
from .portfolio_manager import allocate, Portfolio, Trade, Category
from .backtest import walk_forward_backtest, compute_metrics
