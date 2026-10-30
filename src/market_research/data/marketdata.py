"""Compatibility facade; providers, caching and panels have separate modules."""
from market_research.data.http import _default_http_timeout, http_timeout, _retry, _clean_series
from market_research.data.providers.funds import NAV_BASIS_CUM, NAV_BASIS_TOTAL, NAV_BASIS_NOTE, fetch_fund_nav, total_return_index, _fetch_nav, fetch_fund_unit_nav, fetch_cum_return_curve, calibrate_total_return, fetch_fund_total_return, verify_nav_basis
from market_research.data.providers.indices import fetch_index_sina, fetch_index_sw, _fetch_one, fetch_universe
from market_research.data.providers.valuation import fetch_pe_panel
from market_research.data.cache import data_stamp, _cache_meta_path, _fingerprint, _cache_meta_valid, _cache_meta_write, cache_info, _cache_info_write, _cached, _with_info
from market_research.data.panels import align_prices, to_log_returns, load_pe_panel, load_index_universe, load_fund_nav_cached, fetch_start_for, load_panel
from market_research.data.files import _atomic_csv_write, _os_replace_retry
