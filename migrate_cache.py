"""一次性导入旧行情 CSV 到 DuckDB，不删除源文件。"""
from src import config as C, data as D, storage


def main():
    count = 0
    for path in sorted(C.DATA_DIR.glob("*.csv")):
        if not (path.stem.startswith(("fund_nav_", "index_universe_")) or path.stem == "pe_panel"):
            continue
        if storage.metadata(path.stem) is not None:
            continue
        D._cached(path.stem, lambda: None)
        count += 1
    print(f"已导入 {count} 份行情缓存：{C.DATA_DIR / 'market.duckdb'}")


if __name__ == "__main__":
    main()
