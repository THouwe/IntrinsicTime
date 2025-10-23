import pandas as pd
from pathlib import Path

def run_dcos_from_price_parquet(parquet_path, output_csv_path, threshold=0.01, initialMode=-1):
    """
    Runs DcOS on a Parquet file with 'Timestamp' and 'Price' columns.
    The output CSV has 'Timestamp' as index and records event data.

    Parameters
    ----------
    parquet_path : str or Path
        Path to input Parquet file containing 'Timestamp' and 'Price'.
    output_csv_path : str or Path
        Destination CSV file for event output.
    threshold : float
        Relative directional-change threshold δ.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by 'Timestamp' with columns ['price', 'event_code', 'nDC', 'nOS'].
    """
    import pandas as pd
    from pathlib import Path
    from dcos_core.dcos_core import DcOS, Sample

    parquet_path = Path(parquet_path)
    output_csv_path = Path(output_csv_path)

    # Load and sort input
    df = pd.read_parquet(parquet_path)
    df['time'] = df.index
#     if not {'Timestamp', 'Price'}.issubset(df.columns):
#         raise ValueError("Expected columns ['Timestamp', 'Price'] in input file")
#     df = df.sort_values("Timestamp").reset_index(drop=True)

    # Initialize detector
    dcos = DcOS(threshold=threshold, initialMode=initialMode, midpriceMode=False)

    # Run detection
    results = []
    for _, row in df.iterrows():
        sample = Sample(row["Price"], row["time"])
        event_code = dcos.run(sample)
        results.append([sample.time, sample.level, event_code, dcos.nDC, dcos.nOS])

    # Build DataFrame and set index
    events_df = pd.DataFrame(results, columns=["time", "price", "event_code", "nDC", "nOS"])
#     events_df = events_df.set_index("time")

    # Save
    events_df.to_csv(output_csv_path)
    print(f"Saved {len(events_df)} rows to {output_csv_path}")
    return events_df
    
# run_dcos_from_price_parquet(
#     parquet_path="BTCUSDT_sample.parquet",
#     output_csv_path="BTCUSDT_events.csv",
#     threshold=0.001
# )
