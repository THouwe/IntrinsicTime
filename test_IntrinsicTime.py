import numpy as np
import pandas as pd
from IntrinsicTime import DcOS_fractal

def main():
    df = pd.DataFrame({
        "Timestamp": np.arange(1000),
        "Price": 100 + np.cumsum(np.random.randn(1000))
    })
    analyzer = DcOS_fractal(debugMode=True)
    results = analyzer.run_count_and_analysis(df)
    print(results.head())

if __name__ == "__main__":
    main()
