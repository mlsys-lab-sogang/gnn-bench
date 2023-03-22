import numpy as np
import pandas as pd

from tqdm import tqdm


def generate_freq():
  for dataset in ["reddit", "ogbn_products", "ogbn_papers100M"]:
    degrees = pd.read_csv(f"logs/degree_{dataset}.csv")["degree"]
    frequencies = np.zeros(shape=(degrees.max()-degrees.min()+1), dtype=np.int64)

    for degree in tqdm(degrees):
      frequencies[degree-degrees.min()] += 1

    df = pd.DataFrame(columns=["degree", "frequency"])
    for degree, freq in enumerate(tqdm(frequencies)):
      df.loc[len(df)] = [degree+degrees.min(), freq]

    df.to_csv(f"logs/freq_{dataset}.csv", index=False)

if __name__ == "__main__":
  generate_freq()
