import sys

import numpy as np
import pandas as pd

from tqdm import tqdm


def generate_pdf(filename):
  degrees = pd.read_csv(filename).loc[:,"degree"].astype(np.int64)
  counts = np.zeros(shape=(degrees.max()-degrees.min()+1), dtype=np.float64)

  for degree in tqdm(degrees):
    counts[degree-degrees.min()] += 1.0

  pdf = pd.DataFrame({"degree": range(degrees.min(), degrees.max()+1), "norm": range(degrees.min(), degrees.max()+1), "pdf": counts})
  pdf.loc[:,"norm"] = pdf.loc[:,"norm"].astype(np.float64)

  for idx in tqdm(range(len(pdf))):
    pdf.loc[idx,"norm"] = (pdf.loc[idx,"norm"]-degrees.min())/(degrees.max()-degrees.min())
    pdf.loc[idx,"pdf"] /= degrees.size

  pdf.to_csv("degree_pdf".join(filename.split("degree")), index=False)

def main(argv):
  if len(argv) != 2:
    sys.exit("Expected one argument but got {}.".format(len(argv)-1))
  generate_pdf(argv[1])

if __name__ == "__main__":
  main(sys.argv)
