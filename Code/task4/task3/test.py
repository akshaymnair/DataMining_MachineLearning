import pandas as pd
import numpy as np


df = pd.read_pickle("movie_matrix_svd.pkl")


print df.loc[[1,2,3,4]]
