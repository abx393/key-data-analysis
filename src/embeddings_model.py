import os
import pandas as pd

DIR_IN = "features/"
KEYBOARD_TYPE = "mechanical"
DATA_FILE = "vggish_embeddings"

df = pd.read_csv(os.path.join(DIR_IN, KEYBOARD_TYPE, ))
print(df.head())
