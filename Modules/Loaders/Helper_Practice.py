import pandas as pd
from DataFormatter import load_covasim_data_drums
import joblib

fp1 = "//Users//jordanklein22//Documents//GitHub//COVASIM_EQL_BINNS//Data//covasim_data//drums_data//covasim_50000_0.1_0.3_dynamic_piecewise_1000.joblib"
dict = joblib.load(fp1)

df = dict["data"][0]

load_covasim_data_drums("//Users//jordanklein22//Documents//GitHub//COVASIM_EQL_BINNS//Data//covasim_data//drums_data//", 50000, "50000_0.1_0.3_dynamic_piecewise_1000")