import andes
from andes.utils.paths import get_case
import pandas as pd

andes.config_logger(stream_level=20)

# use the dynamic 39-bus (you already have it)
ss = andes.run(get_case('kundur/kundur_full.xlsx'), default_config=True)
ss.TDS.config.tf = 10.0
ss.TDS.run()
ss.TDS.load_plotter()
ss.TDS.plt.export_csv("tds_out.csv")

# Try in-memory table first
df = getattr(ss.TDS.plt, "df", None)
if df is None:
    ss.TDS.plt.export_csv("tds_out.csv")
    df = pd.read_csv("tds_out.csv")

print("OK; first columns:", list(df.columns)[:12])
print("Rows:", len(df))
