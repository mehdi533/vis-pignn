# test_timeseries_pq.py
import pandas as pd
import andes



# 2) Load a case (adjust path if needed)
ss = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=False)  # example case bundled with ANDES

print(ss.PQ.as_df())
# 3) Pick a PQ to control (here: first PQ on Bus 5, else the first PQ in the case)
pq_idx = ss.PQ.as_df().query("bus == 3").idx.values[0]
P0_base = ss.PQ.as_df().query('name == "PQ_2"').p0.values[0]
Q0_base = ss.PQ.as_df().query('name == "PQ_2"').q0.values[0]
print(f"Controlling PQ idx={pq_idx}")

# print(P0_base)
# 1) Create a small schedule CSV: exact timestamps for mode=1
df = pd.DataFrame({
    "t":  [10, 20, 30],
    "P": [1.20 * P0_base,   0.80 * P0_base,   1.00 * P0_base],
    "Q": [1.10 * Q0_base,   0.90 * Q0_base,   1.00 * Q0_base],
})
df.to_csv("load_schedule.csv", index=False)
print("Wrote load_schedule.csv:")
print(df)

# 5) Attach TimeSeries to overwrite PQ.P and PQ.Q at those timestamps
#    Set silent=0 so it prints a log each time it sets a value
ss.add("TimeSeries", dict(
    # mode=2,                 # 1 = exact time matches (default)
    path="/Users/cloud9/Desktop/ETH Project/Working folder/03_Code/load_schedule.csv",
    sheet="",               # not used for CSV
    tkey="t",
    fields="P,Q",           # columns in CSV to read
    model="PQ",             # model to write into
    dev=pq_idx,             # idx of the PQ row
    dests="p0,q0",            # fields on PQ to set
))

ss.TimeSeries.config.silent=0

print(ss.Bus.as_df())
# 4) Power flow, then configure TDS to hit exact times 0,10,20,30
ss.setup()
ss.PFlow.run()
ss.TDS.config.h = 1.0      # step = 1 s  → visits t = 0, 1, 2, ..., 30
ss.TDS.config.tf = 30.0    # final time
ss.TDS.config.t0 = 0.0

# 6) Run TDS
ss.TDS.run()
ss.TDS.load_plotter()
print(ss.PQ.as_df())
ss.TDS.plt.plot(ss.PQ.Ppf, a=(2,))
