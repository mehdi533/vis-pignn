import andes
from plot_network import plot_freq_and_rocof, plot_network

if __name__ == "__main__":
    ss = andes.load(andes.get_case("ieee39/ieee39_full.xlsx"), name="IEEE39")
    plot_network(ss, layout="kamada", save_path=f"figures/network_{ss.name}_kamada.png")