from .dataset import VISDataset
from .config import DATA_DIR

def main():
    _ = VISDataset(DATA_DIR, regenerate=True)
    print("✅ Dataset generated at:", (DATA_DIR / "vis_pignn_dataset.pt").as_posix())

if __name__ == "__main__":
    main()
