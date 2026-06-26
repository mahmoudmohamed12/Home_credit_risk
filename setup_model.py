import lightgbm as lgb
import joblib

def main():
    print("Loading text model...")
    booster = lgb.Booster(model_file="final_lgb_production_model.txt")
    print("Saving to joblib pkl...")
    joblib.dump(booster, "model/final_model.pkl")
    print("Done!")

if __name__ == "__main__":
    main()
