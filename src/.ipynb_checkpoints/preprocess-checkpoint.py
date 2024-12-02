import pandas as pd

def preprocess_logs(filepath):
    data = pd.read_csv(filepath)
    # Focus on the last 3 columns
    processed_data = data[["EventId", "ParameterList", "Content"]]
    processed_data.dropna(inplace=True)
    return processed_data

if __name__ == "__main__":
    logs = preprocess_logs("data/Linux_2k.log_structured.csv")
    logs.to_csv("data/processed_logs.csv", index=False)
