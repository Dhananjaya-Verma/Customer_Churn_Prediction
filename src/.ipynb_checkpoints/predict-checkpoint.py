# src/predict.py
import argparse, os, joblib, pandas as pd

def main(args):
    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.input)
    # if 'customerID' in df: keep or drop based on your needs
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)[:,1]
    df['pred_churn'] = preds
    df['prob_churn'] = probs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print("Saved predictions to:", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/churn_pipeline.pkl', help='path to pipeline .pkl')
    parser.add_argument('--input', required=True, help='input csv for prediction')
    parser.add_argument('--output', default='outputs/predictions.csv', help='output csv path')
    args = parser.parse_args()
    main(args)