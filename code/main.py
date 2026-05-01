import os
import argparse
import pandas as pd
from tqdm import tqdm
from agent import process_ticket

def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate Triage Agent")
    parser.add_argument("--input", type=str, default="support_tickets/support_tickets.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="support_tickets/output.csv", help="Path to output CSV")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading tickets from {input_path}")
    df = pd.read_csv(input_path)
    
    # Normalize column names to lowercase
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Prepare output lists
    statuses = []
    product_areas = []
    responses = []
    justifications = []
    request_types = []

    print("Processing tickets...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        issue = str(row.get("issue", ""))
        subject = str(row.get("subject", ""))
        company = str(row.get("company", "None"))
        
        # Guard against fully empty rows
        if not issue.strip() and not subject.strip():
            statuses.append("escalated")
            product_areas.append("unknown")
            responses.append("No issue provided.")
            justifications.append("Empty ticket.")
            request_types.append("invalid")
            continue

        result = process_ticket(issue, subject, company)
        
        statuses.append(result["status"])
        product_areas.append(result["product_area"])
        responses.append(result["response"])
        justifications.append(result["justification"])
        request_types.append(result["request_type"])

    # Update DataFrame with predictions
    df_out = df.copy()
    df_out["status"] = statuses
    df_out["product_area"] = product_areas
    df_out["response"] = responses
    df_out["justification"] = justifications
    df_out["request_type"] = request_types

    print(f"Saving predictions to {output_path}")
    df_out.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
