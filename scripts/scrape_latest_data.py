import pandas as pd
import os

# 1. This finds exactly where this script is sitting
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. This moves "up" one level to your main epl-moneyball folder
project_root = os.path.dirname(script_dir)

# 3. This builds the perfect path to the data/raw folder
save_path = os.path.join(project_root, 'data', 'raw', 'scraped_latest_standings.csv')

url = "https://www.skysports.com/premier-league-table"

try:
    # Scrape the table
    tables = pd.read_html(url)
    latest_table = tables[0]
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the file
    latest_table.to_csv(save_path, index=False)
    print(f"✅ Success! File saved to: {save_path}")

except Exception as e:
    print(f"❌ Scraping failed: {e}")