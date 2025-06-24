import csv

def convert_to_csv(txt_path, csv_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        parts = line.strip().split(":::")
        if len(parts) == 4:
            id_, title_year, genre, plot = [p.strip() for p in parts]
            rows.append([id_, title_year, genre, plot])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Title_Year", "Genre", "Plot"])
        writer.writerows(rows)

#  file path
txt_path = r"C:\Users\wishv\OneDrive\Documents\train_data.txt"
csv_path = r"C:\Users\wishv\OneDrive\Documents\movies.csv"

convert_to_csv(txt_path, csv_path)
print("✅ CSV file created at:", csv_path)
