"""
Generate extended raw dataset for Product Price Forecasting with Explainability.
Generates 500+ realistic product records with footwear domain attributes.
"""

import os
import csv
import random
from datetime import datetime, timedelta

def generate_full_dataset(target_path, num_records=500):
    brands = [
        ("Nike", 5200, 1.3),
        ("Adidas", 4900, 1.25),
        ("Puma", 3400, 1.05),
        ("Reebok", 3100, 1.0),
        ("Skechers", 3600, 1.1),
        ("Asics", 5500, 1.35),
        ("Under Armour", 5100, 1.25),
        ("Woodland", 4200, 1.15),
        ("Bata", 1800, 0.75),
        ("Campus", 1250, 0.6)
    ]

    categories = [
        ("Running", 1.2, ["Mesh", "Knit", "Synthetic"]),
        ("Sneakers", 1.1, ["Leather", "Canvas", "Suede"]),
        ("Training", 1.15, ["Mesh", "Knit"]),
        ("Casual", 0.95, ["Canvas", "Leather", "Suede"]),
        ("Formal", 1.25, ["Leather"]),
        ("Walking", 0.9, ["Mesh", "Knit"]),
        ("Boots", 1.45, ["Leather", "Suede"]),
        ("Basketball", 1.35, ["Synthetic", "Leather"])
    ]

    genders = ["Men", "Women", "Unisex"]
    start_date = datetime(2023, 6, 1)

    random.seed(42)
    rows = [[
        "Product_ID", "Date", "Product_Name", "Brand", "Category", "Gender",
        "Material", "Size", "Rating", "Reviews_Count", "Stock_Quantity",
        "Sales_Quantity", "Competitor_Price", "Discount_Percentage", "Original_Price", "Price"
    ]]

    for i in range(1, num_records + 1):
        brand_name, brand_base, brand_mult = random.choice(brands)
        cat_name, cat_mult, possible_materials = random.choice(categories)
        material = random.choice(possible_materials)
        gender = random.choice(genders)
        size = random.choice([6, 7, 8, 9, 10, 11, 12])

        # Date distribution
        day_offset = random.randint(0, 450)
        curr_date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        # Material multiplier
        mat_mult = 1.2 if material in ["Leather", "Suede"] else (1.1 if material == "Knit" else 1.0)

        # Base price and variations
        raw_base = brand_base * cat_mult * mat_mult * random.uniform(0.85, 1.25)
        original_price = round(raw_base / 50) * 50  # Round to nearest 50

        # Competitor price close to original price
        competitor_price = round((original_price * random.uniform(0.90, 1.10)) / 50) * 50

        # Rating and Reviews
        rating = round(random.uniform(3.3, 4.9), 1)
        reviews_count = int(random.expovariate(1/120) + 15)
        stock_quantity = random.randint(15, 250)
        sales_quantity = max(5, int(random.normalvariate(45, 18)))

        # Discount percentage (0% to 50%)
        discount_percentage = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 50])
        # If older stock or lower rating, higher discount
        if rating < 3.8:
            discount_percentage = min(50, discount_percentage + 15)

        price = round(original_price * (1.0 - (discount_percentage / 100.0)), 2)

        tier = "Elite" if original_price > 5500 else ("Pro" if original_price > 3500 else "Classic")
        product_name = f"{brand_name} {cat_name} {tier}"
        prod_id = f"PROD_{1000 + i}"

        rows.append([
            prod_id, curr_date, product_name, brand_name, cat_name, gender,
            material, size, rating, reviews_count, stock_quantity,
            sales_quantity, competitor_price, discount_percentage, original_price, price
        ])

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[OK] Successfully wrote {len(rows)-1} product price records to {target_path}")

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "product_prices.csv")
    generate_full_dataset(csv_file, 600)
