import pandas as pd

# Load your dataset
df = pd.read_csv("../train.csv")

# Add breast_id column
df["breast_id"] = df["patient_id"].astype(str) + "_" + df["laterality"]

# View mapping based on your logic
view_map = {
    "CC": "CC",
    "MLO": "MLO",
    "AT": "MLO",
    "LM": "CC",
    "ML": "MLO",
    "LMO": "MLO"
}

# Apply view mapping
df["mapped_view"] = df["view"].map(view_map)

# Track reassigned image_ids
reassigned_images = df[df["view"] != df["mapped_view"]]["image_id"].tolist()

# Pivot to breast-level
breast_rows = []
for breast_id, group in df.groupby("breast_id"):
    patient_id = group["patient_id"].iloc[0]
    laterality = group["laterality"].iloc[0]
    age = group["age"].iloc[0]
    cancer = group["cancer"].iloc[0]
    biopsy = group["biopsy"].iloc[0]
    invasive = group["invasive"].iloc[0]
    BIRADS = group["BIRADS"].iloc[0]
    implant = group["implant"].iloc[0]
    density = group["density"].iloc[0]
    machine_id = group["machine_id"].iloc[0]
    difficult_negative_case = group["difficult_negative_case"].iloc[0]

    # Get image_ids for CC and MLO
    image_ids_CC = group[group["mapped_view"] == "CC"]["image_id"].astype(str).tolist()
    image_ids_MLO = group[group["mapped_view"] == "MLO"]["image_id"].astype(str).tolist()


    # Use first image_id if multiple exist
    row = {
        "breast_id": breast_id,
        "patient_id": patient_id,
        "laterality": laterality,
        "age": age,
        "cancer": cancer,
        "biopsy": biopsy,
        "invasive": invasive,
        "BIRADS": BIRADS,
        "implant": implant,
        "density": density,
        "machine_id": machine_id,
        "difficult_negative_case": difficult_negative_case,
        "image_id_CC": ",".join(image_ids_CC),
        "image_id_MLO": ",".join(image_ids_MLO)

        # "image_id_CC": image_id_CC[0] if image_id_CC else None,
        # "image_id_MLO": image_id_MLO[0] if image_id_MLO else None
    }
    breast_rows.append(row)

# Create final DataFrame
breast_df = pd.DataFrame(breast_rows)

breast_df.to_csv('../train_breast.csv', index=False)

# Save or inspect
print("Converted breast-level data:")
print(breast_df.head())

print(f"\nImage IDs reassigned to CC or MLO: {len(reassigned_images)}")
print(reassigned_images)