import pandas as pd

# Load the image-level CSV
df = pd.read_csv("../DM_expanded.csv")

# Create breast_id from image_id (e.g., P1_L_DM_MLO → P1_L)
df["breast_id"] = df["image_id"].apply(lambda x: "_".join(x.split("_")[:2]))

# Group by breast_id and aggregate relevant fields
breast_df = df.groupby("breast_id").agg({
    "patient_id": "first",
    "cancer": "max",  # If any image has cancer=1, mark breast as cancerous
    "image_id": lambda x: list(x)  # Keep full image_id list for later filtering
}).reset_index()

# Extract image_id_CC and image_id_MLO as comma-separated strings
def extract_views(image_ids, view_type):
    return ",".join([img for img in image_ids if f"_{view_type}" in img])

breast_df["image_id_CC"] = breast_df["image_id"].apply(lambda x: extract_views(x, "CC"))
breast_df["image_id_MLO"] = breast_df["image_id"].apply(lambda x: extract_views(x, "MLO"))

# Drop the raw image_id list column
breast_df.drop(columns=["image_id"], inplace=True)

# Save to new CSV
breast_df.to_csv("../DM_breast.csv", index=False)