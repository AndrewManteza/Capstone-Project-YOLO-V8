import json

with open("gun_detector.ipynb", encoding="utf-8") as f:
    data = json.load(f)

print("Cell IDs and indices:")
for i, cell in enumerate(data["cells"]):
    print(f"{i}: {cell.get('id')} - {cell.get('cell_type')}")

# Find the predict cell - it's the one with model.predict
for i, cell in enumerate(data["cells"]):
    if cell.get("cell_type") == "code":
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "model.predict" in source:
            print(f"\n\nFound predict cell at index {i}:")
            print(source)
            break
