import json 


def exclude_column(path, col_name, col):
    with open(path, "r") as f:
        data = json.load(f) 

    data[col_name] = col

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


