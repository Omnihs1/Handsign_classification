import yaml
import os 

os.chdir("config")
with open("model.yaml", "r") as file:
    output = yaml.safe_load(file)

print(output["author"])

