
import json
import pandas as pd

df = pd.DataFrame()
number_of_files = 1
for i in range(number_of_files):
     file = open(f"./Article_details.json", 'r')
     s = file.read()
     data = json.loads(s)
     dfi = pd.DataFrame([data])
     df = pd.concat([df, dfi], ignore_index=True)

# # Saving df as a pickle
df.to_pickle('./data.pkl')