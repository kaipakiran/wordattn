import pandas as pd
import os

negative_comments = []
negative_score = []
positive_comments = []
positive_score = []
for file in os.listdir('./data/aclImdb/test/neg'):
    with open(os.path.join('./data/aclImdb/test/neg',file)) as f:
        negative_comments.append(f.read())
        negative_score.append(int(file.split("_")[1].split(".")[0])*-1)

for file in os.listdir('./data/aclImdb/test/pos'):
    
    with open(os.path.join('./data/aclImdb/test/pos',file)) as f:
        positive_comments.append(f.read())
        positive_score.append(int(file.split("_")[1].split(".")[0]))

    
df = pd.concat([pd.DataFrame({"comments":negative_comments,"score":negative_score}),pd.DataFrame({"comments":positive_comments,"score":positive_score})]).sample(frac=1).reset_index(drop=True)

df.head()

df.to_csv("./data/imdb_test.csv",index=False)

