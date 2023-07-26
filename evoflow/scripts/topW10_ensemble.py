import pandas as pd
from pathlib import Path
import os
import sys

from evoflow.g3p.grammar import parse_pipe_grammar
from evoflow.utils import compute_performance
from kdis.utils import load_data, create_pipeline
from sklearn.ensemble import VotingClassifier

dataset = sys.argv[1]
seed = int(sys.argv[2])
basedir = sys.argv[3]
indir = os.path.join(basedir, dataset + "_s" + str(seed))
outdir = os.path.join(basedir + "_topW10", dataset + "_s" + str(seed))
#outdir = os.path.join(basedir + "_top10", dataset + "_s" + str(seed))
Path(outdir).mkdir(parents=True, exist_ok=True)

# load the invidividuals
df = pd.read_csv(os.path.join(indir, "individuals.tsv"), sep="\t", usecols=[0,1,2])
df.drop_duplicates(subset=["pipeline", "fitness"], inplace=True)

# transform to numeric and remove invalid invididuals (i.e. nan)
df["fitness"] = pd.to_numeric(df["fitness"], errors='coerce')
df["fit_time"] = pd.to_numeric(df["fit_time"], errors='coerce')
df.dropna(inplace=True)

# reset index and sort individuals by their fitness and index (keep order of creation)
df.reset_index(inplace=True)
df.sort_values(by=["fitness", "index"], ascending=[False, True], inplace=True)

# reset again the index and remove the unwanted index column
df.reset_index(inplace=True, drop=True)
del df["index"]

# get the top 10 individuals
individuals = df["pipeline"].iloc[:10]

# load the grammar and the dataset
X_train, y_train, X_test, y_test = load_data(dataset)
print("WARN: cuidado si se usa otra gramatica")
_, _, _, pset, _ = parse_pipe_grammar("classification.xml", seed)

# generate the pipelines
estimators = []
for idx, ind in enumerate(individuals):
    estimators.append((str(idx), create_pipeline(ind, pset)))

# compute the weights of the ensemble
fitnesses = df["fitness"].iloc[:10]
best_fitness = fitnesses[0]
weights = [fit/best_fitness for fit in fitnesses]

# generate the voting ensemble
vt = VotingClassifier(estimators, voting="hard", weights=weights)
#vt = VotingClassifier(estimators, voting="hard")
vt.fit(X_train, y_train)
y_pred = vt.predict(X_test)

# compute the performance
perf_dic = compute_performance(y_test, y_pred)

# save the results
with open(os.path.join(outdir, "best_ensemble.txt"), 'w') as f:
    f.write(str(y_pred.tolist()) + "\n")
    f.write(str(perf_dic) + "\n")
    for ind in individuals:
        f.write(str(ind) + "\n")
