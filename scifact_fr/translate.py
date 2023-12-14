from datasets import load_dataset

import json

with open('./corpus.jsonl', 'r') as file:
    corpus = [json.loads(line) for line in file]

with open('./queries.jsonl', 'r') as file:
    queries = [json.loads(line) for line in file]

scifact_es = load_dataset('sproos/scifact-fr')

for i in range(len(corpus)):
    corpus[i]['title'] = scifact_es['corpus'][i]['title']
    corpus[i]['text'] = scifact_es['corpus'][i]['text']
for i in range(len(queries)):
    queries[i]['text'] = scifact_es['queries'][i]['text']

with open('./corpus.jsonl', 'w') as file:
    for line in corpus:
        json.dump(line, file)
        file.write('\n')

with open('./queries.jsonl', 'w') as file:
    for line in queries:
        json.dump(line, file)
        file.write('\n')