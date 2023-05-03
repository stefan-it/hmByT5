ubuntu@192-9-137-187:~/hmByT5/bench$ python3 flair-log-parser.py "hipe2022-ajmc/en-stefan-it/byt5-small-historic-english-span20-bs*"
Debug: defaultdict(<class 'list'>, {'wsFalse-bs4-e10-lr0.00015-poolingfirst': [0.8548, 0.846, 0.8565, 0.8683, 0.8653], 'wsFalse-bs8-e10-lr0.00016-poolingfirst': [0.8414, 0.8345, 0.844, 0.849, 0.8582], 'wsFalse-bs4-e10-lr0.00016-poolingfirst': [0.8535, 0.845, 0.8605, 0.851, 0.8518], 'wsFalse-bs8-e10-lr0.00015-poolingfirst': [0.8527, 0.853, 0.8333, 0.8525, 0.817]})
Averaged Development Results:
wsFalse-bs4-e10-lr0.00015-poolingfirst : 85.82
wsFalse-bs4-e10-lr0.00016-poolingfirst : 85.24
wsFalse-bs8-e10-lr0.00016-poolingfirst : 84.54
wsFalse-bs8-e10-lr0.00015-poolingfirst : 84.17
Markdown table:

Best configuration: wsFalse-bs4-e10-lr0.00015-poolingfirst


Best Development Score: 85.82


| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   85.48 |   84.6  |   85.65 |   86.83 |   86.53 | 85.82 ± 0.79 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   85.35 |   84.5  |   86.05 |   85.1  |   85.18 | 85.24 ± 0.5  |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   84.14 |   83.45 |   84.4  |   84.9  |   85.82 | 84.54 ± 0.79 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   85.27 |   85.3  |   83.33 |   85.25 |   81.7  | 84.17 ± 1.45 |
