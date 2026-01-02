import json
import os
import pickle
from inverted_index_gcp import InvertedIndex

out_dir = "index_filtered"

needed = ["body_index.pkl", "doc_len.pkl", "id_title.pkl", "body_posting_locs.pickle"]
missing = [f for f in needed if not os.path.exists(os.path.join(out_dir, f))]
if missing:
    raise SystemExit(f"Missing index files: {missing}")

with open("queries_train.json", "rt", encoding="utf-8") as f:
    queries = json.load(f)

with open(os.path.join(out_dir, "doc_len.pkl"), "rb") as f:
    doc_len = pickle.load(f)
with open(os.path.join(out_dir, "id_title.pkl"), "rb") as f:
    id_title = pickle.load(f)

target_ids = set()
for ids in queries.values():
    for wid in ids:
        try:
            target_ids.add(int(wid))
        except ValueError:
            pass

indexed_ids = set(id_title.keys())
print("Target IDs:", len(target_ids))
print("Indexed IDs:", len(indexed_ids))
print("Indexed ∩ Target:", len(indexed_ids & target_ids))

index = InvertedIndex.read_index(out_dir, "body_index")
print("Index terms:", len(index.df))

term = next(iter(index.df.keys()))
postings = index.read_a_posting_list(out_dir, term)
print("Sample term:", term, "postings:", len(postings))

print("OK")
