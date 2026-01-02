import argparse
import bz2
import json
import os
import pickle
import re
from collections import Counter
from xml.etree import ElementTree

from inverted_index_gcp import InvertedIndex

try:
    import mwparserfromhell as mwp
    mwp.definitions.INVISIBLE_TAGS.append("ref")
    # Strip MediaWiki markup from text when parser is available.
    # Returns plain text for tokenization.
    def _clean_text(text):
        return mwp.parse(text or "").strip_code()
except Exception:
    # Fallback when mwparserfromhell is unavailable.
    # Returns raw text (or empty string) unchanged.
    def _clean_text(text):
        return text or ""

WORD_RE = re.compile(r"[A-Za-z0-9]+")

def simple_tokenize(text):
    # Extract alphanumeric tokens from cleaned text.
    # Returns lowercase tokens for indexing.
    return [t.lower() for t in WORD_RE.findall(text)]


def page_iter(wiki_file):
    # Stream pages from a Wikipedia XML dump.
    # Yields (wiki_id, title, body) for article pages only.
    with bz2.open(wiki_file, "rt", encoding="utf-8", errors="ignore") as f_in:
        elems = (elem for _, elem in ElementTree.iterparse(f_in, events=("end",)))
        elem = next(elems)
        m = re.match(r"^{(http://www\.mediawiki\.org/xml/export-.*?)}", elem.tag)
        if m is None:
            raise ValueError("Malformed MediaWiki dump")
        ns = {"ns": m.group(1)}
        page_tag = ElementTree.QName(ns["ns"], "page").text
        for elem in elems:
            if elem.tag != page_tag:
                continue
            if elem.find("./ns:redirect", ns) is not None or elem.find("./ns:ns", ns).text != "0":
                elem.clear()
                continue
            wiki_id = elem.find("./ns:id", ns).text
            title = elem.find("./ns:title", ns).text
            body = elem.find("./ns:revision/ns:text", ns).text
            yield wiki_id, title, body
            elem.clear()


class FilteredBodyIndexBuilder:
    """Build a body inverted index for only the doc IDs in queries_train.json."""
    def __init__(self, wiki_file, queries_json, out_dir, tokenizer=None):
        self.wiki_file = wiki_file
        self.queries_json = queries_json
        self.out_dir = out_dir
        self.tokenizer = tokenizer or simple_tokenize

    def load_query_doc_ids(self):
        # Load all target wiki IDs from the training queries file.
        # Returns a set of integer wiki IDs.
        """Load all wiki IDs referenced by queries_train.json."""
        with open(self.queries_json, "rt", encoding="utf-8") as f:
            queries = json.load(f)
        ids = set()
        for doc_ids in queries.values():
            for wid in doc_ids:
                try:
                    ids.add(int(wid))
                except ValueError:
                    continue
        return ids

    def build(self):
        # Build an in-memory body index for only the target doc IDs.
        # Returns (index, doc_len, id_title) for later writing.
        """Parse the dump, filter to target IDs, and build the in-memory index."""
        target_ids = self.load_query_doc_ids()
        index = InvertedIndex()
        doc_len = {}
        id_title = {}

        for wiki_id, title, body in page_iter(self.wiki_file):
            try:
                doc_id = int(wiki_id)
            except ValueError:
                continue
            if doc_id not in target_ids:
                continue
            text = _clean_text(body)
            tokens = self.tokenizer(text)
            if not tokens:
                continue
            index.add_doc(doc_id, tokens)
            doc_len[doc_id] = len(tokens)
            id_title[doc_id] = title
        return index, doc_len, id_title

    def write(self, index, doc_len, id_title):
        # Write postings and metadata to disk for later loading.
        # Normalizes posting file paths to be relative to the output dir.
        """Write posting files, index metadata, and doc metadata to disk."""
        os.makedirs(self.out_dir, exist_ok=True)
        list_w_pl = sorted(index._posting_list.items())
        bucket_id = InvertedIndex.write_a_posting_list(("body", list_w_pl), self.out_dir)
        posting_locs_path = os.path.join(self.out_dir, f"{bucket_id}_posting_locs.pickle")
        with open(posting_locs_path, "rb") as f:
            posting_locs = pickle.load(f)
        # Normalize to file basenames so read_a_posting_list doesn't double-join paths.
        for term, locs in posting_locs.items():
            posting_locs[term] = [(os.path.basename(fname), offset) for fname, offset in locs]
        index.posting_locs.update(posting_locs)
        index.write_index(self.out_dir, "body_index")

        with open(os.path.join(self.out_dir, "doc_len.pkl"), "wb") as f:
            pickle.dump(doc_len, f)
        with open(os.path.join(self.out_dir, "id_title.pkl"), "wb") as f:
            pickle.dump(id_title, f)


def main():
    # Parse CLI args and run the build/write pipeline.
    # Prints a short summary when finished.
    parser = argparse.ArgumentParser(description="Build a filtered body index for query doc IDs.")
    parser.add_argument("--wiki", required=True, help="Path to Wikipedia .bz2 dump file")
    parser.add_argument("--queries", required=True, help="Path to queries_train.json")
    parser.add_argument("--out", required=True, help="Output directory for index files")
    args = parser.parse_args()

    builder = FilteredBodyIndexBuilder(args.wiki, args.queries, args.out)
    index, doc_len, id_title = builder.build()
    builder.write(index, doc_len, id_title)
    print(f"Indexed {len(id_title)} docs into {args.out}")


if __name__ == "__main__":
    main()
