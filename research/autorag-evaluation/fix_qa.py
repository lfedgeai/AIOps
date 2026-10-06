import pandas as pd

qa_file = "./data/aiops_qa.parquet"
corpus_file = "./data/aiops_corpus.parquet"
output_file = "./data/aiops_qa_fixed.parquet"

# Read datasets
qa = pd.read_parquet(qa_file)
corpus = pd.read_parquet(corpus_file)

print("QA columns:", qa.columns.tolist())
print("Corpus columns:", corpus.columns.tolist())

# Map corpus content -> doc_id
content_to_id = {
    str(content).strip(): doc_id
    for content, doc_id in zip(corpus["contents"], corpus["doc_id"])
}

# Convert references (document text) -> retrieval_gt (doc_id)
def convert_references(references):
    result = []

    for reference in references:
        reference = str(reference).strip()

        if reference not in content_to_id:
            raise ValueError(
                f"Could not find reference in corpus:\n{reference[:200]}"
            )

        result.append(content_to_id[reference])

    return result


qa["retrieval_gt"] = qa["references"].apply(convert_references)

# Keep only AutoRAG-required columns
qa = qa[
    [
        "qid",
        "query",
        "retrieval_gt",
        "generation_gt",
    ]
]

# Save
qa.to_parquet(output_file, index=False)

print("\nCreated:", output_file)
print("\nFinal columns:")
print(qa.columns.tolist())

print("\nFinal QA dataset:")
print(qa.to_string(index=False))