import gc
from .parse import spacy_stanza_parse
from .analyze import extract_verb_info
from .save_results import save_txt, save_results_to_csv
from .conllu import VariantConlluBatcher
import os
from tqdm import tqdm

from . import save_results as sr

from typing import List

def analyze_doc(text: str, output_path: str, out_format: str = "txt", save_conllu=False) -> None:
    """
    Analyze the text and saves the results to a file in the specified format (i.e. txt/csv).

    Args:
        text (str): The text to analyze.
        output_path (str): The path to save the output file.
        out_format (str): The format of the output file. Default is "txt".
        save_conllu (bool): If True, saves the parsed document in CoNLL-U format.

    """
    doc = spacy_stanza_parse(text)
    if save_conllu:
        conllu_path = output_path.replace('.txt', '.conllu')
        sr.save_conllu(doc, conllu_path)

    doc_results = []
    for sentence in doc:
        sentence_results = extract_verb_info(sentence)
        if sentence_results:
            doc_results.append(sentence_results)
    if doc_results:
        if out_format == "txt":
            save_txt(doc_results, output_path)
        else:
            raise ValueError(f"Unsupported format: {out_format}. Supported formats are: txt.")
        

def analyze_corpus(corpus_dir_path: str, output_dir_path: str, batch_size: int) -> None:
    # Check if the corpus directory exists
    if not os.path.exists(corpus_dir_path):
        raise FileNotFoundError(f"The corpus directory '{corpus_dir_path}' does not exist.")
    # Read each subdirectory.
    for subdir, dirs, files in os.walk(corpus_dir_path):
        en_variant = subdir.split(os.path.sep)[-1]  # Get the last part of the path as the variant name
        # Process each file in the subdirectory
        if files:
            n_files = len(files)
            print(f"\n\tReading variant {en_variant.upper()} with {n_files} files\n")
            # Create a batcher for the variant
            batcher = VariantConlluBatcher(subdir, batch_size=batch_size)
            # Analyze the variant in batches
            analyze_variant(en_variant, output_dir_path, batcher)


def analyze_variant(en_variant: str,
                            output_dir_path: str,
                            batcher: VariantConlluBatcher) -> None:
    """
    Iterate through batches produced by `batcher`. For each batch:
      - run `extract_verb_info` per sentence (with add_metadata=True)
      - accumulate results
      - save as CSV to: <output_dir>/<variant>/<variant>_extraction_<N>.csv

    Progress is shown with tqdm over sentences in each batch.
    """
    # Create variant output directory: <output_dir>/<variant>/
    variant_out_dir = os.path.join(output_dir_path, en_variant)
    os.makedirs(variant_out_dir, exist_ok=True)

    total_rows = 0
    for batch_idx, docs in enumerate(batcher.iter_batches()):
        # Process sentences in this batch with progress bar
        batch_results: List[dict] = []
        for sent in tqdm(docs, desc=f"{en_variant.upper()} | Analyzing Batch {batch_idx}", unit="sent"):
            try:
                sentence_results = extract_verb_info(sent, add_metadata=True)
            except Exception as e:
                # Avoid crashing on a single sentence error
                print(f"Error processing sentence {sent._.sent_id} in doc {sent._.doc_id}: {e}")
                continue
            if sentence_results:
                # `extract_verb_info` returns a list of rows (dicts); extend the flat list
                batch_results.extend(sentence_results)

        if not batch_results:
            # Nothing to save for this batch
            # Free memory for this batch before continuing
            del docs
            del batch_results
            gc.collect()
            continue

        # Build output CSV path with incremental index
        try:
            csv_path = os.path.join(variant_out_dir, f"{en_variant}_extraction_{batch_idx}.csv")
            save_results_to_csv(batch_results, csv_path)
            total_rows += len(batch_results)
        except Exception as e:
            print(f"Error saving results for variant {en_variant} batch {batch_idx}: {e}")
            continue
            
        # Explicitly free memory for this batch
        del docs
        del batch_results
        gc.collect()

    print(f"Variant {en_variant.upper()} finished. Total rows saved: {total_rows}")
