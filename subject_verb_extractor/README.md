# Subject-Verb Extractor

This module provides tools for extracting subject-verb info from raw text.

## Files and Functionality

### `controller.py`
Main entry point of the module. It defines the `analyze_doc()` and `analyze_corpus()` functions, which orchestrate the complete processing pipeline. 

- `analyze_doc()` takes a raw text input and an output path, processes the text, and saves the extracted information in the specified format.
- `analyze_corpus()` extracts subject–verb information from pre-computed CONLL-U files.

### `parse.py`
Contains parsing utilities for raw text:
- `spacy_parse()`: Parses text using SpaCy models (**not UD-based**).
- `spacy_stanza_parser()`: Parses text using Stanza and wraps the result into SpaCy `Doc` objects for convenience (**UD**).

Additional parsers can be added here if needed.

### `conllu.py`

Provides all the necessary functionality to read pre-parsed documents in CONLL-U format and initialize SpaCy objects.

### `load_resources.py`

Defines utility functions for accessing external resources. For example, `get_doc_metadata()` loads the metadata file containing information about the corpus, such as title, ID, URL, etc.


### `analyze.py`
Core component of the module. Defines the main function `extract_verb_info()` which performs the extraction of verb-related information from each sentence. Also includes helper functions to support the extraction process.

### `save_results.py`

Provides functions to export the extracted data in different formats. Currently supports plain `.txt` files with row-wise results and `.csv` files with column-wise results.


### `resources/`

This directory contains external resources required by the module, such as `glowbe_sources.csv`, which includes metadata about the corpus.


## How to Use

This module provides two main entry points: `analyze_doc()` and `analyze_corpus()`.

### 1. Analyze a Single Raw Text

To process a raw text input, use the `analyze_doc()` function. It extracts subject–verb information and saves the output in the specified format.

**Arguments:**
- `text`: The raw input text (as a string).
- `output_path`: Path to the directory where results will be saved.
- `output_format`: Output format (currently only `.txt` is supported).

```python
from subject_verb_extractor import analyze_doc

analyze_doc(
    text="Your raw text here",
    output_path="path/to/output",
    output_format="txt"
)
```


### 2. Analyze a Pre-parsed Corpus

If you already have a parsed corpus in CONLL-U format, you can use `analyze_corpus()` to process it.

The function recursively reads all .conllu files in subdirectories of the given input path, processing each variant. This means, the .conllu files should be organized in subdirectories by "English variant" (e.g., `us/`, `au/`, `ie/`, `nz/`, etc.)

The results are generated as `.csv` files using a column-wise format. Each subdirectory in the input directory is treated as a different language variant, so the results are grouped accordingly, producing a separate CSV file for each variant.

**Arguments:**

- `corpus_dir_path`: Path to the root directory containing the CONLL-U files organized in subdirectories (e.g., one per variant).

- `output_dir_path`: Path to the output directory where results will be saved.

```python
from subject_verb_extractor import analyze_corpus

analyze_corpus(
    corpus_dir_path="path/to/conllu_corpus",
    output_dir_path="path/to/save_results"
)
```



