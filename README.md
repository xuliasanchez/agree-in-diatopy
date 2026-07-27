# Verb Extraction

Below, you'll find instructions on how to set up, use, and understand the structure of this project.

---

## Installation

Follow these steps to set up the project environment:

1. **Create a virtual environment. Python 3.10+ supported**
 
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Project Structure

The project is organized as follows:

- **`input/`**  
  Directory containing text files with sentences to analyze.

- **`output/`**  
  Directory where analysis results will be stored.

- **`subject_verb_extractor/`**  
Module responsible for text analysis. Refer to the [README](./subject_verb_extractor/README.md) within this directory for more details.

Note that the `input/` and `output/` directories are convenient but not mandatory. Use `-i` to specify the input file path and `-o` for the output directory.

## Usage

### Running the Analysis

The main entry point for the application is `main.py`. Use the following options to specify input and output paths:

- `-i`  
  Specifies the input **file** containing text to analyze.

- `-o`  
  Specifies the output **directory** where results will be saved.

- `-c`  
  Path to an input corpus directory with conllu files already parsed.

- `--columns`
  Flag to indicate that the output should be generated in a CSV tabular analysis. Note that this is only available for pre-parsed corpus.

- `-b`
  Number of sentences per batch (streamed across files). Default: 100000. Use 0 to process file-by-file.

An analysis file will be generated in the output directory with the same name as the input file, appended with the suffix `_analysis`.

#### Example Command

From the root directory of the project, run:

```bash
python main.py -i ./input/file.txt -o ./output/
```

This command will process `file.txt` from the `input/` directory and generate the analysis results in the `output/` directory as `file_analysis.txt`.


### Analyzing a Pre-parsed Corpus

You can also run the analysis on a pre-parsed corpus in CONLL-U format using the `-c` option. The input directory should contain one subdirectory per language variant (e.g., `us/`, `ie/`, `nz/`, etc.), each containing `.conllu` files.

- `-c`  
  Specifies the input **corpus directory** containing subdirectories with CONLL-U files.
  
- `-o`  
  Specifies the output **directory** where the CSV results will be saved. A separate file will be created for each variant using the format `<variant>_extraction.csv` (e.g., `us_extraction.csv`).

#### Example Command

```bash
python main.py -c ./corpus/ -o ./output/
```

This will process all .conllu files found in the subdirectories of `corpus/` and store the extracted subject–verb information in `output/`, grouped by variant.
