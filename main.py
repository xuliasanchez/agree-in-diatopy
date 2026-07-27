from subject_verb_extractor import analyze_doc, analyze_corpus

import argparse
import os

SAVE_CONLLU = True

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze a document using spaCy.")
    parser.add_argument('-i' , '--input', type=str, help='Path to the input document', required=False)
    parser.add_argument('-c', '--corpus', type=str, help='Path to the input corpus directory', required=False)
    parser.add_argument('-o', '--output', default=os.path.join(os.getcwd(), 'output'), type=str, help='Path to the output directory', required=False)
    parser.add_argument('--columns', default=False, action='store_true', help='If set, output will be in columns format')
    parser.add_argument('-b','--batch-sentences', type=int, default=100000,
                        help='Number of sentences per batch (streamed across files). Default: 100000. Use 0 to process file-by-file.')

    return parser.parse_args()

def main(output_dir: str, 
         doc_path: str = None, 
         corpus_path: str = None, 
         out_format: str = "txt",
         batch_size: int = 100000):
    """
    Main function to analyze a document and print the results.
    
    Args:
        doc_path (str): Path to the document to analyze.
        output_path (str): Path to the output file.
        out_format (str): Format of the output file, either 'txt' or 'csv'. Default is 'txt'.
    """

    if doc_path:
        # Analyze raw text doc
        input_filename = os.path.basename(doc_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}_analysis.{out_format}")

        with open(doc_path, 'r', encoding='utf-8') as file:
            text = file.read()

        analyze_doc(text, output_path, out_format, save_conllu=SAVE_CONLLU)
    elif corpus_path:
        # Analyze a parsed corpus (Conllu-format)
        analyze_corpus(corpus_path, output_dir, batch_size)

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output

    # or input or corpus
    if not(args.input or args.corpus):
        raise ValueError("You must provide either an input file or a corpus directory.")

    out_format = "csv" if args.columns else "txt"

    main(output_dir, args.input, args.corpus, out_format, batch_size=args.batch_sentences)
