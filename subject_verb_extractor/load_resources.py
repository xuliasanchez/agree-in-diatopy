import os
import pandas as pd

from typing import Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_METADATA_DICT = None  # Global variable to hold the document metadata

def _load_doc_metadata() -> Dict[str, Dict[str, str]]:
    """
    Load document metadata from a TSV file into a dict mapping textID to its metadata.

    Returns:
        dict: {
            '3729937': {
                'words': 353,
                'country': 'AU',
                'genre': 'B',
                'URL': 'http://...',
                'title': 'Cheap Oakley...'
            },
            ...
        }
    """
    # Determine the path to the CSV file
    metadata_path = os.path.join(CURRENT_DIR, 'resources', 'glowbe_sources.csv')

    # Read the TSV into a DataFrame
    # - sep='\t' for tab-separated
    # - dtype={'textID': str} to preserve leading zeros if any
    # - encoding='latin-1' to handle special characters
    # - skiprows=[1] to skip the first row as it contains "-----"
    df = pd.read_csv(metadata_path, sep='\t', dtype={'textID': str}, encoding='latin-1', skiprows=[1])

    # Rename the '#words' column to something easier
    if '#words' in df.columns:
        df = df.rename(columns={'#words': 'n_words'})

    if "country genre" in df.columns:
        # Split 'country genre' into two separate columns
        df[['country', 'genre']] = df['country genre'].str.split(' ', n=1, expand=True)
        df = df.drop(columns=['country genre'])

    # Set textID as the DataFrame index
    df = df.set_index('textID')

    # Convert to nested dict: { textID: { col1: val1, ... }, ... }
    metadata = df.to_dict(orient='index')

    return metadata

def get_doc_metadata() -> Dict[str, Dict[str, str]]:
    """
    Get the document metadata, loading it if not already loaded.

    Returns:
        dict: Document metadata mapping textID to its metadata.
    """
    global DOC_METADATA_DICT
    if DOC_METADATA_DICT is None:
        DOC_METADATA_DICT = _load_doc_metadata()
    return DOC_METADATA_DICT