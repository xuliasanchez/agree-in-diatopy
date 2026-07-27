import os
from spacy.tokens import Doc
import pandas as pd

from typing import List

def save_txt(results: list, output_path: str) -> None:
    """
    Save the results in a text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write('COMBINED VERB ANALYSIS:\n\n')
        for sentence_results in results:
            for info in sentence_results:
                f.write(format_text_output(info))
                f.write('\n\n')


def format_text_output(info: dict) -> str:
    """Format the output for a single sentence analysis into a text string.
    Args:
        info (dict): A dictionary containing the analysis information for a sentence.
    Returns:
        str: A formatted string containing the analysis information.
    """
    output_lines = [
        f"\n\nfull_sentence:: {info['sentence']}",
        f"clause:: {info['full_predicate']}",
        f"dependency:: {'MAIN' if info['is_root'] else 'NO_MAIN'}-{info['verb_dep']}",
        f"polarity:: {info['polarity']}",
        f"inversion:: {'EX' if info['is_existential_there'] else 'SV' if info['nsubj_index'] < info['verb_index'] else 'VS'}",
        "\nVERB",
        f"v-form:: {info['verb']}",
        f"v-number:: {info['verb_number']}",
        f"v-person:: {info['verb_person']}",
        f"v-tense:: {info['verb_tense']}",
        "\nSUBJECT",
        f"subj-form:: {'(Elided)' if info['nsubj_elided'] else ''} {info['full_subject']}",
        f"subj-length:: {info['full_subject_non_punct_count']}",  
        f"nsubj-form:: {info['nsubj']}",
        f"nsubj-cat:: {info['subj_pos']}_{info['subj_tag']}_{info['subj_dep']}",
        f"nsubj-number:: {info['subj_number']}",
        f"nsubj-length:: {info['nsubj_non_punct_count']}",
    ]
    
    # pre-nsubj components
    if info['has_pre_nsubj']:
        output_lines.extend([
            f"prensubj-form:: {info['pre_nsubj_text']}",
            f"prensubj-length:: {info['pre_nsubj_non_punct_count']}"
        ])
        for i, component in enumerate(info['pre_nsubj_components'], 1):
            if component['length'] > 0:
                output_lines.extend([
                    f"prensubj_{i}-form:: {component['form']}",
                    f"prensubj_{i}-cat:: {component['cat']}_{component['tag']}_{component['dep']}",
                    f"prensubj_{i}-length:: {component['length']}"
                ])
    
    # post-nsubj components
    if info['has_post_nsubj']:
        output_lines.extend([
            f"postnsubj-form:: {info['post_nsubj_text']}",
            f"postnsubj-length:: {info['post_nsubj_non_punct_count']}"
        ])
        for i, component in enumerate(info['post_nsubj_components'], 1):
            if component['length'] > 0:
                output_lines.extend([
                    f"postnsubj_{i}-form:: {component['form']}",
                    f"postnsubj_{i}-cat:: {component['cat']}_{component['tag']}_{component['dep']}",
                    f"postnsubj_{i}-length:: {component['length']}"
                ])
    
    if info['has_between_subj_verb']:
        output_lines.extend([
            f"\nBETWEEN SUBJECT AND VERB",
            f"between-subj-verb-form:: {info['between_subj_verb_text']}",
            f"between-subj-verb-length:: {info['between_subj_verb_non_punct_count']}"
        ])
        for i, component in enumerate(info['between_subj_verb_components'], 1):
            if component['length'] > 0:
                output_lines.extend([
                    f"between-subj-verb_{i}-form:: {component['form']}",
                    f"between-subj-verb_{i}-cat:: {component['cat']}_{component['tag']}_{component['dep']}",
                    f"between-subj-verb_{i}-length:: {component['length']}"
                ])
    
    return "\n".join(output_lines)


def results_to_dataframe(results):
    """
    Convert a list of sentence-analysis dicts into a pandas DataFrame
    with fixed and variable columns, padding where necessary.
    """

    # 1) Determine the maximum number of components in each variable section
    max_pre   = max((len(info['pre_nsubj_components'])  for info in results), default=0)
    max_post  = max((len(info['post_nsubj_components']) for info in results), default=0)
    max_between = max((len(info['between_subj_verb_components']) for info in results), default=0)

    rows = []
    for info in results:
        row = {}

        # --- Fixed columns ---
        row['doc_id']       = info['doc_id']
        row['sent_id']      = info['sent_id']
        row['url']          = info['url']
        row['title']       = info['title']
        row['genre']      = info['genre']
        row['full_sentence'] = info['sentence']
        row['clause']        = info['full_predicate']
        row['dependency']    = ('MAIN' if info['is_root'] else 'NO_MAIN') + '-' + info['verb_dep']
        row['polarity']      = info['polarity']
        row['inversion']     = ('EX' if info['is_existential_there']
                                else 'SV' if info['nsubj_index'] < info['verb_index']
                                else 'VS')
        row['v-form']   = info['verb']
        row['v-number'] = info['verb_number']
        row['v-person'] = info['verb_person']
        row['v-tense']  = info['verb_tense']

        row['subj-form']    = info['full_subject']
        row['subj-length']  = info['full_subject_non_punct_count']
        row['nsubj-form']   = info['nsubj']
        row['nsubj-cat']    = f"{info['subj_pos']}_{info['subj_tag']}_{info['subj_dep']}"
        row['nsubj-number'] = info['subj_number']
        row['nsubj-length'] = info['nsubj_non_punct_count']
        row['nsubj-elided'] = "yes" if info['nsubj_elided'] else 'no'

        # --- Pre-nsubj section ---
        if info['has_pre_nsubj']:
            row['prensubj-form']   = info['pre_nsubj_text']
            row['prensubj-length'] = info['pre_nsubj_non_punct_count']
        else:
            row['prensubj-form']   = ''
            row['prensubj-length'] = ''

        for i in range(max_pre):
            comp = (info['pre_nsubj_components'][i]
                    if i < len(info['pre_nsubj_components'])
                    else {'form':'', 'cat':'', 'tag':'', 'dep':'', 'length': ''}
                   )
            # combine category, tag and dep for readability
            cat = f"{comp['cat']}_{comp['tag']}_{comp['dep']}" if comp['length'] else ''

            row[f'prensubj_{i+1}-form']   = comp['form']   if comp['length'] else ''
            row[f'prensubj_{i+1}-cat']    = cat
            row[f'prensubj_{i+1}-length'] = comp['length'] if comp['length'] else ''

        # --- Post-nsubj section ---
        if info['has_post_nsubj']:
            row['postnsubj-form']   = info['post_nsubj_text']
            row['postnsubj-length'] = info['post_nsubj_non_punct_count']
        else:
            row['postnsubj-form']   = ''
            row['postnsubj-length'] = ''

        for i in range(max_post):
            comp = (info['post_nsubj_components'][i]
                    if i < len(info['post_nsubj_components'])
                    else {'form':'', 'cat':'', 'tag':'', 'dep':'', 'length': ''}
                   )
            cat = f"{comp['cat']}_{comp['tag']}_{comp['dep']}" if comp['length'] else ''
            row[f'postnsubj_{i+1}-form']   = comp['form']   if comp['length'] else ''
            row[f'postnsubj_{i+1}-cat']    = cat
            row[f'postnsubj_{i+1}-length'] = comp['length'] if comp['length'] else ''

        # --- Between-nsubj-verb section ---
        if info['has_between_subj_verb']:
            row['between-nsubj-verb-form']   = info['between_subj_verb_text']
            row['between-nsubj-verb-length'] = info['between_subj_verb_non_punct_count']
        else:
            row['between-nsubj-verb-form']   = ''
            row['between-nsubj-verb-length'] = ''

        for i in range(max_between):
            comp = (info['between_subj_verb_components'][i]
                    if i < len(info['between_subj_verb_components'])
                    else {'form':'', 'cat':'', 'tag':'', 'dep':'', 'length': ''}
                   )
            cat = f"{comp['cat']}_{comp['tag']}_{comp['dep']}" if comp['length'] else ''
            row[f'between-nsubj-verb_{i+1}-form']   = comp['form']   if comp['length'] else ''
            row[f'between-nsubj-verb_{i+1}-cat']    = cat
            row[f'between-nsubj-verb_{i+1}-length'] = comp['length'] if comp['length'] else ''

        rows.append(row)

    # Build DataFrame and return
    return pd.DataFrame(rows)

def save_results_to_csv(results, csv_path):
    """
    Given a list of analysis dicts, convert to DataFrame and save as CSV.
    """
    df = results_to_dataframe(results)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} sentences into '{csv_path}'")


def save_conllu(doc: List[Doc], output_path: str) -> None:
    """
    Save the results in CoNLL-U format.
    
    Args:
        doc (List[Doc]): Document as a list of sentences (spacy Doc objects).
        output_path (str): The path to save the CoNLL-U file.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sent in enumerate(doc):
            # Optional: Add sentence metadata (CoNLL-U comment lines)
            #f.write(f"# sent_id = {i+1}\n")
            f.write(f"# text = {sent.text}\n")
            
            # Write sentence in CoNLL-U format using spacy_conll
            f.write(sent._.conll_str.strip())  # Remove trailing newline
            f.write('\n\n')  # Separate sentences with blank line
