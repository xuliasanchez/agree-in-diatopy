from spacy.tokens import Doc
from typing import Iterator, List
from spacy_conll import init_parser
from spacy_conll.parser import ConllParser
import os

from tqdm import tqdm


# Suppress warnings from spacy_conll about weights_only=False
import warnings
warnings.filterwarnings("ignore", message=".*torch.load` with `weights_only=False`.*")

Doc.set_extension("doc_id", default=None)
Doc.set_extension("sent_id", default=None)

def valid_sentence(sentence: Doc) -> bool:
    """
    Check if the sentence is valid for further processing.
    
    Args:
        sentence (Doc): The spaCy Doc object representing the sentence.
    
    Returns:
        bool: True if the sentence is valid, False otherwise.
    """
    # Check if the sentence has a root token and is not empty
    return unique_nsubj_per_verb(sentence)

def unique_nsubj_per_verb(sentence: Doc) -> bool:
    """
    Check that there's no two or more nsubj tokens with the same head.
    Args:
        sentence (Doc): The spaCy Doc object representing the sentence.
    Returns:
        bool: True if each verb has a unique nsubj, False otherwise.
    """
    nsubj_heads = set()  # Set to track unique heads of nsubj tokens
    for token in sentence:
        if token.dep_ == "nsubj":
            if token.head in nsubj_heads:
                return False
            nsubj_heads.add(token.head)
    return True

def _iter_file_sentences(conllu_pth: str) -> Iterator[str]:
    """Yield CoNLL-U sentences (as raw text blocks) without loading the whole file."""
    buf = []
    with open(conllu_pth, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                # End of a sentence block
                if buf:
                    yield "".join(buf)
                    buf = []
            else:
                buf.append(line)
        if buf:
            yield "".join(buf)

def _extract_ids_from_sentence_text(sentence_text: str):
    """Extract doc_id and sent_id from '# sent_id = ##3755605_2'."""
    sent_meta = None
    for line in sentence_text.splitlines():
        if line.startswith("# sent_id ="):
            sent_meta = line.split(" = ", 1)[1].strip()
            break
    if not sent_meta:
        return None, None
    doc_id = sent_meta.split("_")[0].split("-")[0].strip("#")
    sent_id = sent_meta.split("_")[1]
    try:
        sent_id = int(sent_id)
    except Exception:
        pass
    return doc_id, sent_id

class VariantConlluBatcher:
    """
    Create batches of spaCy sentences from all .conllu files of a lenguage variant directory.

    Usage:
        batcher = VariantConlluBatcher(variant_dir, batch_size=1000)
        for docs in batcher.iter_batches():
            # docs is List[Doc] (each Doc is one sentence)
            ...
    """
    def __init__(self, variant_dir: str, batch_size: int = 1000):
        self.variant_dir = variant_dir
        self.batch_size = max(1, batch_size)
        # Prepare parser once
        self.nlp: ConllParser = ConllParser(init_parser("en_core_web_trf", "spacy"))

        # Collect variant .conllu files deterministically
        self.files: List[str] = []
        for root, _, files in os.walk(self.variant_dir):
            for fn in sorted(files):
                if fn.endswith(".conllu"):
                    self.files.append(os.path.join(root, fn))

    def _iter_variant(self) -> Iterator[str]:
        """Stream sentence texts from all variant files."""
        for file in self.files:
            yield from _iter_file_sentences(file)

    def _conllu2spacy_batch(self, sentences_text: List[str]) -> List[Doc]:

        spacy_sentences = []
        for sentence in tqdm(sentences_text, desc="Preparing sentences", unit="sent"):
            doc_id, sent_id = _extract_ids_from_sentence_text(sentence)
            if doc_id is None or sent_id is None:
                continue
            spacy_sentence = self.nlp.parse_conll_text_as_spacy(sentence)

            spacy_sentence._.doc_id = doc_id
            spacy_sentence._.sent_id = sent_id
            if valid_sentence(spacy_sentence):
                spacy_sentences.append(spacy_sentence)

        return spacy_sentences

    def iter_batches(self) -> Iterator[List[Doc]]:
        """
        Yield lists of spaCy sentence Docs with size up to batch_size.
        Shows a tqdm bar for the time spent loading each batch.
        """
        batch_txt: List[str] = []
        pbar = None
        count_in_batch = 0

        for sent_txt in self._iter_variant():
            # Start a new progress bar when a new batch begins
            if count_in_batch == 0:
                pbar = tqdm(
                    total=self.batch_size,
                    desc=f"Loading batch (target {self.batch_size} sent)",
                    unit="sent",
                    leave=True,
                    dynamic_ncols=True,
                )

            batch_txt.append(sent_txt)
            count_in_batch += 1
            if pbar is not None:
                pbar.update(1)

            if count_in_batch >= self.batch_size:
                # Close the bar for this batch and yield
                if pbar is not None:
                    pbar.close()
                    pbar = None
                yield self._conllu2spacy_batch(batch_txt)
                batch_txt = []
                count_in_batch = 0

        # Flush the last (partial) batch, if any
        if batch_txt:
            # Make the bar look "complete" for the partial batch
            if pbar is not None:
                pbar.total = count_in_batch
                pbar.refresh()
                pbar.close()
                pbar = None
            yield self._conllu2spacy_batch(batch_txt)
