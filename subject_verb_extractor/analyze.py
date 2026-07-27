import re
from spacy.tokens import Doc, Token

from typing import List, Optional, Set
from .load_resources import get_doc_metadata

# CONSTANTS
NEGATION_TOKENS = {"not"} #  Add more if needed
# Set of deps that can invert the polarity of a verb.
NEGATION_DEPS = {
    # explicit negation
    "neg",
    # adverbial negation in UD v1
    "advmod",
    # negative determiners
    #"det",
    # lexical negation on noun roles
    #"nsubj", "obj", "iobj", "obl"
}

DISALLOWED_SUBJ_DEPS = {'csubj'}  # Disallowed subject dependencies for get_subject function

# MAIN EXTRACTION FUNCTIONS
def extract_verb_info(sentence: Doc, add_metadata = False) -> List[dict]:
    """
    Extract verb information from a parsed sentence.

    Args:
        sentence (spacy.tokens.Doc): The parsed sentence to analyze.

    Returns:
        dict: Dictionary containing the analyzed information.
    """
    
    results = []

    for token in sentence:
        # if not verb or auxiliary verb, continue
        if token.pos_.upper() not in {'VERB', 'AUX'}:
            # Allow AUX PoS for 'To be' as a copula
            continue

        verb = token

        if is_aux(verb):
            # Aux verbs are processed along with their main verb. 
            # So skip independent analysis.
            continue

        # Skip verbs with 2+ auxs
        if count_auxiliaries(verb) >= 2:
            continue

        morph_features = get_morph_features(verb)
        subj = get_subject(verb, disallowed_subj_deps=DISALLOWED_SUBJ_DEPS)

        if not subj or not is_verb_to_process(verb, morph_features):
            # If the verb is not to be processed, skip it
            continue
        
        polarity = get_verb_polarity(verb)


        if is_there_construction(verb):
            result = process_construction(
                verb, subj, sentence, 
                morph_features, 
                is_existential=True, 
                polarity=polarity
                )

        else: # Regular or copula verb
            result = process_construction(
                verb, subj, sentence, 
                morph_features, polarity=polarity
                )
        
        if result:
            if add_metadata:
                result = add_metadata_to_results(result, sentence)
                
            results.append(result)

    return results

def process_construction(verb: Token, 
                         subject: Token, 
                         sentence: Doc, 
                         verb_morph_features: dict,
                         is_existential=False, 
                         polarity: str = ""
                        ):
    """ Process the verb and its subject to extract relevant information.

    Args:
        verb (spacy.tokens.Token): The verb token to analyze.
        subject (spacy.tokens.Token): The nsubj token to analyze.
        sentence (spacy.tokens.Doc): The entire sentence.
        verb_morph_features (dict): Morphological features of the verb.
        is_existential (bool): If True, the construction is existential.
        is_conjunct (bool): If True, the construction is conjunct.

    Returns:
        dict: A dictionary containing the processed information.
    """

    def count_non_punct(tokens: List[Token]) -> int:
        return sum(1 for t in tokens if t.dep_ != "punct")

    # Capture compound nsubj
    full_nsubj = get_compound_nsubj(subject)
    full_nsubj_text = ''.join(t.text_with_ws for t in full_nsubj).strip()
    nsubj_start = min(t.i for t in full_nsubj)
    nsubj_end = max(t.i for t in full_nsubj) 
    
    # Verb phrase
    verb_phrase = get_verb_phrase(verb)
    verb_phrase_str = ' '.join(t.text for t in verb_phrase)
    finite_anchor = get_finite_anchor(verb)

    between_subj_verb, between_components = get_between_subj_verb(verb, subject, sentence)

    # morph features
    verb_number = verb_morph_features['verb_number']
    verb_person = verb_morph_features['verb_person']
    verb_tense = verb_morph_features['verb_tense']

    # Get subject features
    pre_nsubj = [c for c in subject.subtree if c.i < nsubj_start]
    post_nsubj = [c for c in subject.subtree if c.i > nsubj_end]
    post_nsubj = collapse_trailing_punct(post_nsubj) # Collapse trailing punctuation (Avoid "The cat, ,")

    subj_number = subject.morph.get('Number', [''])[0]
    
    full_subject = ''.join(t.text_with_ws for t in pre_nsubj + full_nsubj + post_nsubj).strip()

    # Remove pre and post if they contain only punctuation
    pre_nsubj = [] if only_punct(pre_nsubj) else pre_nsubj
    post_nsubj = [] if only_punct(post_nsubj) else post_nsubj
    
    pre_nsubj_components = get_components(pre_nsubj)
    post_nsubj_components = get_components(post_nsubj)
    
    # Predicate
    full_predicate = get_full_predicate(verb)

    # Verb dependency
    verb_dep = verb.dep_ if verb.dep_ != 'cop' else get_cop_dep(verb, subject)  # Copula verbs have 'cop' dep, but we want to use the head's dep
    if not verb_dep:
        return None # Better to skip
    
    # Check subject elision
    is_elided = elided_subject(subject, verb, verb_dep)
    
    return {
        'verb': verb_phrase_str,
        'verb_token': verb,
        'verb_number': verb_number,
        'verb_person': verb_person,
        'verb_tense': verb_tense,
        'nsubj': full_nsubj_text,
        'subject_token': subject,
        'full_subject': full_subject,
        'pre_nsubj_text': ''.join(t.text_with_ws for t in pre_nsubj),
        'post_nsubj_text': ''.join(t.text_with_ws for t in post_nsubj),
        'pre_nsubj_components': pre_nsubj_components,
        'post_nsubj_components': post_nsubj_components,
        'sentence': sentence.text if hasattr(sentence, 'text') else str(sentence),
        'subj_pos': subject.pos_,
        'subj_tag': subject.tag_,
        'subj_dep': subject.dep_,
        'subj_number': subj_number,
        'is_root': verb_dep.upper() == "ROOT",
        'pre_nsubj_non_punct_count': count_non_punct(pre_nsubj),
        'post_nsubj_non_punct_count': count_non_punct(post_nsubj),
        'nsubj_non_punct_count': count_non_punct(full_nsubj),
        'full_subject_non_punct_count': count_non_punct(pre_nsubj + full_nsubj + post_nsubj),
        'verb_dep': verb_dep,
        'full_predicate': full_predicate,
        'between_subj_verb_text': ''.join(t.text_with_ws for t in between_subj_verb),
        'between_subj_verb_components': between_components,
        'between_subj_verb_non_punct_count': count_non_punct(between_subj_verb),
        'has_pre_nsubj': len(pre_nsubj) > 0,
        'has_post_nsubj': len(post_nsubj) > 0,
        'has_between_subj_verb': len(between_subj_verb) > 0,
        'is_existential_there': is_existential,
        'polarity': polarity,
        'verb_index': finite_anchor.i,
        'verb_token': finite_anchor,
        'lexical_verb_index': verb.i,
        'lexical_verb_token': verb,
        'nsubj_index': subject.i,
        'nsubj_elided': is_elided,
    }

def add_metadata_to_results(result: dict, sentence: Doc) -> dict:

    """ Add metadata to the result dictionary.
    
    Args:
        result (dict): The result dictionary to update.
        sentence (spacy.tokens.Doc): The sentence containing the tokens.
    
    Returns:
        dict: The updated result dictionary with metadata.
    """
    metadata_dict = get_doc_metadata()
    doc_id = sentence._.doc_id if hasattr(sentence, '_') and hasattr(sentence._, 'doc_id') else None

    if doc_id:
        sent_id = sentence._.sent_id if hasattr(sentence, '_') and hasattr(sentence._, 'sent_id') else ''
        url = metadata_dict.get(doc_id, {}).get('URL', '')
        title = metadata_dict.get(doc_id, {}).get('title', '')
        genre = metadata_dict.get(doc_id, {}).get('genre', '')
    else:
        # If no doc_id is found, set default values
        sent_id = ''
        url = ''
        title = ''
        genre = ''

    result['doc_id'] = doc_id
    result['sent_id'] = sent_id
    result['url'] = url
    result['title'] = title
    result['genre'] = genre

    return result

# UTILITY FUNCTIONS

def is_there_construction(verb: Token) -> bool:
    """
    Check if the verb is part of a 'there' construction (existential).
    Example: "There is a cat on the roof."

    Args:
        verb (spacy.tokens.Token): The verb to check.

    Returns:
        bool: True if the verb is part of a 'there' construction, False otherwise.
    """
    # Check if the verb has an 'expl' child with the text 'there'
    if any([child for child in verb.children if child.dep_ == 'expl' and child.lower_ == 'there']):
        return True

def count_auxiliaries(verb: Token) -> int:
    """ Count the number of auxiliaries associated with the verb.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.

    Returns:
        int: The number of auxiliaries associated with the verb.
    """

    aux_list = get_auxiliaries(verb)
    return len(aux_list)

def get_finite_anchor(verb: Token) -> Token:
    """ Return the finite verb that anchors subject-verb order.

    Prefer the leftmost finite token in the verb phrase (typically an auxiliary
    like 'does' or 'are'). If no finite token is available, fall back to the
    leftmost token in the verb phrase.
    """
    verb_phrase = get_verb_phrase(verb, include_negation=False)
    finite_candidates = [t for t in verb_phrase if "VerbForm=Fin" in t.morph]

    if finite_candidates:
        return min(finite_candidates, key=lambda t: t.i)

    return min(verb_phrase, key=lambda t: t.i)

def get_morph_features(verb: Token) -> dict:
    """ Extract morph features from the verb or its auxiliaries.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.
    Returns:
        dict: Dictionary containing the morph features of the verb.
    """

    if "VerbForm=Fin" in verb.morph:
        # morph features
        verb_number = verb.morph.get('Number', [''])[0]
        verb_person = verb.morph.get('Person', [''])[0]
        verb_tense = verb.morph.get('Tense', [''])[0]
        return {
            'verb_number': verb_number,
            'verb_person': verb_person,
            'verb_tense': verb_tense
        }

    # Look for morph features in auxiliary verbs
    auxiliary_verbs = get_auxiliaries(verb)
    if auxiliary_verbs:
        for aux in auxiliary_verbs:
            if "VerbForm=Fin" in aux.morph:
                verb_number = aux.morph.get('Number', [''])[0]
                verb_person = aux.morph.get('Person', [''])[0]
                verb_tense = aux.morph.get('Tense', [''])[0]
                return {
                    'verb_number': verb_number,
                    'verb_person': verb_person,
                    'verb_tense': verb_tense
                }
            
    # If no morph features found, return empty dict
    return {
        'verb_number': '',
        'verb_person': '',
        'verb_tense': ''
    }

def is_verb_to_process(verb: Token, morph_features: dict) -> bool:
    """ Check if the verb is to be processed based on its morph features.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.
        morph_features (dict): Dictionary containing the morph features of the verb. Namely:
            - verb_person: Person of the verb (1, 2, 3)
            - verb_tense: Tense of the verb (Pres, Past, Fut)
    Returns:
        bool: True if the verb is to be processed, False otherwise.
    """
    verb_person = morph_features['verb_person']
    verb_tense = morph_features['verb_tense']

    if not(verb_person and verb_tense):
        # If some features are missing, we cannot determine if the verb is to be processed
        return False

    # Check 3rd person
    if verb_person in {'1', '2'}:
        return False
    
    # Check allowed verbs: to be or a to be aux, or a present tense verb
    auxiliary_verbs = get_auxiliaries(verb)
    if verb.lemma_ in ['be'] or \
            any(aux for aux in auxiliary_verbs if aux.lemma_ == 'be') or \
            (verb.pos_ in {'VERB', 'AUX'} and verb_tense == "Pres"):
       return True
    
    return False

def is_aux(verb: Token) -> bool:
    """ Check if the verb is a valid auxiliary verb.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.
    """
    aux_forms = {
        'be': {'is', 'are', 'was', 'were', "'re", "'s"},
        'have': {'has', 'have', "'ve"},
        'get': {'get', 'gets'},
        'do': {'do', 'does'}
    }

    return (
        verb.dep_ in {'aux', 'aux:pass'} and
        verb.lemma_ in aux_forms #and
        #verb.text.lower() in aux_forms[verb.lemma_]
    )

def get_auxiliaries(verb: Token) -> List[Token]:
    """ Get the auxiliaries of the verb.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.
    Returns:
        List[Token]: List of auxiliary tokens.
    """

    vp_head = verb if verb.dep_ != 'cop' else verb.head  # For copula verbs, the head of the Verbal Phrase is the head of the copula
    return [child for child in vp_head.children if is_aux(child) and child.i < verb.i] # Aux should be before its main verb.

def get_subject(
    verb: Token,
    orig: Optional[Token] = None,
    disallowed_subj_deps: Optional[Set[str]] = {}
) -> Optional[Token]:
    """
    Recursively find the nsubj of a verb token.

    If multiple subject candidates are present (e.g., in copular clauses with nsubj:outer),
    align them with other child tokens requiring subj. subjs are assigned based on their order.

    Only allowed subj dependencies are returned, disallowed ones are ignored.

    Args:
        verb (Token): The verb token to analyze.
        orig (Optional[Token]): The original verb token being analyzed (used for alignment).

    Returns:
        Optional[Token]: The subject token if found, otherwise None.
    """

    def _align_subject(
        head: Token,
        candidates: list[Token],
        orig: Token
    ) -> Optional[Token]:
        """
        Align subject candidates with tokens requiring nsubj based on linear position.

        1. Sort both candidate subjects and verb carndidates by their position in the sentence.
        2. If the number of subjects and verbs match, return the subject aligned
           with the original verb's position.
        3. If alignment is not possible, return None.
        """
        # Sort copular verbs (children with dep_ == "cop") and include the head if it's a VERB
        # conj and aux tokens share the same nsubj as head, so no add them. Only copulas may have different nsubj.
        verbs = [c for c in head.children if c.dep_ == "cop"]
        if head.pos_ == "VERB":
            verbs.append(head)
        verbs_list = sorted(verbs, key=lambda v: v.i)

        # Ensure subject candidates are also sorted by position
        cand_sorted = sorted(candidates, key=lambda c: c.i)

        # Align based on position
        if len(verbs_list) == len(cand_sorted):
            try:
                idx = verbs_list.index(orig)
                return cand_sorted[idx]
            except ValueError:
                return None

        return None

    if orig is None:
        orig = verb

    # Collect all direct subject candidates
    subj_pattern = re.compile(r'^(?:[nc]subj)(?::.*)?$')
    candidates = [c for c in verb.children if subj_pattern.match(c.dep_)]

    # Look for it-extraposition
    #it_extraposition = []
    #if not any(candidate for candidate in candidates if candidate.dep_ == 'nsubj'):
        # If no nsubj candidates, look for it-extraposition
        # e.g. "It is clear that she lied."
        # In this case, we consider the 'it' as a subject candidate
        # to align with the verb.
    #    it_extraposition = [c for c in verb.children if c.dep_ == 'expl' and c.lower_ == 'it']

    #if it_extraposition:
        # remove "csubj" from candidates to avoid disaalignment (e.g. "The problem is that it is clear that she lied.")
    #    candidates = [c for c in candidates if not c.dep_.startswith('csubj')]
    #    candidates.extend(it_extraposition)

    if candidates:
        # If only one candidate, return it directly
        if len(candidates) == 1:
            subj = candidates[0]
            return subj if subj and not any(subj.dep_.startswith(disallowed_dep) for disallowed_dep in disallowed_subj_deps) else None

        # If multiple candidates, attempt alignment
        aligned = _align_subject(verb, candidates, orig)
        # Return if allowed, otherwise None
        return aligned if aligned and not any(aligned.dep_.startswith(disallowed_dep) for disallowed_dep in disallowed_subj_deps) else None

    # If no subject candidates are found, recursively move up through functional dependencies
    if verb.dep_ in ("cop", "aux", "aux:pass", "conj"):
        return get_subject(verb.head, orig=orig, disallowed_subj_deps=disallowed_subj_deps)

    return None

def elided_subject(nsubj: Token, verb: Token, dependency: str) -> bool:
    """
    Check if the verb has an elided subject.
    The subject is considered elided if isn a child of the verb and the verb's dep is 'conj'
    Args:
        nsubj (spacy.tokens.Token): The subject token to analyze.
        verb (spacy.tokens.Token): The verb token to analyze.
        dependency (str): The dependency type of the verb.
    Returns:
        bool: True if the verb has an elided subject, False otherwise.
    """
    token = verb if verb.dep_ != 'cop' else verb.head # Copula verbs have 'cop' dep, but we want to use the head's dep

    if  verb.dep_ == 'cop':
        if verb.head.dep_ == 'conj':
            # The parse tree indicates directly indicates coordination.
            return nsubj not in verb.head.children
        elif dependency == 'conj':
            # Parse tree dont indicate coordination. But is considered coordination.
            # (e.g. "The problem is and always was serious.")
            sibling_copulas = [c for c in verb.head.children if c.dep_ == 'cop' and c != verb]
            # Get the first copula, which is the head of the coordination
            sibling_copulas.sort(key=lambda c: c.i)
            head_copula = sibling_copulas[0]
            head_subject = get_subject(head_copula)
            if head_subject:
                if head_subject.i == nsubj.i:
                    # If the head subject is the same as the nsubj, it is elided
                    return True
            return False
        
    elif dependency == 'conj':
        # Check if the subject is a child of the token
        return nsubj not in token.children

    return False

def get_verb_polarity(verb: Token) -> str:
    """ Get the polarity of the verb based on its children tokens.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.
    Returns:
        str: Polarity of the verb ('Positive', 'Negative')."""
    def get_token_polarity(token: Token) -> str:
        """
        Get the polarity of a token.
        Args:
            token (spacy.tokens.Token): The token to analyze.
        Returns:
            str: Polarity of the token ('Positive', 'Negative', 'Neutral').
        """
        
        polarity = token.morph.get('Polarity', [''])[0]
        pronType = token.morph.get('PronType', [''])[0]

        if polarity == 'Neg' or pronType == 'Neg' or token.lemma_ in NEGATION_TOKENS:
            return 'Negative'
        elif polarity == 'Pos' or pronType == 'Pos':
            return 'Positive'
        else:
            return 'Neutral'

    if verb.dep_ != 'cop':
        if any(child for child in verb.children if child.dep_ in NEGATION_DEPS and get_token_polarity(child) == 'Negative'):
            return 'Negative'
        else:
            return 'Positive'
    else:
        head = verb.head

        sibling_copulas = [c for c in head.children if c.dep_ == 'cop' and c != verb]
        verb_candidates = [verb] + sibling_copulas

        if head.pos_ == 'VERB':
            # If the head is a verb, is also a candidate for negation
            verb_candidates.append(head)

        # sort candidates by id (position)
        verb_candidates.sort(key=lambda c: c.i)

        negation_candidates = [c for c in head.children if c.dep_ in NEGATION_DEPS and get_token_polarity(c) == 'Negative']

        # Assing each negation candidate to the closest copula candidate
        for neg in negation_candidates:
            closest_copula = min(verb_candidates, key=lambda c: abs(c.i - neg.i))
            if closest_copula.i == verb.i:
                return 'Negative'
        
        # No negation candidate found
        return 'Positive'
    
def get_verb_phrase(verb: Token, include_negation: Token = False) -> List[Token]:
    """
    Get the verb phrase of the verb, including auxiliaries.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.
    Returns:
        List[Token]: List of tokens in the verb phrase.
    """
    verb_phrase = get_auxiliaries(verb)
    verb_phrase.append(verb)

    if include_negation:
        if verb.dep_ != 'cop':
            negation_tokens = [c for c in verb.children if c.lemma_ in NEGATION_TOKENS and c.dep_ in NEGATION_DEPS]
            verb_phrase = negation_tokens + verb_phrase
        else:
            head = verb.head
            negation_candidates = [c for c in head.children if c.lemma_ in NEGATION_TOKENS and c.dep_ in NEGATION_DEPS]

            sibling_copulas = [c for c in head.children if c.dep_ == 'cop' and c != verb]
            verb_candidates = [verb] + sibling_copulas

            negation_tokens = []
            # Assign each negation candidate to the closest verb candidate
            for neg in negation_candidates:
                closest_verb = min(verb_candidates, key=lambda c: abs(c.i - neg.i))
                if closest_verb.i == verb.i:
                    negation_tokens.append(neg)
            verb_phrase = negation_tokens + verb_phrase

    verb_phrase.sort(key=lambda x: x.i)

    return verb_phrase

def get_compound_nsubj(subject: Token) -> List[Token]:
    """
    Get the compound subject of the subject token.
    
    Args:
        subject (spacy.tokens.Token): The subject token to analyze.
    
    Returns:
        List[Token]: List of tokens in the compound subject.
    """
    compound_nsubj = [subject]
    for child in subject.children:
        if child.dep_ in {'compound', 'flat'}:
            # extend subtree
            compound_nsubj.extend(list(child.subtree))
        elif child.dep_ == 'punct' and child.text == '-':
            compound_nsubj.append(child)

    compound_nsubj.sort(key=lambda x: x.i)
    return compound_nsubj

def get_full_predicate(verb: Token) -> str:
    """ Get the full predicate of the verb, including auxiliaries and its children.
    Args:
        verb (spacy.tokens.Token): The verb to analyze.
    Returns:
        str: The full predicate as a string.
    """
    root = verb.head if verb.dep_ == 'cop' else verb
    predicate_tokens = [root]
    for child in root.children:
        predicate_tokens.extend(list(child.subtree))
    predicate_tokens = sorted({token.i: token for token in predicate_tokens}.values(), key=lambda x: x.i)
    return ''.join(token.text_with_ws for token in predicate_tokens)

def get_between_subj_verb(verb: Token, subject: Token, sentence: Doc) -> List[Token]:
    """
    Get the tokens between the subject and the verb.

    Args:
        verb (spacy.tokens.Token): The verb to analyze.
        subject (spacy.tokens.Token): The subject of the verb.
        sentence (spacy.tokens.Doc): The sentence containing the tokens.

    Returns:
        List[Token]: List of tokens between the subject and the verb.
    """
    subject_tokens = list(subject.subtree)
    nsubj_start = min(t.i for t in subject_tokens if t.dep_ != "punct")
    nsubj_end = max(t.i for t in subject_tokens if t.dep_ != "punct")
    verb_phrase = get_verb_phrase(verb, include_negation=True)
    verb_start = min(t.i for t in verb_phrase)

    vp_head = verb if verb.dep_ != 'cop' else verb.head # For copula verbs, the head of the Verbal Phrase is the head of the copula

    between_subj_verb = []
    if nsubj_end < verb_start:  # SV
        between_subj_verb = [
            t for t in sentence 
            if nsubj_end < t.i < verb_start
            and t in list(vp_head.subtree)
            and not (t.head == vp_head and t.dep_ in {'cc'}) # If verb is coordinated, skip the conjunction
            and t.pos_ not in {'PUNCT', 'SYM', 'X'} # Skip punctuation, symbols and other non-content words
        ]
    elif verb_start < nsubj_start:  # VS
        verb_end = max(t.i for t in verb_phrase)
        between_subj_verb = [
            t for t in sentence 
            if verb_end < t.i < nsubj_start
            and t in list(vp_head.subtree)
            and not (t.head.head == vp_head and t.dep_ in {'cc'}) # If verb is coordinated, skip the conjunction. cc depends on the last conjunct, therefore we check t.head.head (e.g Laughed and cried the little girl)
            and t.pos_ not in {'PUNCT', 'SYM', 'X'} # Skip punctuation, symbols and other non-content words
        ]
    

    between_components = get_components(between_subj_verb)

    return between_subj_verb, between_components

def get_components(sequence: List[Token]) -> List[dict]:
    """ Get the components of a sequence of tokens. Where a component is a dictionary
    containing the form, category, tag, dependency and length of each subtree.
    Args:
        sequence (List[Token]): The sequence of tokens to analyze.
    Returns:
        List[dict]: List of dictionaries containing the components.
    """
    components = []
    sequence_ids = {t.i for t in sequence}
    # Get the root nodes from the sequence (i.e. the tokens that are not dependents of other tokens in the sequence)
    roots = [t for t in sequence if t.head.i not in sequence_ids]

    for token in roots:
        subtree_tokens = list(token.subtree)
        components.append({
            'form': ''.join([t.text_with_ws for t in subtree_tokens]),
            'cat': token.pos_,
            'tag': token.tag_,
            'dep': token.dep_,
            'length': sum(1 for t in subtree_tokens if t.dep_ != "punct")
        })
    return components

def collapse_trailing_punct(tokens: List[Token]) -> List[Token]:
    """ Collapse trailing punctuation tokens in a list of tokens.
    This function ensures that only the last punctuation token is kept, if any.
    Args:
        tokens (List[Token]): List of tokens to process.
    Returns:
        List[Token]: List of tokens with trailing punctuation collapsed.
    """

    # sort tokens by index
    tokens = sorted(tokens, key=lambda t: t.i)

    # Find the split point: last non-punct token
    last_non_punct = len(tokens) - 1
    while last_non_punct >= 0 and tokens[last_non_punct].dep_ == "punct":
        last_non_punct -= 1

    # Leading tokens (including the last non-punct)
    leading = tokens[: last_non_punct + 1]
    # All trailing punct tokens
    trailing = tokens[last_non_punct + 1 :]

    # Keep at most one punctuation
    if trailing:
        trailing = [trailing[0]]

    return leading + trailing

def only_punct(tokens: List[Token]) -> bool:
    """ Check if the list of tokens contains only punctuation tokens.
    Args:
        tokens (List[Token]): List of tokens to check.
    Returns:
        bool: True if the list contains only punctuation tokens, False otherwise.
    """
    return all(t.dep_ == "punct" for t in tokens)

def get_cop_dep(cop_verb: Token, subj: Token) -> str:
    """
    Identifies the dependency of a copula verb.  Copula generally will inherit the dependency of its head verb.
    However, if the head verb is the Root, and have several dependant copulas, only one can inherit the root dependency.""
    Args:
        cop_verb (Token): The copula verb token.
        subj (Token): The subject token.
    """
    cop_siblings = [c for c in cop_verb.head.children if c.dep_ == 'cop' and c != cop_verb]

    # Filter Inf copulas
    fin_cop_siblings =  [c for c in cop_siblings if c.morph.get('VerbForm', [''])[0] != 'Inf']

    if not fin_cop_siblings:
        if cop_verb.head.pos_ == 'VERB':
            return cop_verb.dep_
        else:
            return cop_verb.head.dep_
    else:

        # Check if some subject has "outer" dep
        sibling_subjs = [get_subject(sibling) for sibling in cop_siblings] # Allow csubj to consider csubj:outer
        all_subjects = sibling_subjs + [subj]
        outer_copulas = [s for s in all_subjects if s and s.dep_.endswith('outer')]

        if outer_copulas:
            outer_copulas.sort(key=lambda c: c.i)
            if outer_copulas[0].i == subj.i:
                # If the outer subj is the subj of the current copula, it can be root. Return its head's dep
                return cop_verb.head.dep_
            else:
                return cop_verb.dep_

        else:
            # might be a coordinated structure.
            # Check if all subjects are the same
            subj_ids = {s.i for s in all_subjects if s}

            if len(subj_ids) == 1:
                # Set as root the first copula verb
                copulas = cop_siblings + [cop_verb]
                copulas.sort(key=lambda c: c.i)
                if copulas[0].i == cop_verb.i:
                    return cop_verb.head.dep_
                else:
                    return "conj"
            else:
                # If subjects are different, return the copula's dep
                return None
