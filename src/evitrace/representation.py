"""Reusable summaries combined with complete original source lines."""
import collections
import hashlib
import math
import re
from .tokenization import _terms

PROFILE = {'version': 'EviTrace', 'max_rendered_chars': 8000,
           'float_sum': 'math.fsum over lexically sorted terms; score ties by line number',
           'context_lines': 3, 'max_relevance_windows': 24,
           'declaration_reserve_windows': 12, 'full_short_source': True,
           'query_terms': 'NFKC/camel-split requirement terms without generic instructions',
           'score': 'sum log((line_count+1)/(term_line_df+1)) over shared terms, normalized by sqrt(line token count+1), declaration bonus 1.3',
           'source_authoritative': True, 'uses_gold_or_prediction': False}


GENERIC = frozenset('system user users shall should must requirement requirements code source software function functionality implement implementation requested request allows allow using use used possible need needs change new existing support supports description summary can may also example like following would'.split())


DECL = re.compile(r'^\s*(?:(?:public|private|protected|static|final|abstract|virtual|override|sealed|extern|const|volatile|synchronized|native|partial)\s+)+|\b(?:class|interface|enum|struct)\s+[A-Za-z_]|^\s*#\s*(?:define|include)\b')


TYPE_DECL = re.compile(r'\b(?:class|interface|enum|struct)\s+[A-Za-z_]')


COMMENT_ONLY = re.compile(r'^\s*(?://|/\*|\*|\*/)')


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render_evidence(requirement, raw_code):
    if not isinstance(requirement, str) or not isinstance(raw_code, str):
        raise TypeError('Require original requirement and raw code text')
    lines = raw_code.splitlines()
    if not lines:
        raise ValueError('Empty source')
    cap = PROFILE['max_rendered_chars']
    full = '\n'.join(f'L{i+1}: {line}' for i, line in enumerate(lines))
    query = _terms(requirement, 'requirement') - GENERIC
    if len(full) <= cap:
        chosen = set(range(len(lines)))
        mode = 'full_source'
    else:
        terms = [_terms(line, 'code') for line in lines]
        frequency = collections.Counter(t for ts in terms for t in ts)
        scores = []
        declarations = []
        type_headers = []
        for i, line in enumerate(lines):
            shared = terms[i] & query
            value = math.fsum(math.log((len(lines)+1)/(frequency[t]+1))+1 for t in sorted(shared))
            value /= math.sqrt(len(terms[i])+1)
            declaration = bool(DECL.search(line)) and not COMMENT_ONLY.match(line)
            if declaration:
                value *= 1.3
                declarations.append(i)
            if TYPE_DECL.search(line) and not COMMENT_ONLY.match(line):
                type_headers.append(i)
            if value > 0:
                scores.append((value, i))
        scored = sorted(scores, key=lambda x: (-x[0], x[1]))
        # Every selected item is an original complete line. No generated code.
        windows = [(max(0, i-1), min(len(lines), i+2)) for i in type_headers[:4]]
        windows += [(max(0, i-3), min(len(lines), i+4)) for _, i in scored[:24]]
        ranked_declarations = sorted(declarations, key=lambda i: (-len(terms[i] & query), i))
        windows += [(max(0, i-1), min(len(lines), i+2)) for i in ranked_declarations[:12]]
        if declarations:
            for j in range(12):
                i = declarations[min(len(declarations)-1, int((j+.5)*len(declarations)/12))]
                windows.append((i, min(len(lines), i+1)))
        if not windows:
            windows = [(0, min(len(lines), 12))]
        chosen = set()
        def length(indices):
            ordered = sorted(indices)
            gaps = sum(b != a+1 for a, b in zip(ordered, ordered[1:]))
            return sum(len(f'L{i+1}: {lines[i]}')+1 for i in indices) + 42*gaps
        for start, end in windows:
            addition = set(range(start, end))
            if length(chosen | addition) <= cap:
                chosen |= addition
            else:
                for i in sorted(addition, key=lambda j: (-len(terms[j] & query), j)):
                    if length(chosen | {i}) <= cap:
                        chosen.add(i)
        mode = 'selected_complete_source_lines'
    chunks, previous = [], None
    spans = []
    for i in sorted(chosen):
        if previous is not None and i != previous+1:
            chunks.append('[noncontiguous source lines omitted]')
        chunks.append(f'L{i+1}: {lines[i]}')
        if not spans or i+1 != spans[-1][1]+1:
            spans.append([i+1, i+1])
        else:
            spans[-1][1] = i+1
        previous = i
    text = '\n'.join(chunks) or '[No complete source line fits the fixed evidence budget.]'
    assert len(text) <= cap
    metadata = {'profile_version': PROFILE['version'], 'selection_mode': mode,
                'raw_source_sha256': _hash(raw_code), 'requirement_sha256': _hash(requirement),
                'evidence_sha256': _hash(text), 'source_line_count': len(lines),
                'selected_line_count': len(chosen), 'line_spans': spans,
                'rendered_chars': len(text), 'source_complete_in_evidence': len(chosen)==len(lines)}
    for start, end in spans:
        for i in range(start-1, end):
            assert f'L{i+1}: {lines[i]}' in text
    return text, metadata


def augmented_code(requirement, raw_code, summary):
    evidence, metadata = render_evidence(requirement, raw_code)
    text = (summary + '\n\nVERIFIABLE ORIGINAL SOURCE EVIDENCE:\n' + evidence +
            '\n\nEvidence note: The summary is compressed. The quoted original source lines are authoritative for details omitted from the summary.')
    return text, metadata
