"""Classical retrieval models for project-wide classification and candidate ranking."""
import math
import random
import re
import string
import unicodedata
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from .io import canonical, sha, PROJECTS, Artifacts, read_jsonl, write_json
METHODS = ("VSM", "BM25", "LSI")

def tokenize_public(text, dataset, kind):
    """Literal removed-branch tokenizer, with correct original text decoding."""
    import nltk
    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))
    specific = {dataset, dataset.lower()}
    words = []
    for line in text.splitlines():
        tokens = nltk.word_tokenize(line)
        if kind == 'cc':
            tokens = [word for token in tokens for fragment in re.split(r'[./]', token)
                      for word in re.findall(r'[A-Z]?[a-z]+', fragment)]
        words.extend(token for token in tokens
                     if not any(char in string.punctuation for char in token)
                     and token.lower() not in stops and token not in specific)
    return words

def fit_public_scores(requirements, codes):
    """Only text tokens are accepted here; target rows and labels are inaccessible."""
    import numpy as np
    from gensim import corpora, models, similarities
    np.random.seed(0)
    random.seed(0)
    dictionary = corpora.Dictionary(requirements)
    req_bow = [dictionary.doc2bow(doc) for doc in requirements]
    code_bow = [dictionary.doc2bow(doc) for doc in codes]
    assert len(dictionary)
    # Public VSM uses a requirement dictionary but fits IDF on available code.
    code_tfidf = models.TfidfModel(code_bow)
    code_vectors = code_tfidf[code_bow]
    vsm_index = similarities.MatrixSimilarity(code_vectors, num_features=len(dictionary))
    vsm = np.asarray([vsm_index[code_tfidf[doc]] for doc in req_bow], dtype=np.float64)
    # Public LSI fits TF-IDF and SVD on requirements, then transforms code.
    req_tfidf = models.TfidfModel(req_bow)
    req_vectors = req_tfidf[req_bow]
    lsi = models.LsiModel(req_vectors, id2word=dictionary, num_topics=200, random_seed=0)
    req_latent = lsi[req_vectors]
    lsi_index = similarities.MatrixSimilarity(req_latent, num_features=lsi.num_topics)
    lsi_matrix = np.asarray(lsi_index[lsi[req_tfidf[code_bow]]], dtype=np.float64).T
    # Default sparse >=.01 topics are retained, matching upstream construction.
    lda = models.LdaModel(req_bow, id2word=dictionary, num_topics=100, random_state=0)
    req_topics = list(lda.get_document_topics(req_bow))
    code_topics = list(lda.get_document_topics(code_bow))
    def topic_matrix(documents):
        matrix = np.zeros((len(documents), 100), dtype=np.float64)
        for i, topics in enumerate(documents):
            for topic, probability in topics:
                matrix[i, topic] = probability
        return matrix
    req_topic_matrix, code_topic_matrix = topic_matrix(req_topics), topic_matrix(code_topics)
    lda_matrix = 1 - np.linalg.norm(np.sqrt(req_topic_matrix[:, None, :]) -
                                   np.sqrt(code_topic_matrix[None, :, :]), axis=2) / math.sqrt(2)
    output = {'VSM': vsm, 'LSI': lsi_matrix, 'LDA': lda_matrix}
    assert all(value.shape == (len(requirements), len(codes)) and np.isfinite(value).all()
               for value in output.values())
    metadata = {'requirement_dictionary_size': len(dictionary),
                'requirement_bow_empty': [i for i, doc in enumerate(req_bow) if not doc],
                'code_bow_empty_under_requirement_dictionary': [i for i, doc in enumerate(code_bow) if not doc],
                'lsi_requested_topics': 200, 'lsi_effective_topics': int(lsi.projection.u.shape[1]),
                'lda_sparse_omitted_mass': {
                    'requirement': (1 - req_topic_matrix.sum(axis=1)).tolist(),
                    'code': (1 - code_topic_matrix.sum(axis=1)).tolist()},
                'vsm_tfidf_idf': {str(k): float(v) for k, v in code_tfidf.idfs.items()},
                'lsi_tfidf_idf': {str(k): float(v) for k, v in req_tfidf.idfs.items()}}
    return output, metadata

CODE_KEYWORDS = frozenset('abstract assert auto bool boolean break byte case catch char class const '
    'continue default delete do double else enum extends extern false final finally float for friend '
    'goto if implements import inline instanceof int interface long native namespace new null nullptr '
    'package private protected public register return short signed sizeof static strictfp struct super '
    'switch synchronized template this throw throws transient true try typedef typename union unsigned '
    'using virtual void volatile while'.split())

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | CODE_KEYWORDS

def tokenize(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return [t for t in re.findall(r'[^\W_]+', text.casefold(), re.UNICODE)
            if len(t) > 1 and not t.isnumeric() and t not in STOPWORDS]

def bm25_matrix(counts, query_counts, binary_query=True):
    counts = counts.tocsr().astype(np.float64)
    n = counts.shape[0]
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    avg = float(lengths.mean())
    df = np.asarray((counts>0).sum(axis=0)).ravel()
    idf = np.log1p((n-df+0.5)/(df+0.5))
    weighted = counts.copy()
    for i in range(n):
        start,end = weighted.indptr[i:i+2]
        tf = weighted.data[start:end].copy()
        ids = weighted.indices[start:end]
        weighted.data[start:end] = idf[ids]*tf*2.2/(tf+1.2*(0.25+0.75*lengths[i]/avg))
    query = (query_counts > 0).astype(np.float64) if binary_query else query_counts
    return (query @ weighted.T).toarray(), idf, lengths

def project_scores(dataset, artifacts, text):
    code_paths = sorted(p for p,r in artifacts.items() if r['dataset']==dataset and r['kind']=='cc')
    req_paths = sorted(p for p,r in artifacts.items() if r['dataset']==dataset and r['kind']=='uc')
    vectorizer = CountVectorizer(analyzer=tokenize, lowercase=False, dtype=np.float64)
    counts = vectorizer.fit_transform([text[p] for p in code_paths])
    query_counts = vectorizer.transform([text[p] for p in req_paths])
    tfidf = TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=True)
    codes = tfidf.fit_transform(counts)
    queries = tfidf.transform(query_counts)
    vsm = np.clip((queries @ codes.T).toarray(),0,1)
    bm25, bm_idf, lengths = bm25_matrix(counts, query_counts)
    dimension = min(100,len(code_paths)-1,counts.shape[1]-1)
    assert dimension>=1
    svd = TruncatedSVD(n_components=dimension,algorithm='randomized',n_iter=7,
                       n_oversamples=10,power_iteration_normalizer='auto',random_state=42)
    svd.fit(codes)
    latent_codes = normalize(svd.transform(codes), norm='l2')
    latent_queries = normalize(svd.transform(queries),norm='l2')
    lsi = np.clip(latent_queries @ latent_codes.T,-1,1)
    raw = {'VSM':vsm,'BM25':bm25,'LSI':lsi}
    normed, maxima = {},{}
    for m,scores in raw.items():
        maxima[m] = np.maximum(0,np.max(scores,axis=1))
        normed[m] = np.divide(np.maximum(0,scores),maxima[m][:,None],
                              out=np.zeros_like(scores),where=maxima[m][:,None]>0)
        assert np.isfinite(scores).all() and np.isfinite(normed[m]).all()
    metadata = {'dataset':dataset,'code_paths':code_paths,'requirement_paths':req_paths,
                'code_files':len(code_paths),'requirements':len(req_paths),
                'features':counts.shape[1],'lsi_dimensions':dimension,
                'vocabulary_sha256':sha(canonical(vectorizer.vocabulary_)),
                'code_text_sha256':sha(canonical([text[p] for p in code_paths])),
                'requirement_text_sha256':sha(canonical([text[p] for p in req_paths])),
                'code_nonzero_tokens':int(counts.nnz),'code_token_count':float(lengths.sum()),
                'empty_code_vectors':int(np.sum(np.asarray(counts.sum(axis=1)).ravel()==0)),
                'empty_query_vectors':int(np.sum(np.asarray(query_counts.sum(axis=1)).ravel()==0)),
                'normalization_maxima':{m:values.tolist() for m,values in maxima.items()}}
    model = {'tfidf_idf':tfidf.idf_,'svd_components':svd.components_,'bm25_idf':bm_idf,
             'code_lengths':lengths,'vocabulary_json':np.array(json.dumps(vectorizer.vocabulary_,sort_keys=True)),
             **{m:raw[m] for m in METHODS},**{m+'_normalized':normed[m] for m in METHODS}}
    return raw,normed,metadata,model


def run_ir(inputs, artifact_root, output, bm25_reference_pairs=None):
    """Score all available code; select each project's fixed retention cutoff."""
    import statistics
    from fractions import Fraction
    from .metrics import classify
    artifacts = Artifacts(inputs, artifact_root)
    pairs = read_jsonl(Path(inputs) / 'pairs.jsonl')
    reference = read_jsonl(bm25_reference_pairs) if bm25_reference_pairs else pairs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ranking_rows, results = [], {}
    for project in PROJECTS:
        requirements = sorted((r for r in artifacts.entries.values() if r['project'] == project and r['side'] == 'requirement'), key=lambda r: r['relative_path'])
        codes = sorted((r for r in artifacts.entries.values() if r['project'] == project and r['side'] == 'code'), key=lambda r: r['relative_path'])
        rtexts = [artifacts.text(r['artifact_id']) for r in requirements]
        ctexts = [artifacts.text(r['artifact_id']) for r in codes]
        tokenization_dataset = 'maven' if project == 'Maven' else project
        rtokens = [tokenize_public(t, tokenization_dataset, 'uc') for t in rtexts]
        ctokens = [tokenize_public(t, tokenization_dataset, 'cc') for t in ctexts]
        matrices, _ = fit_public_scores(rtokens, ctokens)
        public_counts = CountVectorizer(analyzer=lambda tokens: tokens, dtype=np.float64)
        counts = public_counts.fit_transform(ctokens)
        matrices['BM25'], _, _ = bm25_matrix(counts, public_counts.transform(rtokens), binary_query=False)
        manifest = {r['relative_path']: {'dataset': project, 'kind': 'uc' if r['side'] == 'requirement' else 'cc'} for r in requirements + codes}
        texts = {r['relative_path']: artifacts.text(r['artifact_id']) for r in requirements + codes}
        raw, _, _, _ = project_scores(project, manifest, texts)
        ri = {r['artifact_id']: i for i, r in enumerate(requirements)}
        ci = {r['artifact_id']: i for i, r in enumerate(codes)}
        selected = [r for r in pairs if r['dataset'] == project]
        for pair in selected:
            ranking_rows.append({'pair_id': pair['pair_id'], 'scores': {m: float(raw[m][ri[pair['requirement_id']], ci[pair['code_id']]]) for m in METHODS}})
        reference_rows = [r for r in reference if r['dataset'] in (project, 'maven' if project == 'Maven' else project)]
        rnames = {r['name']: r['artifact_id'] for r in requirements}
        cnames = {r['name']: r['artifact_id'] for r in codes}
        reference_keys = []
        for row in reference_rows:
            if type(row['label']) is not int or row['label'] not in (0, 1):
                raise ValueError('Reference labels must be binary integers')
            rname = row.get('req_name') or Path(row.get('requirement_path', row.get('requirement_id', ''))).name
            cname = row.get('code_name') or Path(row.get('code_path', row.get('code_id', ''))).name
            reference_keys.append((rnames[rname], cnames[cname], row['label']))
        if not reference_keys:
            raise ValueError(f'Missing BM25 reference pairs for {project}')
        results[project] = {}
        for method in ('VSM', 'BM25', 'LSI', 'LDA'):
            scope = reference_keys if method == 'BM25' else [(r['requirement_id'], r['code_id'], r['label']) for r in selected]
            active = {r for r, _, _ in scope}
            positive = {(r, c) for r, c, label in scope if label}
            ranked = sorted(((r['artifact_id'], c['artifact_id'], float(matrices[method][i,j])) for i,r in enumerate(requirements) if r['artifact_id'] in active for j,c in enumerate(codes)), key=lambda x: (-x[2], x[0].rsplit('/',1)[-1], x[1].rsplit('/',1)[-1]))
            prefix = np.cumsum([int((r,c) in positive) for r,c,_ in ranked])
            options = []
            for percentage in range(1,101):
                n = round(percentage * len(ranked) / 100) if method == 'BM25' else math.floor(percentage * len(ranked) / 100)
                tp = int(prefix[n-1]) if n else 0
                fp, fn = n-tp, len(positive)-tp
                value = {'precision': tp/n if n else 0., 'recall': tp/len(positive) if positive else 0., 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., 'retention_percent': percentage, 'requirements': len(active), 'reference_links': len(positive), 'retrieval_pairs': len(ranked)}
                objective = Fraction(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)
                options.append((objective,-percentage,value))
            results[project][method] = max(options,key=lambda x:(x[0],x[1]))[2]
    results['Macro'] = {m: {k: statistics.fmean(results[p][m][k] for p in PROJECTS) for k in ('precision','recall','f1')} for m in ('VSM','BM25','LSI','LDA')}
    write_json(output / 'ir_classification.json',results)
    (output / 'ranking_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ranking_rows))
    return results['Macro']
