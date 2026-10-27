"""Exact dense retrieval with complete UTF-8 query coverage."""
import numpy as np


def query_chunks(text, tokenizer):
    raw = text.encode("utf-8")
    count = len(tokenizer.encode(text, add_special_tokens=False).ids)
    if count <= 30000 and len(raw) <= 120000:
        return [text]
    chunks, start = [], 0
    while start < len(raw):
        end = min(start + 24000, len(raw))
        while end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        if end <= start:
            raise ValueError("Cannot form a complete UTF-8 chunk")
        chunks.append(raw[start:end].decode("utf-8"))
        start = end
    assert "".join(chunks) == text
    return chunks


def embed_query(text, client, tokenizer):
    chunks = query_chunks(text, tokenizer)
    matrix = np.asarray([client.embed(x) for x in chunks], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or np.any(norms == 0):
        raise ValueError("Nonfinite or empty query vector")
    matrix /= norms[:, None]
    vector = np.average(matrix, axis=0, weights=[len(x.encode("utf-8")) for x in chunks])
    return vector / np.linalg.norm(vector)


def retrieve(requirement, code, case_ids, matrix, client, tokenizer):
    query = (requirement + "\n" + code).strip()
    vector = embed_query(query, client, tokenizer)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(case_ids), 1024) or not np.isfinite(matrix).all():
        raise ValueError("Case vectors do not match the case catalog")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Empty case embedding")
    scores = (matrix / norms[:, None]) @ vector
    top = sorted(range(len(case_ids)), key=lambda i: (-float(scores[i]), case_ids[i]))[:3]
    return [{"case_id": case_ids[i], "score": float(scores[i])} for i in top if scores[i] > .5]
