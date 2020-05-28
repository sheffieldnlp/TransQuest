

def map_tokens_bpe(tokens, pieces, labels, bpe_sep='▁'):
    pieces = [p.replace(bpe_sep, '') for p in pieces]
    assert ''.join(pieces) == ''.join(tokens)
    token_positions = {}
    pos_counter = 0
    for i, token in enumerate(tokens):
        for _ in range(len(token)):
            token_positions[pos_counter] = labels[i]
            pos_counter += 1
    pieces = [p.replace(bpe_sep, '▁') for p in pieces]
    piece_labels = []
    piece_lengths = [len(p) for p in pieces]
    for i, piece in enumerate(pieces):
        pos = sum(piece_lengths[:i])
        piece_labels.append(token_positions[pos])
    return piece_labels
