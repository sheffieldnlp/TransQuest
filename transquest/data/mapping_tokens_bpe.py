import numpy as np


def fix_chinese_characters(tokens):
    strf = ' '.join(tokens)
    strf = strf.replace('（ ', '(').replace(' ）', ')')
    strf = strf.replace('）', ')')
    strf = strf.replace('？', '?')
    strf = strf.replace('；', ';')
    strf = strf.replace('：', ':')
    strf = strf.replace('！', '!')
    strf = strf.replace('…', '...')
    strf = strf.replace('℃', '°c')
    strf = strf.replace('，', ',')
    return strf.split()


def map_pieces(pieces_from, pieces_to, values, method, to_sep='▁', from_sep='@@'):
    assert len(pieces_from) == len(values)
    pieces_from = fix_chinese_characters(pieces_from)
    pieces_to = fix_chinese_characters(pieces_to)
    if to_sep is not None:
        pieces_to = [p.replace(to_sep, '').strip() for p in pieces_to]
    if from_sep is not None:
        pieces_from = [p.replace(from_sep, '').strip() for p in pieces_from]
    try:
        assert ''.join(pieces_from).lower() == ''.join(pieces_to).lower()
    except AssertionError:
        print(''.join(pieces_from).lower())
        print(''.join(pieces_to).lower())
    positions_from = {}
    pos_counter = 0
    for i, token in enumerate(pieces_from):
        for _ in range(len(token)):
            positions_from[pos_counter] = values[i]
            pos_counter += 1
    if method == 'first':
        return mapping_first(pieces_to, positions_from)
    elif method == 'average':
        return mapping_first(pieces_to, positions_from)
    else:
        raise ValueError


def mapping_first(pieces_to, positions_from):
    result = []
    piece_lengths = [len(p) for p in pieces_to]
    positions = sorted(positions_from.keys())
    for i, piece in enumerate(pieces_to):
        pos = sum(piece_lengths[:i])
        try:
            result.append(positions_from[pos])
        except KeyError:
            print('Warning! Inconsistent number of tokens!')
            result.append(positions_from[positions[-1]])
    return result


def mapping_average(pieces_to, positions_from):
    result = []
    pos = 0
    for i, piece in enumerate(pieces_to):
        piece_vals_i = []
        if piece == '':
            piece_vals_i.append(positions_from[pos])
        for _ in piece:
            piece_vals_i.append(positions_from[pos])
            pos += 1
        result.append(np.mean(piece_vals_i))
    return result
