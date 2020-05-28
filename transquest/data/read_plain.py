from transquest.data.containers import InputExampleWord


def load_examples(src, tgt, labels):
    examples = [
        InputExampleWord(guid=i, text_a=text_b, label=label)
        for i, (text_a, text_b, label) in enumerate(
            zip(src, tgt, labels)
        )
    ]
    return examples


def read_labels(path):
    labels = []
    for line in open(path):
        tags = line.strip().split()
        tags = [1 if t == 'OK' else 0 for t in tags]
        labels_i = [t for i, t in enumerate(tags) if i % 2 != 0]
        assert len(labels_i) * 2 + 1 == len(tags)
        labels.append(labels_i)
    return labels


def read_data(src_path, tgt_path, labels_path):
    labels = read_labels(labels_path)
    src = [l.strip() for l in open(src_path)]
    tgt = [l.strip() for l in open(tgt_path)]
    return src, tgt, labels
