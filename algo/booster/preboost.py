import scipy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def prepare_training_file(test_df, column_name, train_df=None):

    embedder = SentenceTransformer('bert-base-nli-mean-tokens')

    test_sentence_list = test_df[column_name].tolist()
    test_embeddings = embedder.encode(test_sentence_list)

    if train_df is not None:
        train_sentence_list = train_df[column_name].tolist()
        train_embeddings = embedder.encode(train_sentence_list)
        train_quality_list = train_df["labels"].tolist()

    else:
        train_sentence_list = test_sentence_list
        train_embeddings = test_embeddings
        train_quality_list = test_df["labels"].tolist()

    similarity_sentence_list = []
    similarity_list = []
    quality_list = []

    for test_sentence, test_embedding in zip(test_sentence_list, test_embeddings):
        distances = scipy.spatial.distance.cdist([test_embedding], train_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        if train_df is not None:
            idx, distance = results[0]
        else:
            idx, distance = results[1]

        similarity_sentence_list.append(train_sentence_list[idx])
        similarity_list.append(1 - distance)
        quality_list.append(train_quality_list[idx])

    test_df['similar_sentence'] = similarity_sentence_list
    test_df['similarity'] = similarity_list
    test_df['similar_sentence_quality'] = quality_list

    return test_df
