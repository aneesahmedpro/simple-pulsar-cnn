import numpy as np


def stratified_shuffle_split_for_binary(labels, test_fraction=0.25):

    pos, neg = [], []
    for i in range(len(labels)):
        if labels[i]:
            pos.append(i)
        else:
            neg.append(i)

    if len(pos) < 2:
        raise RuntimeError('Bad dataset: Too few positive examples.')
    if len(neg) < 2:
        raise RuntimeError('Bad dataset: Too few negative examples.')

    pos_test_samples_count = max(1, int(test_fraction * len(pos)))
    neg_test_samples_count = max(1, int(test_fraction * len(neg)))

    np.random.shuffle(pos)
    np.random.shuffle(neg)

    pos_test_samples_idx = pos[:pos_test_samples_count]
    neg_test_samples_idx = neg[:neg_test_samples_count]
    pos_train_samples_idx = pos[pos_test_samples_count:]
    neg_train_samples_idx = neg[neg_test_samples_count:]

    train_idx = np.concatenate([pos_train_samples_idx, neg_train_samples_idx])
    test_idx = np.concatenate([pos_test_samples_idx, neg_test_samples_idx])

    return train_idx, test_idx


def example():

    count_neg = 80
    count_pos = 20

    x = []
    for i in range(count_neg):
        x.append('N_{}'.format(i))
    for i in range(count_pos):
        x.append('Y_{}'.format(i))
    x = np.array(x)

    y = np.concatenate(
        [np.full(count_neg, fill_value=0), np.full(count_pos, fill_value=1)])

    shuffled_idx = np.random.permutation(count_pos+count_neg)
    x = x[shuffled_idx]
    y = y[shuffled_idx]

    print(x, len(x))
    print(y, len(y))

    train_idx, test_idx = stratified_shuffle_split_for_binary(y)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print()

    print(x_test, len(x_test))
    print(y_test, len(y_test))
    print(x_train, len(x_train))
    print(y_train, len(y_train))


if __name__ == '__main__':

    example()
