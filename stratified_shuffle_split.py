import numpy as np


def stratified_shuffle_split_for_binary(x, y, test_fraction=0.25):

    pos, neg = [], []
    for i in range(len(y)):
        if y[i]:
            pos.append(i)
        else:
            neg.append(i)

    pos_test_samples_count = int(test_fraction * len(pos))
    neg_test_samples_count = int(test_fraction * len(neg))

    np.random.shuffle(pos)
    np.random.shuffle(neg)

    pos_test_samples_idx = pos[:pos_test_samples_count]
    neg_test_samples_idx = neg[:neg_test_samples_count]
    pos_train_samples_idx = pos[pos_test_samples_count:]
    neg_train_samples_idx = neg[neg_test_samples_count:]

    concat = np.concatenate
    x_test = concat((x[pos_test_samples_idx], x[neg_test_samples_idx]))
    x_train = concat((x[pos_train_samples_idx], x[neg_train_samples_idx]))
    y_test = concat((y[pos_test_samples_idx], y[neg_test_samples_idx]))
    y_train = concat((y[pos_train_samples_idx], y[neg_train_samples_idx]))

    return x_train, y_train, x_test, y_test


def example():

    x = []
    for i in range(80):
        x.append('N_{}'.format(i))
    for i in range(20):
        x.append('Y_{}'.format(i))
    x = np.array(x)

    y = np.concatenate((np.full(80, fill_value=0), np.full(20, fill_value=1)))

    # shuffled_idx = np.random.permutation(80+20)
    # x = x[shuffled_idx]
    # y = y[shuffled_idx]

    print(x, len(x))
    print(y, len(y))

    retval = stratified_shuffle_split_for_binary(x, y)
    x_train, y_train, x_test, y_test = retval
    print()

    print(x_test, len(x_test))
    print(y_test, len(y_test))
    print(x_train, len(x_train))
    print(y_train, len(y_train))


if __name__ == '__main__':

    example()
