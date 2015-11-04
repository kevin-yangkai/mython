import cPickle

__author__ = 'zhangxulong'


def load_dataset(pkl_dir):
    pkl_file = file(pkl_dir, 'rb')
    dataset = cPickle.load(pkl_file)
    pkl_file.close()
    return dataset


def pickle_dataset(dataset):
    pkl_file = file('trans_dataset.pkl', 'wb')
    cPickle.dump(dataset, pkl_file, True)
    pkl_file.close()
    return 0


def transform(dataset=load_dataset('dataset.pkl')):
    transform_dataset = {"author": "zhangxulong"}
    data = dataset[0]
    target = dataset[1]
    transform_dataset['data'] = data
    transform_dataset['singers_label'] = target
    pickle_dataset(transform_dataset)
    return transform_dataset
transform()