from preprocess.collect_features import get_features
from preprocess.stack_features import stack_features
from preprocess.encode import encode_features


def execute():

    get_features()
    stack_features()
    encode_features()


if __name__ == "__main__":

    execute()
