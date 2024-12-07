import sys

from preprocess.collect_features import get_features
from preprocess.stack_features import stack_features


def execute(features = True):

    if features:
        get_features()
        stack_features()
    


# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         if sys.argv[1] in ["-h", "-help"]:
#             print("Usage:")
#             print("  python -m preprocess.execute               # Run without encoding")
#             print("  python -m preprocess.execute -e            # Run with automatic encoding")
#             print("  python -m preprocess.execute -t [type]     # Create project-specific dataset")
#             print("  python -m preprocess.execute [-h, -help]   # For help")
#             exit()
#         if sys.argv[1] in ["-e"]:
#             execute(encode= True)
#             exit()

#     print("Error occured, type '-h' for help")
