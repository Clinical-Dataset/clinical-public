
import sys

from preprocess.execute import execute
from project_specific.label_attatch import label_and_drop as fr_label
from project_specific.collect_criteria import extract_criteria as ex_crit
from project_specific.create_json import identify_unique as sv_json
from encode.onehot import encode_features as encode
from encode.protocol_encode import save_sentence_bert_dict_pkl as semb

def show_help():
    print("Usage:")
    print("  python -m process                              # Run preprocess")
    print("  python -m process [-l, -label] [output_path]   # Create project-specific dataset")
    print("  python -m process [-e, -encode] [output_path]  # Produce one-hot/sentence encoding for the dataset")
    print("  python -m process [-f, -full] [output_path]    # All processes above will run")
    print("  python -m process [-h, -help]                  # For help")

def error_call(x):
    if x == -1:
        raise FileNotFoundError("File not found. Please check the input.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "-help"]:
            show_help()
            exit()
        elif sys.argv[1] in ["-e", "-encode"]:
            if len(sys.argv) < 3:
                sv_json()
                encode()
                semb()
            else:
                output_path = f"{sys.argv[2]}"
                sv_json(output_path)
                encode(output_path)
                semb(output_path)
            exit()
        elif sys.argv[1] in ["-l", "-label"]:
            if len(sys.argv) < 3:
                err = fr_label()
                error_call(err)
                ex_crit()
            else:
                output_path = f"{sys.argv[2]}"
                err = fr_label(output_path)
                error_call(err)
                ex_crit(output_path)
            exit()
        elif sys.argv[1] in ["-f", "-full"]:
            execute(features=True)
            if len(sys.argv) < 3:
                err = fr_label()
                error_call(err)
                ex_crit()
                sv_json()
                encode()
            else:
                output_path = f"{sys.argv[2]}"
                err = fr_label(output_path)
                error_call(err)
                ex_crit(output_path)
                sv_json(output_path)
                encode(output_path)
            exit()
        else:
            print(f"Error: Unknown argument '{sys.argv[1]}'. Use '-h' for help.")
            exit()
    else:
        execute()
        exit()

    print("Error Occured, type '-h' for help.")
        