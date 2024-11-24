
import sys

from preprocess.execute import execute
from project_specific.label_attatch import label_and_drop as fr_label
from project_specific.collect_criteria import extract_criteria as ex_crit
from config import BASE_DIR

def show_help():
    print("Usage:")
    print("  python -m process                              # Run preprocess without encoding")
    print("  python -m process [-e, -encode]                # Produce one-hot encoding")
    print("  python -m process [-l, -label] [output_path]   # Create project-specific dataset")
    print("  python -m process [-f, -full] [output_path]    # All processes above will run.")
    print("  python -m process [-h, -help]                  # For help")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "-help"]:
            show_help()
            exit()
        elif sys.argv[1] in ["-e", "-encode"]:
            execute(features= False, encode= True)
            exit()
        elif sys.argv[1] in ["-l", "-label"]:
            if len(sys.argv) < 3:
                fr_label()
                ex_crit()
            else:
                output_path = f"{BASE_DIR}/{sys.argv[2]}"
                fr_label(output_path)
                ex_crit(output_path)
            exit()
        elif sys.argv[1] in ["-f", "-full"]:
            execute(features=True, encode=True)
            if len(sys.argv) < 3:
                fr_label()
                ex_crit()
            else:
                output_path = f"{BASE_DIR}/{sys.argv[2]}"
                fr_label(output_path)
                ex_crit(output_path)
        else:
            print(f"Error: Unknown argument '{sys.argv[1]}'. Use '-h' for help.")
            exit()
    else:
        execute(features=True, encode=False)
        exit()

    print("Error Occured, type '-h' for help.")
        