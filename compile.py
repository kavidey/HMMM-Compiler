import argparse
from pycparser import parse_file

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Compile a C file into Hmmm assembly")
    argparser.add_argument(
        "filename",
        default="examples/c_files/basic.c",
        nargs="?",
        help="name of file to compile",
    )
    args = argparser.parse_args()

    ast = parse_file(args.filename, use_cpp=True,
            cpp_path='clang',
            cpp_args=['-E', r'-Iutils/pycparser/utils/fake_libc_include'])
    ast.show(showcoord=True)
