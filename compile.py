import argparse
import pycparser
from pycparser import parse_file

def check_ast(ast):
    # TODO: check ast
    # This includes checking for:
    # - all variables are ints
    # - all variables are declared before use (and in scope)
    # - all variables are named r1, r2, ..., rX
    # - all printf calls have the correct format

    # Some of these may make more sense to check during the code generation process which is why I'm not writing this function yet.
    pass

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Compile a C file into Hmmm assembly")
    argparser.add_argument(
        "filename",
        default="examples/c_files/basic.c",
        nargs="?",
        help="name of file to compile",
    )
    args = argparser.parse_args()

    # Generate the AST
    ast = parse_file(args.filename, use_cpp=True,
            cpp_path='clang',
            cpp_args=['-E', r'-Iutils/pycparser/utils/fake_libc_include'])
    # ast.show(showcoord=True)

    # Make sure the AST contains valid code
    check_ast(ast)

    for child in ast.ext:
        if type(child) == pycparser.c_ast.FuncDef:
            if child.decl.name == "main":
                for stmt in child.body.block_items:
                    print(stmt)