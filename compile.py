import argparse
import pycparser
from pycparser import parse_file

import re

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

    code = []

    for child in ast.ext:
        if type(child) == pycparser.c_ast.FuncDef:
            if child.decl.name == "main":
                for stmt in child.body.block_items:
                    # If the user is declaring a new variable
                    if type(stmt) == pycparser.c_ast.Decl:
                        assert stmt.type.type.names[0] == "int", "Only ints are supported"
                        assert re.match("r\d", stmt.name), "Variables names must be in the form rX"
                        assert 1 <= int(stmt.name[1:]) <= 12, "Variables must be named r1, r2, ..., r12"
                        assert stmt.init != None, "Variable declarations must have an initial value"
                        assert -128 <= int(stmt.init.value) <= 127, "Variable initial values must be between -128 and 127"

                        code += [f"setn {stmt.name} {stmt.init.value}"]
                    
                    elif type(stmt) == pycparser.c_ast.FuncCall:
                        assert stmt.name.name == "printf", "Only printf is supported"
                        assert stmt.args.exprs[0].value == '"%d\\n"', "Only printf(\"%d\\n\", ...) is supported"

                        code += [f"write {stmt.args.exprs[1].name}"]
    
    output = ""
    for i,line in enumerate(code):
        output += f"{i:03d} {line} \n"
    output += f"{len(code):03d} halt"
    print(output)