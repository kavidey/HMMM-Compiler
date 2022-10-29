import argparse
import pycparser
from pycparser import parse_file
from hmmm_code import HmmmProgram, HmmmRegister, generate_instruction

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

def find_used_registers(node: pycparser.c_ast.Node) -> set[HmmmRegister]:
    """Gets the set of all registers used in the given node (and its children)

    Args:
        node (pycparser.c_ast.Node) -- The node to search
    
    Returns:
        set[HmmmRegister] -- A set of all registers used in the given node and its children
    """
    if isinstance(node, pycparser.c_ast.ID):
        return {HmmmRegister(node.name)}
    elif isinstance(node, pycparser.c_ast.Constant):
        return set()
    elif isinstance(node, pycparser.c_ast.BinaryOp):
        return find_used_registers(node.left) | find_used_registers(node.right)
    elif isinstance(node, pycparser.c_ast.UnaryOp):
        return find_used_registers(node.expr)
    elif isinstance(node, pycparser.c_ast.Assignment):
        return find_used_registers(node.rvalue)
    else:
        raise NotImplementedError(f"Cannot find used registers for {type(node)}")

def find_unused_registers(set_of_registers: set[HmmmRegister]) -> set[HmmmRegister]:
    """ Finds all registers that are not in the given set of registers

    Args:
        set_of_registers (set[HmmmRegister]) -- The set of registers to check
    
    Returns:
        set[HmmmRegister] -- A set of all registers that are not in the given set of registers
    """
    return {register for register in HmmmRegister if register not in set_of_registers}

def parse_binary_op(node: pycparser.c_ast.BinaryOp, program: HmmmProgram) -> None:
    """Parses a binary operation and adds the appropriate instructions to the given program

    Recursively walks through the tree of the given node and turns binary operations (+, -, *, /) into Hmmm instructions.
    
    Args:
        node (pycparser.c_ast.BinaryOp) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
    """
    used_registers = find_used_registers(node)
    assert len(used_registers) < 11, "Cannot use more than 10 variables in a single expression"
    unused_registers = find_unused_registers(used_registers.union({HmmmRegister.R0, HmmmRegister.R13, HmmmRegister.R14, HmmmRegister.R15}))
    left_register = unused_registers.pop()
    right_register = unused_registers.pop()

    ### Left Side ###
    # If left is none, then the result of the left side is in R13
    # if left is an int, then the result of the left side is a constant
    # if left is a register, then the result of the left side is in that register
    # Whatever the left side is, we want to move it into an unused register
    left = None
    program.add_comment("Left Side")
    program.add_instruction(generate_instruction("pushr", left_register))
    if isinstance(node.left, pycparser.c_ast.ID):
        left = HmmmRegister(node.left.name)
        program.add_instruction(generate_instruction("copy", left_register, left))
    elif isinstance(node.left, pycparser.c_ast.Constant):
        left = int(node.left.value)
        program.add_instruction(generate_instruction("setn", left_register, left))
    elif isinstance(node.left, pycparser.c_ast.BinaryOp):
        program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
        parse_binary_op(node.left, program)
        program.add_instruction(generate_instruction("copy", left_register, HmmmRegister.R13))
        program.add_instruction(generate_instruction("popr", HmmmRegister.R13))
    
    ### Right Side ###
    # The logic here is the same as the left side
    right = None
    program.add_comment("Right Side")
    program.add_instruction(generate_instruction("pushr", right_register))
    if isinstance(node.right, pycparser.c_ast.ID):
        right = HmmmRegister(node.right.name)
        program.add_instruction(generate_instruction("copy", right_register, right))
    elif isinstance(node.right, pycparser.c_ast.Constant):  
        right = int(node.right.value)
        program.add_instruction(generate_instruction("setn", right_register, right))
    elif isinstance(node.right, pycparser.c_ast.BinaryOp):
        program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
        parse_binary_op(node.right, program)
        program.add_instruction(generate_instruction("copy", right_register, HmmmRegister.R13))
        program.add_instruction(generate_instruction("popr", HmmmRegister.R13))

    ### Operation ###
    if node.op == "+":
        program.add_instruction(generate_instruction("add", HmmmRegister.R13, left_register, right_register))
    elif node.op == "-":
        program.add_instruction(generate_instruction("sub", HmmmRegister.R13, left_register, right_register))
    elif node.op == "*":
        program.add_instruction(generate_instruction("mul", HmmmRegister.R13, left_register, right_register))
    elif node.op == "/":
        program.add_instruction(generate_instruction("div", HmmmRegister.R13, left_register, right_register))
    elif node.op == "%":
        program.add_instruction(generate_instruction("mod", HmmmRegister.R13, left_register, right_register))
    
    # Pop both registers that we used to store the left and right side of this operation
    program.add_instruction(generate_instruction("popr", right_register))
    program.add_instruction(generate_instruction("popr", left_register))

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
    ast.show(showcoord=True)

    # Make sure the AST contains valid code
    check_ast(ast)

    program = HmmmProgram()

    for child in ast.ext:
        if type(child) == pycparser.c_ast.FuncDef:
            if child.decl.name == "main":
                for stmt in child.body.block_items:
                    # If the user is declaring a new variable
                    if type(stmt) == pycparser.c_ast.Decl:
                        assert stmt.type.type.names[0] == "int", "Only ints are supported"
                        assert re.match("r\d", stmt.name), "Variables names must be in the form rX"
                        assert 1 <= int(stmt.name[1:]) <= 12, "Variables must be named r1, r2, ..., r12"
                        
                        if isinstance(stmt.init, pycparser.c_ast.Constant):
                            assert -128 <= int(stmt.init.value) <= 127, "Variable initial values must be between -128 and 127"
                            program.add_instruction(generate_instruction("setn", HmmmRegister(stmt.name), int(stmt.init.value)))
                        elif isinstance(stmt.init, pycparser.c_ast.BinaryOp):
                            parse_binary_op(stmt.init, program)
                            program.add_instruction(generate_instruction("copy", HmmmRegister(stmt.name), HmmmRegister.R13))
                            program.add_instruction(generate_instruction("popr", HmmmRegister.R13))
                        elif isinstance(stmt.init, pycparser.c_ast.ID):
                            program.add_instruction(generate_instruction("copy", HmmmRegister(stmt.name), HmmmRegister(stmt.init.name)))
                        # Test this later
                        # elif isinstance(stmt.init, pycparser.c_ast.UnaryOp):
                        #     assert stmt.init.op == "-", "Only unary negation is supported"
                        #     assert isinstance(stmt.init.expr, pycparser.c_ast.Constant), "Unary negation must be applied to a constant"
                        #     assert -128 <= int(stmt.init.expr.value) <= 127, "Unary negation must be applied to a constant between -128 and 127"
                        #     program.add_instruction(generate_instruction("setn", HmmmRegister(stmt.name), -int(stmt.init.expr.value)))
                        elif stmt.init == None:
                            program.add_instruction(generate_instruction("setn", HmmmRegister(stmt.name), 0))
                    
                    elif type(stmt) == pycparser.c_ast.FuncCall:
                        assert stmt.name.name in ["printf", "scanf"], "Only printf and scanf are supported"
                        if stmt.name.name == "printf":
                            assert stmt.args.exprs[0].value == '"%d\\n"', "Only printf(\"%d\\n\", ...) is supported"
                            assert len(stmt.args.exprs) == 2, "Only printf(\"%d\\n\", ...) is supported"

                            program.add_instruction(generate_instruction("write", HmmmRegister(stmt.args.exprs[1].name)))
                        elif stmt.name.name == "scanf":
                            assert stmt.args.exprs[0].value == '"%d"', "Only scanf(\"%d\", ...) is supported"
                            assert len(stmt.args.exprs) == 2, "Only scanf(\"%d\", ...) is supported"

                            program.add_instruction(generate_instruction("read", HmmmRegister(stmt.args.exprs[1].expr.name)))
    
    program.add_instruction(generate_instruction("halt"))
    program.add_stack_pointer_code()
    program.assign_line_numbers()
    print(program)