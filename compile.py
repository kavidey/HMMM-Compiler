import argparse
import pycparser
from pycparser import parse_file
from hmmm_code import HmmmInstruction, HmmmProgram, HmmmRegister, MemoryAddress, generate_instruction
from typing import List, Tuple, Union

import re
import copy

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
    # program.add_comment("Left Side")
    program.add_instruction(generate_instruction("pushr", left_register))
    # TODO: Add Unary Op
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
    # program.add_comment("Right Side")
    program.add_instruction(generate_instruction("pushr", right_register))
    # TODO: Add Unary Op
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

def parse_unary_op(node: pycparser.c_ast.UnaryOp, program: HmmmProgram) -> None:
    """Parses a unary operation and adds the appropriate instructions to the given program

    Recursively walks through the tree of the given node and turns unary operations (-, !) into Hmmm instructions.
    
    Args:
        node (pycparser.c_ast.UnaryOp) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
    """

    if isinstance(node.expr, pycparser.c_ast.ID):
        program.add_instruction(generate_instruction("copy", HmmmRegister.R13, HmmmRegister(node.expr.name)))
    elif isinstance(node.expr, pycparser.c_ast.Constant):
        program.add_instruction(generate_instruction("setn", HmmmRegister.R13, int(node.expr.value)))
    elif isinstance(node.expr, pycparser.c_ast.BinaryOp):
        parse_binary_op(node.expr, program)
    
    if node.op == "-":
        program.add_instruction(generate_instruction("sub", HmmmRegister.R13, HmmmRegister.R0, HmmmRegister.R13))
    elif node.op == "p--":
        program.add_instruction(generate_instruction("addn", HmmmRegister.R13, -1))
    elif node.op == "p++":
        program.add_instruction(generate_instruction("addn", HmmmRegister.R13, 1))
    # elif node.op == "!":
    #     program.add_instruction(generate_instruction("not", HmmmRegister.R13, HmmmRegister.R13))
    else:
        raise NotImplementedError(f"Unary operation {node.op} is not implemented")

def parse_decl_assign(node: Union[pycparser.c_ast.Decl, pycparser.c_ast.Assignment], program: HmmmProgram) -> None:
    """Parses a declaration and adds the appropriate instructions to the given program
    
    Args:
        node (pycparser.c_ast.UnaryOp) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
    """

    if isinstance(node, pycparser.c_ast.Decl):
        assert node.type.type.names[0] == "int", "Only ints are supported"
        varname = node.name
        val = node.init
    elif isinstance(node, pycparser.c_ast.Assignment):
        varname = node.lvalue.name
        val = node.rvalue
        assert node.op == "=", "Only = is supported"
    else:
        raise Exception("Invalid node type")
    
    assert re.match("r\d", varname), "Variables names must be in the form rX"
    assert 1 <= int(varname[1:]) <= 12, "Variables must be named r1, r2, ..., r12"

    # TODO: Add Unary Op
    if isinstance(val, pycparser.c_ast.Constant):
        assert -128 <= int(val.value) <= 127, "Variable initial values must be between -128 and 127"
        program.add_instruction(generate_instruction("setn", HmmmRegister(varname), int(val.value)))
    elif isinstance(val, pycparser.c_ast.BinaryOp):
        program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
        parse_binary_op(val, program)
        program.add_instruction(generate_instruction("copy", HmmmRegister(varname), HmmmRegister.R13))
        program.add_instruction(generate_instruction("popr", HmmmRegister.R13))
    elif isinstance(val, pycparser.c_ast.ID):
        program.add_instruction(generate_instruction("copy", HmmmRegister(varname), HmmmRegister(val.name)))
    # Test this later
    # elif isinstance(stmt.init, pycparser.c_ast.UnaryOp):
    #     assert stmt.init.op == "-", "Only unary negation is supported"
    #     assert isinstance(stmt.init.expr, pycparser.c_ast.Constant), "Unary negation must be applied to a constant"
    #     assert -128 <= int(stmt.init.expr.value) <= 127, "Unary negation must be applied to a constant between -128 and 127"
    #     program.add_instruction(generate_instruction("setn", HmmmRegister(stmt.name), -int(stmt.init.expr.value)))
    elif val == None:
        program.add_instruction(generate_instruction("setn", HmmmRegister(varname), 0))

def parse_condition(node: pycparser.c_ast.BinaryOp, program: HmmmProgram) -> Tuple[HmmmInstruction, List[HmmmInstruction]]:
    """Parses a condition and returns the appropriate instructions to the given program
    
    Args:
        node (pycparser.c_ast.UnaryOp) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
    
    Returns:
        jump_instruction (HmmmInstruction) -- The instruction to jump to the end of the condition. The jump address needs to be set later
        cleanup_instructions (List[HmmmInstruction]) -- The instructions to clean up the stack. These instructions should be added after the condition is evaluated
    """
    cleanup_code: List[HmmmInstruction] = []

    used_registers = find_used_registers(node)
    assert len(used_registers) < 12, "Cannot use more than 11 variables in a single expression"
    unused_registers = find_unused_registers(used_registers.union({HmmmRegister.R0, HmmmRegister.R13, HmmmRegister.R14, HmmmRegister.R15}))
    left_register = unused_registers.pop()

    ### Left Side ###
    # TODO: Add Unary Op
    program.add_instruction(generate_instruction("pushr", left_register))
    cleanup_code.append(generate_instruction("popr", left_register))
    if isinstance(node.left, pycparser.c_ast.ID):
        program.add_instruction(generate_instruction("copy", left_register, HmmmRegister(node.left.name)))
    elif isinstance(node.left, pycparser.c_ast.Constant):
        program.add_instruction(generate_instruction("setn", left_register, int(node.left.value)))
    elif isinstance(node.left, pycparser.c_ast.BinaryOp):
        program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
        parse_binary_op(node.left, program)
        program.add_instruction(generate_instruction("copy", left_register, HmmmRegister.R13))
        program.add_instruction(generate_instruction("popr", HmmmRegister.R13))
    
    ### Right Side ###
    # Inside each if statement we take the right side and subtract it from the left to get ready for the comparison
    if isinstance(node.right, pycparser.c_ast.Constant) and node.right.value != "0":
        program.add_instruction(generate_instruction("addn", left_register, -int(node.right.value)))
    elif isinstance(node.right, pycparser.c_ast.ID):
        program.add_instruction(generate_instruction("sub", left_register, left_register, HmmmRegister(node.right.name)))
    
    ### Operation ###
    # The memory address to jump to needs to be overwritten with the correct address later
    if node.op == "==":
        jump_instruction = generate_instruction("jeqzn", left_register, MemoryAddress(-1))
    elif node.op == "!=":
        jump_instruction = generate_instruction("jnezn", left_register, MemoryAddress(-1))
    elif node.op == "<":
        jump_instruction = generate_instruction("jltzn", left_register, MemoryAddress(-1))
    elif node.op == ">":
        jump_instruction = generate_instruction("jgtzn", left_register, MemoryAddress(-1))
    else:
        assert False, "Invalid operation"
    
    program.add_instruction(jump_instruction)
    program.add_instructions(copy.deepcopy(cleanup_code))

    return jump_instruction, cleanup_code

def parse_if(node: pycparser.c_ast.If, program: HmmmProgram, is_in_loop=False) -> Tuple[List[HmmmInstruction], List[HmmmInstruction]]:
    """Parses an if statement and adds the appropriate instructions to the given program
    
    Args:
        node (pycparser.c_ast.If) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
        is_in_loop (bool) -- Whether or not the if statement is in a loop
    
    Returns:
        break_code (List[HmmmInstruction]) -- The instructions to execute when a break statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
        continue_code (List[HmmmInstruction]) -- The instructions to execute when a continue statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
    """

    assert isinstance(node.cond, pycparser.c_ast.BinaryOp), "Only binary operations are supported in if statements"
    assert node.cond.op in ["<", ">", "<=", ">=", "==", "!="], "Only <, >, <=, >=, ==, and != are supported"

    jump_iftrue, cleanup_code = parse_condition(node.cond, program)
    
    jump_iffalse = generate_instruction("jumpn", MemoryAddress(-1))
    program.add_instruction(jump_iffalse)
    
    program.add_instructions(cleanup_code)
    jump_iftrue.arg2 = cleanup_code[-1].address
    break_statements, continue_statments = parse_compound(node.iftrue, program, is_in_loop)

    jump_end = None
    # If there is an else statement add it to the program
    if node.iffalse != None:
        jump_end = generate_instruction("jumpn", MemoryAddress(-1))
        program.add_instruction(jump_end)
        beginning_of_iffalse = generate_instruction("nop")
        program.add_instruction(beginning_of_iffalse)
        jump_iffalse.arg1 = beginning_of_iffalse.address
        if isinstance(node.iffalse, pycparser.c_ast.Compound):
            parse_compound(node.iffalse, program, is_in_loop)
        elif isinstance(node.iffalse, pycparser.c_ast.If):
            parse_if(node.iffalse, program)
    else:
        jump_end = jump_iffalse

    end_of_if_block = generate_instruction("nop")
    program.add_instruction(end_of_if_block)
    jump_end.arg1 = end_of_if_block.address

    return break_statements, continue_statments

def parse_while(node: pycparser.c_ast.While, program: HmmmProgram) -> None:
    """Parses a while loop and adds the appropriate instructions to the given program
    
    Args:
        node (pycparser.c_ast.While) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
    """

    assert isinstance(node.cond, pycparser.c_ast.BinaryOp), "Only binary operations are supported in while loops"
    assert node.cond.op in ["<", ">", "<=", ">=", "==", "!="], "Only <, >, <=, >=, ==, and != are supported"

    beginning_of_while_check = generate_instruction("nop")
    program.add_instruction(beginning_of_while_check)

    jump_if_not_done, cleanup_code = parse_condition(node.cond, program)

    jump_if_done = generate_instruction("jumpn", MemoryAddress(-1))
    program.add_instruction(jump_if_done)

    beginning_of_while = generate_instruction("nop")
    program.add_instruction(beginning_of_while)
    jump_if_not_done.arg2 = beginning_of_while.address

    program.add_instructions(cleanup_code)
    break_statements, continue_statements = parse_compound(node.stmt, program, is_in_loop=True)

    program.add_instruction(generate_instruction("jumpn", beginning_of_while_check.address))

    end_of_while = generate_instruction("nop")
    program.add_instruction(end_of_while)
    jump_if_done.arg1 = end_of_while.address

    for break_statement in break_statements:
        break_statement.arg1 = end_of_while.address

    for continue_statement in continue_statements:
        continue_statement.arg1 = beginning_of_while_check.address

def parse_for(node: pycparser.c_ast.While, program: HmmmProgram) -> None:
    """Parses a for loop and adds the appropriate instructions to the given program
    
    Args:
        node (pycparser.c_ast.While) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
    """

    assert isinstance(node.cond, pycparser.c_ast.BinaryOp), "Only binary operations are supported for loops conditions"
    assert node.cond.op in ["<", ">", "<=", ">=", "==", "!="], "Only <, >, <=, >=, ==, and != are supported"

    assert isinstance(node.next, pycparser.c_ast.UnaryOp), "Only unary operations are supported for loops next statements"
    assert node.next.op == "p--" or node.next.op == "p++", "Only -- and ++ are supported"

    assert len(node.init.decls) == 1, "Only one variable can be declared in a for loop"

    parse_decl_assign(node.init.decls[0], program)

    beginning_of_for_check = generate_instruction("nop")
    program.add_instruction(beginning_of_for_check)

    jump_if_not_done, cleanup_code = parse_condition(node.cond, program)

    jump_if_done = generate_instruction("jumpn", MemoryAddress(-1))
    program.add_instruction(jump_if_done)
    
    beginning_of_for = generate_instruction("nop")
    program.add_instruction(beginning_of_for)
    jump_if_not_done.arg2 = beginning_of_for.address

    program.add_instructions(cleanup_code)
    break_statements, continue_statements = parse_compound(node.stmt, program, is_in_loop=True)

    program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
    parse_unary_op(node.next, program)
    program.add_instruction(generate_instruction("copy", HmmmRegister(node.next.expr.name), HmmmRegister.R13))
    program.add_instruction(generate_instruction("popr", HmmmRegister.R13))

    program.add_instruction(generate_instruction("jumpn", beginning_of_for_check.address))

    end_of_for = generate_instruction("nop")
    program.add_instruction(end_of_for)
    jump_if_done.arg1 = end_of_for.address

    for break_statement in break_statements:
        break_statement.arg1 = end_of_for.address
    
    for continue_statement in continue_statements:
        continue_statement.arg1 = beginning_of_for_check.address

def parse_compound(node: pycparser.c_ast.Compound, program: HmmmProgram, is_in_loop = False) -> Tuple[List[HmmmInstruction], List[HmmmInstruction]]:
    """Parses a compound statement and adds the appropriate instructions to the given program
    
    Args:
        node (pycparser.c_ast.Compound) -- The node to parse
        program (HmmmProgram) -- The program to add instructions to
        is_in_loop (bool) -- Whether or not the compound statement is in a loop (affects how break & continue are handled)

    Returns:
        break_code (List[HmmmInstruction]) -- The instructions to execute when a break statement is encountered. The user needs to set the jump address of these manually
        continue_code (List[HmmmInstruction]) -- The instructions to execute when a continue statement is encountered. The user needs to set the jump address of these manually
    """

    break_code = []
    continue_code = []

    for stmt in node.block_items:
        # If the user is declaring a new variable
        if isinstance(stmt, pycparser.c_ast.Decl) or isinstance(stmt, pycparser.c_ast.Assignment):
            parse_decl_assign(stmt, program)
        elif isinstance(stmt, pycparser.c_ast.If):
            break_statements, if_statements = parse_if(stmt, program, is_in_loop)
            break_code += break_statements
            continue_code += if_statements
        elif isinstance(stmt, pycparser.c_ast.While):
            parse_while(stmt, program)
        elif isinstance(stmt, pycparser.c_ast.For):
            parse_for(stmt, program)
        elif isinstance(stmt, pycparser.c_ast.Continue):
            if is_in_loop:
                continue_statement = generate_instruction("jumpn", MemoryAddress(-1))
                program.add_instruction(continue_statement)
                continue_code.append(continue_statement)
            else:
                raise Exception("Continue statement not in a loop")
        elif isinstance(stmt, pycparser.c_ast.Break):
            if is_in_loop:
                break_statement = generate_instruction("jumpn", MemoryAddress(-1))
                program.add_instruction(break_statement)
                break_code.append(break_statement)
            else:
                raise Exception("Break statement not in a loop")
        elif isinstance(stmt, pycparser.c_ast.Compound):
            parse_compound(stmt, program)
        elif isinstance(stmt, pycparser.c_ast.FuncCall):
            assert stmt.name.name in ["printf", "scanf"], "Only printf and scanf are supported"
            if stmt.name.name == "printf":
                assert stmt.args.exprs[0].value == '"%d\\n"', "Only printf(\"%d\\n\", ...) is supported"
                assert len(stmt.args.exprs) == 2, "Only printf(\"%d\\n\", ...) is supported"

                # If the argument is a variable
                if isinstance(stmt.args.exprs[1], pycparser.c_ast.ID):
                    program.add_instruction(generate_instruction("write", HmmmRegister(stmt.args.exprs[1].name)))
                # If the argument is a constant
                elif isinstance(stmt.args.exprs[1], pycparser.c_ast.Constant):
                    program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
                    program.add_instruction(generate_instruction("setn", HmmmRegister.R13, int(stmt.args.exprs[1].value)))
                    program.add_instruction(generate_instruction("write", HmmmRegister.R13))
                    program.add_instruction(generate_instruction("popr", HmmmRegister.R13))
                elif isinstance(stmt.args.exprs[1], pycparser.c_ast.UnaryOp):
                    assert stmt.args.exprs[1].op == "-", "Only unary - is supported"
                    program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
                    parse_unary_op(stmt.args.exprs[1], program)
                    program.add_instruction(generate_instruction("write", HmmmRegister.R13))
                    program.add_instruction(generate_instruction("popr", HmmmRegister.R13))
            elif stmt.name.name == "scanf":
                assert stmt.args.exprs[0].value == '"%d"', "Only scanf(\"%d\", ...) is supported"
                assert len(stmt.args.exprs) == 2, "Only scanf(\"%d\", ...) is supported"

                program.add_instruction(generate_instruction("read", HmmmRegister(stmt.args.exprs[1].expr.name)))
        elif isinstance(stmt, pycparser.c_ast.UnaryOp):
            program.add_instruction(generate_instruction("pushr", HmmmRegister.R13))
            parse_unary_op(stmt, program)
            program.add_instruction(generate_instruction("copy", HmmmRegister(stmt.expr.name), HmmmRegister.R13))
            program.add_instruction(generate_instruction("popr", HmmmRegister.R13))
        elif isinstance(stmt, pycparser.c_ast.Return):
            pass
        else:
            raise NotImplementedError(f"Unsupported statement: {type(stmt)}")
    
    return break_code, continue_code

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

    program = HmmmProgram()

    for child in ast.ext:
        if isinstance(child, pycparser.c_ast.FuncDef):
            if child.decl.name == "main":
                if isinstance(child.body, pycparser.c_ast.Compound):
                    parse_compound(child.body, program)
    
    program.add_instruction(generate_instruction("halt"))
    program.add_stack_pointer_code()
    program.assign_line_numbers()
    print(program)