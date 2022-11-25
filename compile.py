import argparse
import copy
import re
from typing import List, Tuple, Union, Optional

import pycparser
from pycparser import parse_file

from lib.hmmm_program import HmmmProgram
from lib.hmmm_utils import (
    TemporaryRegister,
    get_temporary_register,
    generate_instruction,
    HmmmRegister,
    HmmmInstruction,
    MemoryAddress,
)


class Scope:
    def __init__(self, parent_scope: "Optional[Scope]" = None) -> None:
        self.scope: dict[object, TemporaryRegister] = {}
        self.parent_scope = parent_scope

    def __getitem__(self, key: object) -> TemporaryRegister:
        if key in self.scope:
            return self.scope[key]
        elif self.parent_scope is not None:
            return self.parent_scope[key]
        else:
            raise ReferenceError(f"Variable {key} not found in scope")

    def declar_var(
        self,
        key: object,
        register: Optional[HmmmRegister] = None,
    ) -> TemporaryRegister:
        return get_temporary_register(self.scope, key, register)

    def make_temporary(self, key: Optional[object] = None) -> TemporaryRegister:
        return get_temporary_register(self.scope, key)

    def get_vars(self) -> List[TemporaryRegister]:
        return list(self.scope.values())


class Compiler:
    def __init__(self) -> None:
        pass

    def parse_binary_op(
        self, node: pycparser.c_ast.BinaryOp, program: HmmmProgram
    ) -> TemporaryRegister:
        """Parses a binary operation and adds the appropriate instructions to the given program

        Recursively walks through the tree of the given node and turns binary operations (+, -, *, /) into Hmmm instructions.

        Args:
            node (pycparser.c_ast.BinaryOp) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to
        """
        result = self.current_scope.make_temporary()
        left_register = self.parse_expression(node.left, program)
        right_register = self.parse_expression(node.right, program)

        ### Operation ###
        if node.op == "+":
            program.add_instruction(
                generate_instruction("add", result, left_register, right_register)
            )
        elif node.op == "-":
            program.add_instruction(
                generate_instruction("sub", result, left_register, right_register)
            )
        elif node.op == "*":
            program.add_instruction(
                generate_instruction("mul", result, left_register, right_register)
            )
        elif node.op == "/":
            program.add_instruction(
                generate_instruction("div", result, left_register, right_register)
            )
        elif node.op == "%":
            program.add_instruction(
                generate_instruction("mod", result, left_register, right_register)
            )

        return result

    def parse_unary_op(
        self, node: pycparser.c_ast.UnaryOp, program: HmmmProgram
    ) -> TemporaryRegister:
        """Parses a unary operation and adds the appropriate instructions to the given program

        Recursively walks through the tree of the given node and turns unary operations (-, !) into Hmmm instructions.

        Args:
            node (pycparser.c_ast.UnaryOp) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to
        """

        result = self.parse_expression(node.expr, program)

        if node.op == "-":
            program.add_instruction(generate_instruction("neg", result, result))
        elif node.op == "p--":
            program.add_instruction(generate_instruction("addn", result, -1))
        elif node.op == "p++":
            program.add_instruction(generate_instruction("addn", result, 1))
        # elif node.op == "!":
        #     program.add_instruction(generate_instruction("not", HmmmRegister.R13, HmmmRegister.R13))
        else:
            raise NotImplementedError(f"Unary operation {node.op} is not implemented")

        return result

    def parse_expression(self, expr: object, program: HmmmProgram, result: Optional[TemporaryRegister] = None) -> TemporaryRegister:
        """Parses an expression and adds the appropriate instructions to the given program"""

        if not result:
            result = self.current_scope.make_temporary()
        
        if isinstance(expr, pycparser.c_ast.ID):
            program.add_instruction(
                generate_instruction(
                    "copy",
                    result,
                    self.current_scope[expr.name],
                )
            )
        elif isinstance(expr, pycparser.c_ast.Constant):
            program.add_instruction(
                generate_instruction("setn", result, int(expr.value))
            )
        elif isinstance(expr, pycparser.c_ast.BinaryOp):
            binary_op_result = self.parse_binary_op(expr, program)
            program.add_instruction(
                generate_instruction("copy", result, binary_op_result)
            )
        elif isinstance(expr, pycparser.c_ast.UnaryOp):
            unary_op_result = self.parse_unary_op(expr, program)
            program.add_instruction(
                generate_instruction("copy", result, unary_op_result)
            )
        else:
            raise NotImplementedError(f"Expression type {type(expr)} is not implemented")

        return result

    def parse_decl_assign(
        self,
        node: Union[pycparser.c_ast.Decl, pycparser.c_ast.Assignment],
        program: HmmmProgram,
    ) -> None:
        """Parses a declaration and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.UnaryOp) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to
        """

        if isinstance(node, pycparser.c_ast.Decl):
            assert node.type.type.names[0] == "int", "Only ints are supported"
            temp = self.current_scope.declar_var(node.name)
            val = node.init
        elif isinstance(node, pycparser.c_ast.Assignment):
            temp = self.current_scope[node.lvalue.name]
            val = node.rvalue
            assert node.op == "=", "Only = is supported"
        else:
            raise Exception("Invalid node type")

        if val == None:
            program.add_instruction(generate_instruction("setn", temp, 0))
        else:
            self.parse_expression(val, program, temp)

    def parse_condition(
        self, node: pycparser.c_ast.BinaryOp, program: HmmmProgram
    ) -> HmmmInstruction:
        """Parses a condition and returns the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.UnaryOp) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to

        Returns:
            jump_instruction (HmmmInstruction) -- The instruction to jump to the end of the condition. The jump address needs to be set later
            cleanup_instructions (List[HmmmInstruction]) -- The instructions to clean up the stack. These instructions should be added after the condition is evaluated
        """
        left_register = self.parse_expression(node.left, program)

        ### Right Side ###
        # Inside each if statement we take the right side and subtract it from the left to get ready for the comparison
        if isinstance(node.right, pycparser.c_ast.Constant) and node.right.value != "0":
            program.add_instruction(
                generate_instruction("addn", left_register, -int(node.right.value))
            )
        elif isinstance(node.right, pycparser.c_ast.ID):
            program.add_instruction(
                generate_instruction(
                    "sub",
                    left_register,
                    left_register,
                    self.current_scope[node.right.name],
                )
            )

        ### Operation ###
        # The memory address to jump to needs to be overwritten with the correct address later
        if node.op == "==":
            jump_instruction = generate_instruction(
                "jeqzn", left_register, MemoryAddress(-1, None)
            )
        elif node.op == "!=":
            jump_instruction = generate_instruction(
                "jnezn", left_register, MemoryAddress(-1, None)
            )
        elif node.op == "<":
            jump_instruction = generate_instruction(
                "jltzn", left_register, MemoryAddress(-1, None)
            )
        elif node.op == ">":
            jump_instruction = generate_instruction(
                "jgtzn", left_register, MemoryAddress(-1, None)
            )
        else:
            assert False, "Invalid operation"

        program.add_instruction(jump_instruction)

        return jump_instruction

    def parse_if(
        self, node: pycparser.c_ast.If, program: HmmmProgram, is_in_loop=False
    ) -> Tuple[List[HmmmInstruction], List[HmmmInstruction]]:
        """Parses an if statement and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.If) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to
            is_in_loop (bool) -- Whether or not the if statement is in a loop

        Returns:
            break_code (List[HmmmInstruction]) -- The instructions to execute when a break statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
            continue_code (List[HmmmInstruction]) -- The instructions to execute when a continue statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
        """

        assert isinstance(
            node.cond, pycparser.c_ast.BinaryOp
        ), "Only binary operations are supported in if statements"
        assert node.cond.op in [
            "<",
            ">",
            "<=",
            ">=",
            "==",
            "!=",
        ], "Only <, >, <=, >=, ==, and != are supported"

        jump_iftrue = self.parse_condition(node.cond, program)

        jump_iffalse = generate_instruction("jumpn", MemoryAddress(-1, None))
        program.add_instruction(jump_iffalse)

        nop = generate_instruction("nop")
        program.add_instruction(nop)
        jump_iftrue.arg2 = MemoryAddress(-1, nop)
        break_statements, continue_statements = self.parse_compound(
            node.iftrue, program, is_in_loop
        )

        jump_end = None
        # If there is an else statement add it to the program
        if node.iffalse != None:
            jump_end = generate_instruction("jumpn", MemoryAddress(-1, None))
            program.add_instruction(jump_end)
            beginning_of_iffalse = generate_instruction("nop")
            program.add_instruction(beginning_of_iffalse)
            jump_iffalse.arg1 = MemoryAddress(-1, beginning_of_iffalse)
            if isinstance(node.iffalse, pycparser.c_ast.Compound):
                self.parse_compound(node.iffalse, program, is_in_loop)
            elif isinstance(node.iffalse, pycparser.c_ast.If):
                self.parse_if(node.iffalse, program)
        else:
            jump_end = jump_iffalse

        end_of_if_block = generate_instruction("nop")
        program.add_instruction(end_of_if_block)
        jump_end.arg1 = MemoryAddress(-1, end_of_if_block)

        return break_statements, continue_statements

    def parse_while(self, node: pycparser.c_ast.While, program: HmmmProgram) -> None:
        """Parses a while loop and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.While) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to
        """

        assert isinstance(
            node.cond, pycparser.c_ast.BinaryOp
        ), "Only binary operations are supported in while loops"
        assert node.cond.op in [
            "<",
            ">",
            "<=",
            ">=",
            "==",
            "!=",
        ], "Only <, >, <=, >=, ==, and != are supported"

        beginning_of_while_check = generate_instruction("nop")
        program.add_instruction(beginning_of_while_check)

        jump_if_not_done = self.parse_condition(node.cond, program)

        jump_if_done = generate_instruction("jumpn", MemoryAddress(-1, None))
        program.add_instruction(jump_if_done)

        beginning_of_while = generate_instruction("nop")
        program.add_instruction(beginning_of_while)
        jump_if_not_done.arg2 = MemoryAddress(-1, beginning_of_while)

        break_statements, continue_statements = self.parse_compound(
            node.stmt, program, is_in_loop=True
        )

        program.add_instruction(
            generate_instruction("jumpn", MemoryAddress(-1, beginning_of_while_check))
        )

        end_of_while = generate_instruction("nop")
        program.add_instruction(end_of_while)
        jump_if_done.arg1 = MemoryAddress(-1, end_of_while)

        for break_statement in break_statements:
            break_statement.arg1 = MemoryAddress(-1, end_of_while)

        for continue_statement in continue_statements:
            continue_statement.arg1 = MemoryAddress(-1, beginning_of_while_check)

    def parse_for(self, node: pycparser.c_ast.While, program: HmmmProgram) -> None:
        """Parses a for loop and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.While) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to
        """

        assert isinstance(
            node.cond, pycparser.c_ast.BinaryOp
        ), "Only binary operations are supported for loops conditions"
        assert node.cond.op in [
            "<",
            ">",
            "<=",
            ">=",
            "==",
            "!=",
        ], "Only <, >, <=, >=, ==, and != are supported"

        assert isinstance(
            node.next, pycparser.c_ast.UnaryOp
        ), "Only unary operations are supported for loops next statements"
        assert (
            node.next.op == "p--" or node.next.op == "p++"
        ), "Only -- and ++ are supported"

        assert (
            len(node.init.decls) == 1
        ), "Only one variable can be declared in a for loop"

        self.parse_decl_assign(node.init.decls[0], program)

        beginning_of_for_check = generate_instruction("nop")
        program.add_instruction(beginning_of_for_check)

        jump_if_not_done = self.parse_condition(node.cond, program)

        jump_if_done = generate_instruction("jumpn", MemoryAddress(-1, None))
        program.add_instruction(jump_if_done)

        beginning_of_for = generate_instruction("nop")
        program.add_instruction(beginning_of_for)
        jump_if_not_done.arg2 = MemoryAddress(-1, beginning_of_for)

        break_statements, continue_statements = self.parse_compound(
            node.stmt, program, is_in_loop=True
        )

        temp = self.parse_unary_op(node.next, program)
        program.add_instruction(
            generate_instruction(
                "copy",
                self.current_scope[node.next.expr.name],
                temp,
            )
        )

        program.add_instruction(
            generate_instruction("jumpn", MemoryAddress(-1, beginning_of_for_check))
        )

        end_of_for = generate_instruction("nop")
        program.add_instruction(end_of_for)
        jump_if_done.arg1 = MemoryAddress(-1, end_of_for)

        for break_statement in break_statements:
            break_statement.arg1 = MemoryAddress(-1, end_of_for)

        for continue_statement in continue_statements:
            continue_statement.arg1 = MemoryAddress(-1, beginning_of_for_check)

    def parse_compound(
        self, node: pycparser.c_ast.Compound, program: HmmmProgram, is_in_loop=False
    ) -> Tuple[List[HmmmInstruction], List[HmmmInstruction]]:
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
            if isinstance(stmt, pycparser.c_ast.Decl) or isinstance(
                stmt, pycparser.c_ast.Assignment
            ):
                self.parse_decl_assign(stmt, program)
            elif isinstance(stmt, pycparser.c_ast.If):
                break_statements, if_statements = self.parse_if(
                    stmt, program, is_in_loop
                )
                break_code += break_statements
                continue_code += if_statements
            elif isinstance(stmt, pycparser.c_ast.While):
                self.parse_while(stmt, program)
            elif isinstance(stmt, pycparser.c_ast.For):
                self.parse_for(stmt, program)
            elif isinstance(stmt, pycparser.c_ast.Continue):
                if is_in_loop:
                    continue_statement = generate_instruction(
                        "jumpn", MemoryAddress(-1, None)
                    )
                    program.add_instruction(continue_statement)
                    continue_code.append(continue_statement)
                else:
                    raise Exception("Continue statement not in a loop")
            elif isinstance(stmt, pycparser.c_ast.Break):
                if is_in_loop:
                    break_statement = generate_instruction(
                        "jumpn", MemoryAddress(-1, None)
                    )
                    program.add_instruction(break_statement)
                    break_code.append(break_statement)
                else:
                    raise Exception("Break statement not in a loop")
            elif isinstance(stmt, pycparser.c_ast.Compound):
                self.parse_compound(stmt, program)
            elif isinstance(stmt, pycparser.c_ast.FuncCall):
                assert stmt.name.name in [
                    "printf",
                    "scanf",
                ], "Only printf and scanf are supported"
                if stmt.name.name == "printf":
                    assert (
                        stmt.args.exprs[0].value == '"%d\\n"'
                    ), 'Only printf("%d\\n", ...) is supported'
                    assert (
                        len(stmt.args.exprs) == 2
                    ), 'Only printf("%d\\n", ...) is supported'

                    temp = self.parse_expression(stmt.args.exprs[1], program)
                    program.add_instruction(generate_instruction("write", temp))

                elif stmt.name.name == "scanf":
                    assert (
                        stmt.args.exprs[0].value == '"%d"'
                    ), 'Only scanf("%d", ...) is supported'
                    assert (
                        len(stmt.args.exprs) == 2
                    ), 'Only scanf("%d", ...) is supported'

                    program.add_instruction(
                        generate_instruction(
                            "read",
                            self.current_scope[stmt.args.exprs[1].expr.name],
                        )
                    )
            elif isinstance(stmt, pycparser.c_ast.UnaryOp):
                temp = self.parse_unary_op(stmt, program)
                program.add_instruction(
                    generate_instruction(
                        "copy",
                        self.current_scope[stmt.expr.name],
                        temp,
                    )
                )
            elif isinstance(stmt, pycparser.c_ast.Return):
                pass
            else:
                raise NotImplementedError(f"Unsupported statement: {type(stmt)}")

        return break_code, continue_code

    def compile(
        self,
        filepath: Optional[str] = None,
        ast: Optional[pycparser.c_ast.FileAST] = None,
    ) -> HmmmProgram:

        if filepath is not None:
            ast = generate_ast(filepath)
        elif ast is None:
            raise ValueError("Must provide either a filepath or an AST")

        self.global_scope = Scope()
        
        self.current_scope = self.global_scope

        main_program = HmmmProgram()

        for child in ast.ext:
            if isinstance(child, pycparser.c_ast.FuncDef):
                if child.decl.name == "main":
                    if isinstance(child.body, pycparser.c_ast.Compound):
                        self.parse_compound(child.body, main_program)

        main_program.add_instruction(generate_instruction("halt"))
        main_program.compile(self.current_scope.get_vars())

        return main_program


def generate_ast(filepath: str) -> pycparser.c_ast.FileAST:
    """Generates an AST from the given C code

    Args:
        code (str) -- The C code to parse

    Returns:
        ast (pycparser.c_ast.FileAST) -- The AST of the C code
    """

    ast = parse_file(
        filepath,
        use_cpp=True,
        cpp_path="clang",
        cpp_args=["-E", r"-Iutils/pycparser/utils/fake_libc_include"],  # type: ignore
    )
    # ast.show(showcoord=True)
    return ast


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Compile a C file into Hmmm assembly")
    argparser.add_argument(
        "filename",
        default="examples/c_files/basic.c",
        nargs="?",
        help="name of file to compile",
    )
    args = argparser.parse_args()

    compiler = Compiler()
    program = compiler.compile(args.filename)

    print(program.to_str())
