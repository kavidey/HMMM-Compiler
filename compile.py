import argparse
import copy
import re
from typing import List, Tuple, Union, Optional, TypedDict

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
    INDEX_TO_REGISTER,
)


class Scope:
    def __init__(self, parent_scope: "Optional[Scope]" = None) -> None:
        self.scope: dict[object, TemporaryRegister] = {}
        self.parent_scope = parent_scope

        for register in HmmmRegister:
            self.scope[register] = TemporaryRegister(register, register)

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

    def __repr__(self) -> str:
        return str(self.scope)


class Function(TypedDict):
    program: HmmmProgram
    args: List
    body: pycparser.c_ast.Compound
    jump_location: HmmmInstruction


class Compiler:
    def __init__(self) -> None:
        self.is_in_loop = False
        self.is_in_function = False
        self.function_calls: List[Tuple[str, HmmmInstruction, HmmmInstruction, HmmmInstruction]] = [] # (name, push before calln, calln instruction, pop after calln)

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

    def parse_expression(
        self,
        expr: object,
        program: HmmmProgram,
        result: Optional[TemporaryRegister] = None,
    ) -> TemporaryRegister:
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
        elif isinstance(expr, pycparser.c_ast.FuncCall):
            func_call_result = self.parse_func_call(expr, program)
            program.add_instruction(
                generate_instruction("copy", result, func_call_result)
            )
        else:
            raise NotImplementedError(
                f"Expression type {type(expr)} is not implemented"
            )

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
        self,
        node: pycparser.c_ast.If,
        program: HmmmProgram,
    ) -> Tuple[List[HmmmInstruction], List[HmmmInstruction], List[HmmmInstruction]]:
        """Parses an if statement and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.If) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to

        Returns:
            break_code (List[HmmmInstruction]) -- The instructions to execute when a break statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
            continue_code (List[HmmmInstruction]) -- The instructions to execute when a continue statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
            return_code (List[HmmmInstruction]) -- The instructions to execute when a return statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
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
        break_statements, continue_statements, return_statements = self.parse_compound(
            node.iftrue, program
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
                self.parse_compound(node.iffalse, program)
            elif isinstance(node.iffalse, pycparser.c_ast.If):
                self.parse_if(node.iffalse, program)
        else:
            jump_end = jump_iffalse

        end_of_if_block = generate_instruction("nop")
        program.add_instruction(end_of_if_block)
        jump_end.arg1 = MemoryAddress(-1, end_of_if_block)

        return break_statements, continue_statements, return_statements

    def parse_while(
        self, node: pycparser.c_ast.While, program: HmmmProgram
    ) -> List[HmmmInstruction]:
        """Parses a while loop and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.While) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to

        Returns:
            return_statements (List[HmmmInstruction]) -- The instructions to execute when a return statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
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

        already_in_loop = self.is_in_loop
        self.is_in_loop = True
        break_statements, continue_statements, return_statements = self.parse_compound(
            node.stmt, program
        )
        self.is_in_loop = already_in_loop

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

        return return_statements

    def parse_for(
        self, node: pycparser.c_ast.While, program: HmmmProgram
    ) -> List[HmmmInstruction]:
        """Parses a for loop and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.While) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to

        Returns:
            return_statements (List[HmmmInstruction]) -- The instructions to execute when a return statement is encountered. The user needs to set the jump address of these manually and passed up to any outer loops
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

        already_in_loop = self.is_in_loop
        self.is_in_loop = True
        break_statements, continue_statements, return_statements = self.parse_compound(
            node.stmt, program
        )
        self.is_in_loop = already_in_loop

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

        return return_statements

    def parse_func_call(
        self, node: pycparser.c_ast.FuncCall, program: HmmmProgram
    ) -> TemporaryRegister:
        assert (
            node.name.name in self.functions
        ), f"Function {node.name.name} not defined"

        args_given = len(node.args.exprs)
        args_needed = len(self.functions[node.name.name]["args"])
        assert (
            args_given == args_needed
        ), f"Function {node.name.name} takes {args_needed} arguments, but {args_given} were given"

        func_call_details = (node.name.name, program.instructions[-1])

        # Move the arguments into the correct registers
        for i, arg in enumerate(node.args.exprs):
            arg_temp = self.parse_expression(arg, program)
            program.add_instruction(
                generate_instruction(
                    "copy",
                    self.current_scope[INDEX_TO_REGISTER[i]],
                    arg_temp,
                )
            )

        if self.is_in_function:
            program.add_instruction(
                generate_instruction("pushr", self.current_scope[HmmmRegister.R14], self.current_scope[HmmmRegister.R15])
            )

        # Call the function
        program.add_instruction(
            generate_instruction(
                "calln",
                self.current_scope[HmmmRegister.R14],
                MemoryAddress(-1, self.functions[node.name.name]["jump_location"]),
            )
        )
        func_call_details += (program.instructions[-1],)

        if self.is_in_function:
            program.add_instruction(
                generate_instruction("popr", self.current_scope[HmmmRegister.R14], self.current_scope[HmmmRegister.R15])
            )

        # Move the return value into a temporary
        return_temp = self.current_scope.make_temporary()
        program.add_instruction(
            generate_instruction(
                "copy", return_temp, self.current_scope[HmmmRegister.R13]
            )
        )
        func_call_details += (program.instructions[-1],)
        self.function_calls.append(func_call_details)

        return return_temp

    def parse_compound(
        self,
        node: pycparser.c_ast.Compound,
        program: HmmmProgram,
        is_main: bool = False,
    ) -> Tuple[List[HmmmInstruction], List[HmmmInstruction], List[HmmmInstruction]]:
        """Parses a compound statement and adds the appropriate instructions to the given program

        Args:
            node (pycparser.c_ast.Compound) -- The node to parse
            program (HmmmProgram) -- The program to add instructions to
            is_main (bool, optional) -- Whether or not this is the main function. Defaults to False.

        Returns:
            break_code (List[HmmmInstruction]) -- The instructions to execute when a break statement is encountered. The user needs to set the jump address of these manually
            continue_code (List[HmmmInstruction]) -- The instructions to execute when a continue statement is encountered. The user needs to set the jump address of these manually
            return_code (List[HmmmInstruction]) -- The instructions to execute when a return statement is encountered. The user needs to set the jump address of these manually
        """

        break_code = []
        continue_code = []
        return_code = []

        for stmt in node.block_items:
            # If the user is declaring a new variable
            if isinstance(stmt, pycparser.c_ast.Decl) or isinstance(
                stmt, pycparser.c_ast.Assignment
            ):
                self.parse_decl_assign(stmt, program)
            elif isinstance(stmt, pycparser.c_ast.If):
                break_statements, if_statements, return_statements = self.parse_if(
                    stmt, program
                )
                break_code += break_statements
                continue_code += if_statements
                return_code += return_statements
            elif isinstance(stmt, pycparser.c_ast.While):
                return_statements = self.parse_while(stmt, program)
                return_code += return_statements
            elif isinstance(stmt, pycparser.c_ast.For):
                return_statements = self.parse_for(stmt, program)
                return_code += return_statements
            elif isinstance(stmt, pycparser.c_ast.Continue):
                if self.is_in_loop:
                    continue_statement = generate_instruction(
                        "jumpn", MemoryAddress(-1, None)
                    )
                    program.add_instruction(continue_statement)
                    continue_code.append(continue_statement)
                else:
                    raise Exception("Continue statement not in a loop")
            elif isinstance(stmt, pycparser.c_ast.Break):
                if self.is_in_loop:
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
                ] + list(self.functions.keys()), "Function not defined"
                if stmt.name.name == "printf":
                    assert (
                        stmt.args.exprs[0].value == '"%d\\n"'
                    ), 'Only printf("%d\\n", var_to_print) is supported. First input must be "%d\\n"'
                    assert (
                        len(stmt.args.exprs) == 2
                    ), 'Only printf("%d\\n", var_to_print) is supported. Must have exactly 2 inputs'

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
                else:
                    self.parse_func_call(stmt, program)
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
                if is_main:
                    program.add_instruction(generate_instruction("halt"))
                else:
                    if self.is_in_function:
                        return_result = self.parse_expression(stmt.expr, program)
                        program.add_instruction(
                            generate_instruction(
                                "copy",
                                self.current_scope[HmmmRegister.R13],
                                return_result,
                            )
                        )
                        return_statement = generate_instruction(
                            "jumpr", self.current_scope[HmmmRegister.R14]
                        )
                        program.add_instruction(return_statement)
                        return_code.append(return_statement)
                    else:
                        raise Exception("Return statement not in a function")
            else:
                raise NotImplementedError(f"Unsupported statement: {type(stmt)}")

        return break_code, continue_code, return_code

    def save_variables_during_function_calls(self, program: HmmmProgram, live_in: List[TemporaryRegister] = []):
        """Saves the values of all variables in the current scope to the stack

        This should be called *after* register allocation has been completed

        Args:
            program (HmmmProgram) -- The program to add instructions to
        """
        program.assign_line_numbers()

        liveness = program.run_liveness_analysis(self.current_scope.get_vars(), live_in=live_in)
        # add_before_push: List[Tuple[HmmmInstruction, HmmmInstruction]] = []
        add_after_push: List[Tuple[HmmmInstruction, HmmmInstruction]] = []
        add_after_pop: List[Tuple[HmmmInstruction, HmmmInstruction]] = []
        for func_call in self.function_calls:

            # This gets the list of variables that were live before any registers were setup for the function call
            live_before_func_call = liveness.get_node_by_name(func_call[1]).live_in

            # This gets the list of variables that were live after the result of the function call was moved from R13 to a temporary
            live_after_func_call = liveness.get_node_by_name(func_call[3]).live_in

            # live = live_before_func_call+live_after_func_call
            live = live_after_func_call
            live = set([register for register in live if register._register not in [HmmmRegister.R14, HmmmRegister.R15]])

            for register in live:
                add_after_push.append((func_call[1], generate_instruction("pushr", register, self.current_scope[HmmmRegister.R15])))
                add_after_pop.insert(0, (func_call[3], generate_instruction("popr", register, self.current_scope[HmmmRegister.R15])))

        # for instruction, new_instruction in add_before_push:
        #     program.add_instruction_before(instruction, new_instruction)
        for instruction, new_instruction in add_after_push:
            program.add_instruction_after(new_instruction, instruction)
        for instruction, new_instruction in add_after_pop:
            program.add_instruction_after(new_instruction, instruction)


    def compile(
        self,
        filepath: Optional[str] = None,
        ast: Optional[pycparser.c_ast.FileAST] = None,
    ) -> HmmmProgram:

        ### Get the AST ###
        if filepath is not None:
            ast = generate_ast(filepath)
        elif ast is None:
            raise ValueError("Must provide either a filepath or an AST")

        self.global_scope = Scope()

        ### Parse functions ###

        ## Generate a dict to store the parsed functions in ##
        # Functions are parsed in two passes, so that if one function calls another (or itself) jump location for that function has already been generated
        self.functions: dict[str, Function] = {}
        for func in ast.ext:
            if isinstance(func, pycparser.c_ast.FuncDef) and func.decl.name != "main":
                func_program = HmmmProgram()

                func_program.add_instruction(generate_instruction("nop"))

                self.functions[func.decl.name] = {
                    "program": func_program,
                    "args": func.decl.type.args.params,
                    "body": func.body,
                    "jump_location": func_program.instructions[0],
                }

        for func in self.functions:
            self.current_scope = Scope(self.global_scope)

            # Add all the function's arguments to the current scope
            for i, decl in enumerate(self.functions[func]["args"]):
                assert (
                    decl.type.type.names[0] == "int"
                ), "Only int variables are supported"

                self.current_scope.declar_var(decl.name, INDEX_TO_REGISTER[i])

            self.is_in_function = True
            self.function_calls = []

            self.parse_compound(
                self.functions[func]["body"],
                self.functions[func]["program"],
            )
            self.save_variables_during_function_calls(self.functions[func]["program"], live_in=[self.current_scope[INDEX_TO_REGISTER[i]] for i in range(len(self.functions[func]["args"]))])
            self.functions[func]["program"].assign_registers(
                self.current_scope.get_vars()
            )

        # Process the code in the main function
        self.current_scope = self.global_scope
        self.is_in_function = False
        self.function_calls = []

        main_program = HmmmProgram()
        main_program.add_instruction(generate_instruction("nop"))

        for child in ast.ext:
            if isinstance(child, pycparser.c_ast.FuncDef):
                if child.decl.name == "main":
                    if isinstance(child.body, pycparser.c_ast.Compound):
                        self.parse_compound(child.body, main_program, is_main=True)

        self.save_variables_during_function_calls(main_program)
        print(main_program)
        print()
        main_program.assign_registers(self.current_scope.get_vars())

        # Add pre-compiled functions to the main program
        for func in self.functions:
            main_program.add_function(self.functions[func]["program"])

        main_program.compile()

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
