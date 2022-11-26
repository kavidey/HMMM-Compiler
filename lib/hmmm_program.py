from enum import Enum
from typing import List, Set, Dict, Tuple, Optional, NamedTuple, Union
from dataclasses import dataclass
from lib.graph import DirectedGraph

from lib.hmmm_utils import (
    HmmmRegister,
    MemoryAddress,
    TemporaryRegister,
    HmmmInstruction,
    generate_instruction,
)

class HmmmProgram:
    def __init__(self):
        self.instructions: List[HmmmInstruction] = []
        self.compiled = False

    def add_instruction(self, instruction: HmmmInstruction):
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.instructions.append(instruction)

    def add_instruction_before(self, instruction: HmmmInstruction, before: HmmmInstruction):
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.assign_line_numbers()
        self.instructions.insert(self.instructions.index(before), instruction)
    
    def add_instruction_after(self, instruction: HmmmInstruction, after: HmmmInstruction):
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.assign_line_numbers()
        self.instructions.insert(self.instructions.index(after) + 1, instruction)

    def add_instructions(self, instructions: List[HmmmInstruction]):
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.instructions.extend(instructions)

    def add_function(self, function: "HmmmProgram"):
        self.add_instructions(function.instructions)

    def add_stack_pointer_code(self):
        """
        This function should only be called once, after the entire program has been generated.
        It adds code to the beginning of the program to initialize the stack pointer.
        """
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.instructions.insert(
            0, generate_instruction("setn", HmmmRegister.R15, len(self.instructions) + 1)
        )

    def compile(self):
        if self.compiled:
            raise Exception("Cannot compile program twice")

        self.add_stack_pointer_code()
        self.assign_line_numbers()
        self.compiled = True

    def assign_line_numbers(self):
        for i, instruction in enumerate(self.instructions):
            if isinstance(instruction, HmmmInstruction):
                instruction.address.address = i

    def __getitem__(self, index: int):
        return self.instructions[index]

    def __str__(self):
        return "\n".join(self.to_array())

    def to_str(self):
        return str(self)

    def to_array(self):
        output = []
        for instruction in self.instructions:
            output.append(instruction.to_string(include_unassigned_registers=(not self.compiled)))
        return output

    def run_liveness_analysis(self, temporary_registers: List[TemporaryRegister], live_in: List[TemporaryRegister] = []):
        """Run liveness analysis on the program
        
        Args:
            temporary_registers (List[TemporaryRegister]): List of temporary registers used in the program
            live_in (List[TemporaryRegister], optional): List of temporary registers that are live in to the program. Defaults to [].

        Returns:
            LivenessGraph: Directed control flow graph with liveness information
        """

         # This makes sure lines that do the identical thing are treated separately because they will have different addresses
        self.assign_line_numbers()

        control_flow_graph = DirectedGraph()

        # Add temporary registers to the graph
        for register in temporary_registers:
            control_flow_graph.add_temporary(register)

        # Add instructions to the graph
        constant_registers: dict[str, TemporaryRegister] = {}
        for register in temporary_registers:
            if register._register == HmmmRegister.R13:
                constant_registers["r13"] = register
            elif register._register == HmmmRegister.R14:
                constant_registers["r14"] = register
            elif register._register == HmmmRegister.R15:
                constant_registers["r15"] = register

        for i in range(len(self.instructions)):
            instruction = self.instructions[i]
            if isinstance(instruction, HmmmInstruction):
                defines, uses = instruction.get_def_use(constant_registers)

                if i == 0:
                    defines += live_in

                control_flow_graph.add_node(instruction, defines=defines, uses=uses, start_node=True if i == 0 else False)  # type: ignore

        # Add edges to the graph
        for i in range(len(self.instructions) - 1):
            instruction = self.instructions[i]
            if isinstance(instruction, HmmmInstruction):
                control_flow_graph.add_edge(instruction, self.instructions[i + 1])  # type: ignore
                if instruction.opcode in ["jumpn", "jumpr"]:
                    control_flow_graph.add_edge(instruction, self.instructions[instruction.arg1.get_address()])  # type: ignore
                elif instruction.opcode in [
                    "jeqzn",
                    "jnezn",
                    "jgtzn",
                    "jltzn",
                ]:
                    control_flow_graph.add_edge(instruction, self.instructions[instruction.arg2.get_address()])  # type: ignore
        
        # Run liveness analysis
        control_flow_graph.calculate_liveness()

        return control_flow_graph

    def assign_registers(self, temporary_registers: List[TemporaryRegister]):
        """
        Assign registers to temporary registers.

        This function should only be called once, after the entire program has been generated.
        """
        if self.compiled:
            raise Exception("Cannot assign registers to compiled program")
        
        # Run liveness analysis
        control_flow_graph = self.run_liveness_analysis(temporary_registers)

        # Assign registers
        interference_graph = control_flow_graph.generate_interference_graph(
            [
                HmmmRegister.R1,
                HmmmRegister.R2,
                HmmmRegister.R3,
                HmmmRegister.R4,
                HmmmRegister.R5,
                HmmmRegister.R6,
                HmmmRegister.R7,
                HmmmRegister.R8,
                HmmmRegister.R9,
                HmmmRegister.R10,
                HmmmRegister.R11,
                HmmmRegister.R12,
            ]
        )

        colored_temporaries = interference_graph.assign_registers()
        for colored_temporary in colored_temporaries:
            assert isinstance(colored_temporary.name, TemporaryRegister)
            assert isinstance(colored_temporary.color, HmmmRegister)

            colored_temporary.name.set_register(colored_temporary.color)

        to_remove = []
        for i in range(len(self.instructions)):
            instruction = self.instructions[i]
            assert isinstance(instruction, HmmmInstruction)
            if instruction.opcode == "copy":
                assert isinstance(instruction.arg1, TemporaryRegister)
                assert isinstance(instruction.arg2, TemporaryRegister)

                if instruction.arg1.get_register() == instruction.arg2.get_register():
                    to_remove.append(i)
        
        self.instructions = [ele for idx, ele in enumerate(self.instructions) if idx not in to_remove]

        for instruction in self.instructions:
            if isinstance(instruction, HmmmInstruction):
                if (
                    isinstance(instruction.arg1, TemporaryRegister)
                    and not instruction.arg1.get_register()
                ):
                    raise Exception(
                        f"Temporary register {instruction.arg1.get_temporary_id()} is not assigned"
                    )
                if (
                    isinstance(instruction.arg2, TemporaryRegister)
                    and not instruction.arg2.get_register()
                ):
                    raise Exception(
                        f"Temporary register {instruction.arg2.get_temporary_id()} is not assigned"
                    )
                if (
                    isinstance(instruction.arg3, TemporaryRegister)
                    and not instruction.arg3.get_register()
                ):
                    raise Exception(
                        f"Temporary register {instruction.arg3.get_temporary_id()} is not assigned"
                    )

if __name__ == "__main__":
    print("Simple Adder")
    simple_adder_program = HmmmProgram()
    simple_adder_program.add_instruction(generate_instruction("read", HmmmRegister.R1))
    simple_adder_program.add_instruction(generate_instruction("read", HmmmRegister.R2))
    simple_adder_program.add_instruction(
        generate_instruction("add", HmmmRegister.R3, HmmmRegister.R1, HmmmRegister.R2)
    )
    simple_adder_program.add_instruction(generate_instruction("write", HmmmRegister.R3))
    simple_adder_program.add_instruction(generate_instruction("halt"))
    simple_adder_program.assign_line_numbers()
    print(simple_adder_program)
    print("")
    print("Minimum")
    minimum_program = HmmmProgram()
    minimum_program.add_instruction(generate_instruction("read", HmmmRegister.R1))
    minimum_program.add_instruction(generate_instruction("read", HmmmRegister.R2))
    minimum_program.add_instruction(
        generate_instruction("sub", HmmmRegister.R3, HmmmRegister.R1, HmmmRegister.R2)
    )
    write_r2 = generate_instruction("write", HmmmRegister.R2)
    minimum_program.add_instruction(
        generate_instruction("jgtzn", HmmmRegister.R3, write_r2.address)
    )
    minimum_program.add_instruction(generate_instruction("write", HmmmRegister.R1))
    halt = generate_instruction("halt")
    minimum_program.add_instruction(generate_instruction("jumpn", halt.address))
    minimum_program.add_instruction(write_r2)
    minimum_program.add_instruction(halt)
    minimum_program.assign_line_numbers()
    print(minimum_program)
