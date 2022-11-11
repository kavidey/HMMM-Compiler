from enum import Enum
from typing import List, Set, Dict, Tuple, Optional, NamedTuple, Union
from dataclasses import dataclass

# Documentation for Hmmm is here: https://www.cs.hmc.edu/~cs5grad/cs5/hmmm/documentation/documentation.html

class HmmmRegister(Enum):
    R0 = "r0"
    R1 = "r1"
    R2 = "r2"
    R3 = "r3"
    R4 = "r4"
    R5 = "r5"
    R6 = "r6"
    R7 = "r7"
    R8 = "r8"
    R9 = "r9"
    R10 = "r10"
    R11 = "r11"
    R12 = "r12"
    R13 = "r13"
    R14 = "r14"
    R15 = "r15"

@dataclass
class MemoryAddress:
    address: int

@dataclass
class HmmmInstruction:
    opcode: str
    address: MemoryAddress
    arg1: Optional[Union[HmmmRegister, int, MemoryAddress]]
    arg2: Optional[Union[HmmmRegister, int, MemoryAddress]]
    arg3: Optional[HmmmRegister]

    def format_arg(self, arg):
        if arg is None:
            return ""
        elif isinstance(arg, HmmmRegister):
            return arg.value
        elif isinstance(arg, MemoryAddress):
            return f"{arg.address}"
        elif isinstance(arg, int):
            return str(arg)
        else:
            raise Exception(f"Invalid argument type: {arg}")

    def __str__(self):
        return f"{self.address.address} {self.opcode} {self.format_arg(self.arg1)} {self.format_arg(self.arg2)} {self.format_arg(self.arg3)}"

def generate_instruction(opcode: str, arg1: Optional[Union[HmmmRegister, int, MemoryAddress]] = None, arg2: Optional[Union[HmmmRegister, int, MemoryAddress]] = None, arg3: Optional[HmmmRegister] = None) -> HmmmInstruction:
    if opcode == "halt" or opcode == "nop":
        assert arg1 == None
        assert arg2 == None
        assert arg3 == None
    elif opcode == "read" or opcode == "write":
        assert type(arg1) == HmmmRegister
        assert arg2 == None
        assert arg3 == None
    elif opcode == "setn" or opcode == "addn":
        assert type(arg1) == HmmmRegister
        assert type(arg2) == int
        assert arg3 == None
    elif opcode == "copy":
        assert type(arg1) == HmmmRegister
        assert type(arg2) == HmmmRegister
        assert arg3 == None
    elif opcode == "add" or opcode == "sub" or opcode == "mul" or opcode == "div" or opcode == "mod":
        assert type(arg1) == HmmmRegister
        assert type(arg2) == HmmmRegister
        assert type(arg3) == HmmmRegister
    elif opcode == "neg":
        assert type(arg1) == HmmmRegister
        assert type(arg2) == HmmmRegister
        assert arg3 == None
    elif opcode == "jumpn":
        assert type(arg1) == MemoryAddress
        assert arg2 == None
        assert arg3 == None
    elif opcode == "jumpr":
        assert type(arg1) == HmmmRegister
        assert arg2 == None
        assert arg3 == None
    elif opcode == "jeqzn" or opcode == "jnezn" or opcode == "jgtzn" or opcode == "jltzn" or opcode == "calln":
        assert type(arg1) == HmmmRegister
        assert type(arg2) == MemoryAddress
        assert arg3 == None
    elif opcode == "pushr" or opcode == "popr":
        assert type(arg1) == HmmmRegister
        assert arg2 == None
        assert arg3 == None
        arg2 = HmmmRegister.R15
    elif opcode == "loadn" or opcode == "storen":
        assert type(arg1) == HmmmRegister
        assert type(arg2) == MemoryAddress
        assert arg3 == None
    elif opcode == "loadr" or opcode == "storer":
        assert type(arg1) == HmmmRegister
        assert type(arg2) == HmmmRegister
    return HmmmInstruction(opcode, MemoryAddress(-1), arg1, arg2, arg3)
    

class HmmmProgram:
    def __init__(self):
        self.code: List[HmmmInstruction] = []
    
    def add_instruction(self, instruction: HmmmInstruction):
        self.code.append(instruction)
    
    def add_instructions(self, instructions: List[HmmmInstruction]):
        self.code.extend(instructions)
    
    def add_comment(self, comment: str):
        self.code.append(f"# {comment}")

    def add_stack_pointer_code(self):
        """
        This function should only be called once, after the entire program has been generated.
        It adds code to the beginning of the program to initialize the stack pointer.
        """
        self.code.insert(0, generate_instruction("setn", HmmmRegister.R15, len(self.code)+1))
    
    def assign_line_numbers(self):
        for i, instruction in enumerate(filter(lambda instruction: type(instruction) == HmmmInstruction, self.code)):
            instruction.address.address = i
    
    def __getitem__(self, index: int):
        return self.code[index]
    
    def __str__(self):
        return "\n".join([str(instruction) for instruction in self.code])

    def to_str(self):
        return str(self)
    
    def to_array(self):
        return [str(instruction) for instruction in self.code]

if __name__ == "__main__":
    print("Simple Adder")
    simple_adder_program = HmmmProgram()
    simple_adder_program.add_instruction(generate_instruction("read", HmmmRegister.R1))
    simple_adder_program.add_instruction(generate_instruction("read", HmmmRegister.R2))
    simple_adder_program.add_instruction(generate_instruction("add", HmmmRegister.R3, HmmmRegister.R1, HmmmRegister.R2))
    simple_adder_program.add_instruction(generate_instruction("write", HmmmRegister.R3))
    simple_adder_program.add_instruction(generate_instruction("halt"))
    simple_adder_program.assign_line_numbers()
    print(simple_adder_program)
    print("")
    print("Minimum")
    minimum_program = HmmmProgram()
    minimum_program.add_instruction(generate_instruction("read", HmmmRegister.R1))
    minimum_program.add_instruction(generate_instruction("read", HmmmRegister.R2))
    minimum_program.add_instruction(generate_instruction("sub", HmmmRegister.R3, HmmmRegister.R1, HmmmRegister.R2))
    write_r2 = generate_instruction("write", HmmmRegister.R2)
    minimum_program.add_instruction(generate_instruction("jgtzn", HmmmRegister.R3, write_r2.address))
    minimum_program.add_instruction(generate_instruction("write", HmmmRegister.R1))
    halt = generate_instruction("halt")
    minimum_program.add_instruction(generate_instruction("jumpn", halt.address))
    minimum_program.add_instruction(write_r2)
    minimum_program.add_instruction(halt)
    minimum_program.assign_line_numbers()
    print(minimum_program)