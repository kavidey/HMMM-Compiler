from enum import Enum
from typing import List, Set, Dict, Tuple, Optional, NamedTuple, Union

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

class MemoryAddress(NamedTuple):
    line: int

class HmmmInstruction(NamedTuple):
    opcode: str
    line: MemoryAddress
    arg1: Optional(Union(HmmmRegister, int))
    arg2: Optional(Union(HmmmRegister, int, MemoryAddress))
    arg3: Optional(HmmmRegister)

    def __str__(self):
        return f"{self.line.line} {self.opcode} {self.arg1} {self.arg2} {self.arg3}"

def generate_instruction(opcode: str, arg1: Optional(Union(HmmmRegister, int)), arg2: Optional(Union(HmmmRegister, int, MemoryAddress)), arg3: Optional(HmmmRegister)) -> HmmmInstruction:
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
        assert arg2 == HmmmRegister
        assert arg3 == None
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
        self.code: List(HmmmInstruction) = []
    
    def add_instruction(self, instruction: HmmmInstruction):
        self.code.append(instruction)
    
    def add_instructions(self, instructions: List(HmmmInstruction)):
        self.code.extend(instructions)

    def assign_line_numbers(self):
        for i, instruction in enumerate(self.code):
            instruction.line = MemoryAddress(i)
    
    def __getitem__(self, index: int):
        return self.code[index]
    
    def __str__(self):
        return "\n".join([str(instruction) for instruction in self.code])