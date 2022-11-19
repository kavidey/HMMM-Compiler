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


class TemporaryRegister:
    """A temporary register that is used to store values during parsing

    This class is used to store a temporary register that is used to store values during parsing.
    It is used to make sure that the same temporary register is used for the same value.
    """

    def __init__(
        self, temporary_id: object, register: Optional[HmmmRegister] = None
    ) -> None:
        self._register = register
        self._temporary_id = temporary_id

    def set_register(self, register: HmmmRegister) -> None:
        """Sets the register of this temporary register

        Args:
            register (HmmmRegister) -- The register to set
        """
        self._register = register

    def get_register(self) -> HmmmRegister:
        """Gets the register of this temporary register

        Returns:
            HmmmRegister -- The register of this temporary register
        """
        if isinstance(self._register, HmmmRegister):
            return self._register
        raise ValueError(f"TemporaryRegister({self._temporary_id}) has no register")

    def get_temporary_id(self) -> object:
        """Gets the temporary id of this temporary register

        Returns:
            object -- The temporary id of this temporary register
        """
        return self._temporary_id

    def __repr__(self) -> str:
        if self._register:
            return f"TemporaryRegister({self._temporary_id}, {self._register})"
        return f"TemporaryRegister({self._temporary_id})"


def get_temporary_register(
    temporary_register_dict: dict[object, TemporaryRegister],
    temporary_id: Optional[object] = None,
    register: Optional[HmmmRegister] = None,
) -> TemporaryRegister:
    """Gets the temporary register for the given temporary id (will create a new one if it does not exist)

    Args:
        temporary_id (optional) {object} -- The temporary id to get the register for
        register (optional) {HmmmRegister} -- The register to set for the temporary register

    Returns:
        TemporaryRegister -- The temporary register for the given temporary id
    """
    if not temporary_id:
        temporary_id = len(temporary_register_dict)
        if temporary_id in temporary_register_dict:
            raise ValueError(f"Temporary id {temporary_id} already exists")
    if temporary_id not in temporary_register_dict:
        temporary_register_dict[temporary_id] = TemporaryRegister(
            temporary_id, register
        )
    return temporary_register_dict[temporary_id]


@dataclass
class MemoryAddress:
    address: int


@dataclass
class HmmmInstruction:
    opcode: str
    address: MemoryAddress
    arg1: Optional[Union[HmmmRegister, TemporaryRegister, int, MemoryAddress]]
    arg2: Optional[Union[HmmmRegister, TemporaryRegister, int, MemoryAddress]]
    arg3: Optional[Union[HmmmRegister, TemporaryRegister]]

    def format_arg(self, arg):
        if arg is None:
            return ""
        elif isinstance(arg, HmmmRegister):
            return arg.value
        elif isinstance(arg, TemporaryRegister):
            return arg.get_temporary_id()
        elif isinstance(arg, MemoryAddress):
            return f"{arg.address}"
        elif isinstance(arg, int):
            return str(arg)
        else:
            raise Exception(f"Invalid argument type: {arg}")

    def __str__(self):
        return f"{self.address.address} {self.opcode} {self.format_arg(self.arg1)} {self.format_arg(self.arg2)} {self.format_arg(self.arg3)}"


def generate_instruction(
    opcode: str,
    arg1: Optional[Union[HmmmRegister, TemporaryRegister, int, MemoryAddress]] = None,
    arg2: Optional[Union[HmmmRegister, TemporaryRegister, int, MemoryAddress]] = None,
    arg3: Optional[Union[HmmmRegister, TemporaryRegister]] = None,
) -> HmmmInstruction:
    if opcode == "halt" or opcode == "nop":
        assert arg1 == None
        assert arg2 == None
        assert arg3 == None
    elif opcode == "read" or opcode == "write":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert arg2 == None
        assert arg3 == None
    elif opcode == "setn" or opcode == "addn":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == int
        assert arg3 == None
    elif opcode == "copy":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == HmmmRegister or type(arg2) == TemporaryRegister
        assert arg3 == None
    elif (
        opcode == "add"
        or opcode == "sub"
        or opcode == "mul"
        or opcode == "div"
        or opcode == "mod"
    ):
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == HmmmRegister or type(arg2) == TemporaryRegister
        assert type(arg3) == HmmmRegister or type(arg3) == TemporaryRegister
    elif opcode == "neg":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == HmmmRegister or type(arg2) == TemporaryRegister
        assert arg3 == None
    elif opcode == "jumpn":
        assert type(arg1) == MemoryAddress
        assert arg2 == None
        assert arg3 == None
    elif opcode == "jumpr":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert arg2 == None
        assert arg3 == None
    elif (
        opcode == "jeqzn"
        or opcode == "jnezn"
        or opcode == "jgtzn"
        or opcode == "jltzn"
        or opcode == "calln"
    ):
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == MemoryAddress
        assert arg3 == None
    elif opcode == "pushr" or opcode == "popr":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert arg2 == None
        assert arg3 == None
        arg2 = HmmmRegister.R15
    elif opcode == "loadn" or opcode == "storen":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == MemoryAddress
        assert arg3 == None
    elif opcode == "loadr" or opcode == "storer":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == HmmmRegister or type(arg2) == TemporaryRegister
    return HmmmInstruction(opcode, MemoryAddress(-1), arg1, arg2, arg3)


class HmmmProgram:
    def __init__(self):
        self.code: List[Union[HmmmInstruction, str]] = []
        self.compiled = False

    def add_instruction(self, instruction: HmmmInstruction):
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.code.append(instruction)

    def add_instructions(self, instructions: List[HmmmInstruction]):
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.code.extend(instructions)

    def add_comment(self, comment: str):
        self.code.append(f"# {comment}")

    def add_stack_pointer_code(self):
        """
        This function should only be called once, after the entire program has been generated.
        It adds code to the beginning of the program to initialize the stack pointer.
        """
        if self.compiled:
            raise Exception("Cannot add instruction to compiled program")
        self.code.insert(
            0, generate_instruction("setn", HmmmRegister.R15, len(self.code) + 1)
        )
    
    def compile(self):
        if self.compiled:
            raise Exception("Cannot compile program twice")
        
        for instruction in self.code:
            if isinstance(instruction, HmmmInstruction):
                if isinstance(instruction.arg1, TemporaryRegister) and not instruction.arg1.get_register():
                    raise Exception(f"Temporary register {instruction.arg1.get_temporary_id()} is not assigned")
                if isinstance(instruction.arg2, TemporaryRegister) and not instruction.arg2.get_register():
                    raise Exception(f"Temporary register {instruction.arg2.get_temporary_id()} is not assigned")
                if isinstance(instruction.arg3, TemporaryRegister) and not instruction.arg3.get_register():
                    raise Exception(f"Temporary register {instruction.arg3.get_temporary_id()} is not assigned")

        self.add_stack_pointer_code()
        self.assign_line_numbers()
        self.compiled = True

    def assign_line_numbers(self):
        for i, instruction in enumerate(self.code):
            if isinstance(instruction, HmmmInstruction):
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
