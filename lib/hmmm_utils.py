from enum import Enum
from typing import List, Set, Dict, Tuple, Optional, NamedTuple, Union
from dataclasses import dataclass

@dataclass
class MemoryAddress:
    address: int
    hmmm_instruction: "Optional[HmmmInstruction]"

    def get_address(self) -> int:
        if self.hmmm_instruction:
            return self.hmmm_instruction.address.get_address()
        return self.address

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
        temporary_id = f"t{len(temporary_register_dict)}"
        if temporary_id in temporary_register_dict:
            raise ValueError(f"Temporary id {temporary_id} already exists")
    if temporary_id not in temporary_register_dict:
        temporary_register_dict[temporary_id] = TemporaryRegister(
            temporary_id, register
        )
    return temporary_register_dict[temporary_id]

@dataclass
class HmmmInstruction:
    opcode: str
    address: MemoryAddress
    arg1: Optional[Union[HmmmRegister, TemporaryRegister, int, MemoryAddress]]
    arg2: Optional[Union[HmmmRegister, TemporaryRegister, int, MemoryAddress]]
    arg3: Optional[Union[HmmmRegister, TemporaryRegister]]

    def format_arg(self, arg, include_unassigned_registers: bool = False) -> str:
        if arg is None:
            return ""
        elif isinstance(arg, HmmmRegister):
            return arg.value
        elif isinstance(arg, TemporaryRegister):
            if include_unassigned_registers:
                if arg._register:
                    return arg._register.value
                else:
                    return f"{arg._temporary_id}"
            else:
                return arg.get_register().value
        elif isinstance(arg, MemoryAddress):
            return f"{arg.get_address()}"
        elif isinstance(arg, int):
            return str(arg)
        else:
            raise Exception(f"Invalid argument type: {arg}")

    def to_string(self, include_unassigned_registers: bool = False) -> str:
        return f"{self.address.get_address()} {self.opcode} {self.format_arg(self.arg1, include_unassigned_registers)} {self.format_arg(self.arg2, include_unassigned_registers)} {self.format_arg(self.arg3, include_unassigned_registers)}"

    def get_def_use(
        self, constant_registers: dict[str, TemporaryRegister]
    ) -> Tuple[
        List[Union[TemporaryRegister, HmmmRegister]],
        List[Union[TemporaryRegister, HmmmRegister]],
    ]:
        """Gets the temporary registers used by this instruction

        Returns:
            defines, uses -- The temporary registers defined and used by this instruction
        """
        if self.opcode in ["halt", "nop"]:
            return [], []
        elif self.opcode == "read":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            return [self.arg1], []
        elif self.opcode == "write":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            return [], [self.arg1]
        elif self.opcode == "setn":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, int)
            return [self.arg1], []
        elif self.opcode == "addn":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, int)
            return [self.arg1], [self.arg1]
        elif self.opcode == "copy":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (HmmmRegister, TemporaryRegister))
            return [self.arg1], [self.arg2]
        elif self.opcode in ["add", "sub", "mul", "div", "mod"]:
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg3, (HmmmRegister, TemporaryRegister))
            return [self.arg1], [self.arg2, self.arg3]
        elif self.opcode == "neg":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (HmmmRegister, TemporaryRegister))
            return [self.arg1], [self.arg2]
        elif self.opcode == "jumpn":
            assert isinstance(self.arg1, (int, MemoryAddress))
            return [], []
        elif self.opcode == "jumpr":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            return [], [self.arg1]
        elif self.opcode in ["jeqzn", "jnezn", "jltzn", "jgtzn"]:
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (int, MemoryAddress))
            return [], [self.arg1]
        elif self.opcode == "calln":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (int, MemoryAddress))
            return [constant_registers["r13"]], [self.arg1]
        elif self.opcode == "pushr":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (HmmmRegister, TemporaryRegister))
            return [], [self.arg1]
        elif self.opcode == "popr":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (HmmmRegister, TemporaryRegister))
            return [self.arg1], []
        elif self.opcode == "storer":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (HmmmRegister, TemporaryRegister))
            return [], [self.arg1, self.arg2]
        elif self.opcode == "loadr":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, (HmmmRegister, TemporaryRegister))
            return [self.arg1], [self.arg2]
        elif self.opcode == "loadn":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, int)
            return [self.arg1], []
        elif self.opcode == "storen":
            assert isinstance(self.arg1, (HmmmRegister, TemporaryRegister))
            assert isinstance(self.arg2, int)
            return [], [self.arg1]
        else:
            raise Exception(f"Unknown opcode: {self.opcode}")

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
        assert type(arg2) == HmmmRegister or type(arg2) == TemporaryRegister
        assert arg3 == None
        if isinstance(arg2, TemporaryRegister):
            assert arg2.get_register() == HmmmRegister.R15
        if isinstance(arg2, HmmmRegister):
            assert arg2 == HmmmRegister.R15
    elif opcode == "loadn" or opcode == "storen":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == MemoryAddress
        assert arg3 == None
    elif opcode == "loadr" or opcode == "storer":
        assert type(arg1) == HmmmRegister or type(arg1) == TemporaryRegister
        assert type(arg2) == HmmmRegister or type(arg2) == TemporaryRegister
    else:
        raise Exception(f"Invalid opcode: {opcode}")
    return HmmmInstruction(opcode, MemoryAddress(-1, None), arg1, arg2, arg3)

INDEX_TO_REGISTER = {
    0: HmmmRegister.R1,
    1: HmmmRegister.R2,
    2: HmmmRegister.R3,
    3: HmmmRegister.R4,
    4: HmmmRegister.R5,
    5: HmmmRegister.R6,
    6: HmmmRegister.R7,
    7: HmmmRegister.R8,
    8: HmmmRegister.R9,
    9: HmmmRegister.R10,
    10: HmmmRegister.R11,
    11: HmmmRegister.R12,
}
