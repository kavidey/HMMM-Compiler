import unittest
from compile import Compiler
from pathlib import Path

import os, sys
from io import StringIO
import importlib


stdin_inputs = []

# Copied from: https://stackoverflow.com/a/45669280
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Copied from: https://stackoverflow.com/a/16571630
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class TestProgram(unittest.TestCase):
    def run_file(self, filename, inputs=[]):
        global stdin_inputs

        # Import the simulator
        import utils.hmmm as hmmm_simulator

        # It needs to be reloaded each time its used
        importlib.reload(hmmm_simulator)

        # Monkey patch the read function to use our custom function (instead of reading from terminal it reads from stdin_inputs)
        def custom_op_read(args):
            global stdin_inputs
            if len(stdin_inputs) == 0:
                raise Exception("No more inputs to read")
            hmmm_simulator.registers[args[0]] = stdin_inputs.pop(0)

        hmmm_simulator.implementations["read"] = custom_op_read

        compiler = Compiler()
        program = compiler.compile(str(Path("sample_files") / filename))

        with HiddenPrints():
            machine_code = hmmm_simulator.programToMachineCode(
                hmmm_simulator.hmmmAssembler(program.to_array())
            )
            hmmm_simulator.convertMachineCode(machine_code)

        if len(inputs) > 0:
            stdin_inputs = []
            for i in inputs:
                stdin_inputs.append(i)

        with Capturing() as output:
            hmmm_simulator.runHmmm()

        output = [int(x) for x in output]
        return output


class TestMath(TestProgram):
    FILENAME = "math.c"

    def testA(self):
        assert self.run_file(self.FILENAME) == [18]


class TestMaxOfThree(TestProgram):
    FILENAME = "max_3.c"

    def testA(self):
        assert self.run_file(self.FILENAME, [0, 1, 2]) == [2]

    def testB(self):
        assert self.run_file(self.FILENAME, [1, 2470, 121]) == [2470]


class TestPosNegZero(TestProgram):
    FILENAME = "pos_neg_zero.c"

    def testA(self):
        assert self.run_file(self.FILENAME, [300]) == [1]

    def testB(self):
        assert self.run_file(self.FILENAME, [0]) == [0]

    def testC(self):
        assert self.run_file(self.FILENAME, [-152]) == [-1]


class TestSumUpTo(TestProgram):
    FILENAME = "sum_up_to.c"

    def testA(self):
        assert self.run_file(self.FILENAME, [3]) == [6]

    def testB(self):
        assert self.run_file(self.FILENAME, [10]) == [55]


class TestFibonacci(TestProgram):
    FILENAME = "fibonacci.c"

    def testA(self):
        assert self.run_file(self.FILENAME, [0]) == []

    def testB(self):
        assert self.run_file(self.FILENAME, [10]) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


class TestPrimesBetween(TestProgram):
    FILENAME = "primes_between.c"

    def testA(self):
        assert self.run_file(self.FILENAME, [1, 9]) == [2, 3, 5, 7]

    def testB(self):
        assert self.run_file(self.FILENAME, [82, 98]) == [83, 89, 97]

class TestDoubleFunc(TestProgram):
    FILENAME = "double_func.c"

    def testA(self):
        assert self.run_file(self.FILENAME, [24]) == [48, 6]

    def testB(self):
        assert self.run_file(self.FILENAME, [17]) == [34, 6]


if __name__ == "__main__":
    unittest.main()
