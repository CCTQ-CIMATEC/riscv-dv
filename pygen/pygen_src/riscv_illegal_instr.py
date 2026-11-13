
"""
Copyright 2020 Google LLC
Copyright 2020 PerfectVIPs Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""

import vsc
import random
import logging
from enum import IntEnum, auto
from importlib import import_module
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.riscv_instr_pkg import riscv_instr_group_t, privileged_reg_t
rcs = import_module("pygen_src.target." + cfg.argv.target + ".riscv_core_setting")


# ---------------------------------------------------------------------------------------------
# This class is used to generate illegal or HINT instructions.
# The illegal instruction will be generated in binary format and mixed with other valid instr.
# The mixed instruction stream will be stored in data section and loaded to instruction pages
# at run time.
#
# Port of src/riscv_illegal_instr.sv. pyvsc builds the constraints when the object is created,
# so every list they use (legal opcodes, CSRs) is complete in __init__, and the constraints use
# the vsc operators (&, |, ~, implies): Python's and/or/not do not build vsc expressions.
# ---------------------------------------------------------------------------------------------


class illegal_instr_type_e(IntEnum):
    kIllegalOpcode = 0
    kIllegalCompressedOpcode = auto()
    kIllegalFunc3 = auto()
    kIllegalFunc7 = auto()
    kReservedCompressedInstr = auto()
    kHintInstr = auto()
    kIllegalSystemInstr = auto()


class reserved_c_instr_e(IntEnum):
    kIllegalCompressed = 0
    kReservedAddispn = auto()
    kReservedAddiw = auto()
    kReservedAddi16sp = auto()
    kReservedLui = auto()
    kReservedLwsp = auto()
    kReservedLdsp = auto()
    kReservedLqsp = auto()
    kReservedJr = auto()
    kReservedC0 = auto()
    kReservedC1 = auto()
    kReservedC2 = auto()


def supports_compressed():
    return any(group in rcs.supported_isa for group in
               (riscv_instr_group_t.RV32C, riscv_instr_group_t.RV64C,
                riscv_instr_group_t.RV128C))


@vsc.randobj
class riscv_illegal_instr:
    def __init__(self):
        self.comment = ""
        self.exception = vsc.rand_enum_t(illegal_instr_type_e)
        self.reserved_c = vsc.rand_enum_t(reserved_c_instr_e)
        self.instr_bin = vsc.rand_bit_t(32)
        self.opcode = vsc.rand_bit_t(7)
        self.compressed = vsc.rand_bit_t(1)
        self.func3 = vsc.rand_bit_t(3)
        self.func7 = vsc.rand_bit_t(7)
        self.has_func3 = vsc.rand_bit_t(1)
        self.has_func7 = vsc.rand_bit_t(1)
        self.c_op = vsc.rand_bit_t(2)
        self.c_msb = vsc.rand_bit_t(3)
        # Every CSR the generator knows, implemented or not: an illegal CSR access uses an
        # address outside this list, so no ISS implements it
        self.csrs = [csr.value for csr in privileged_reg_t] + list(rcs.custom_csr)
        # Default legal opcodes for RV32I instructions
        self.legal_opcode = [0b0000011, 0b0001111, 0b0010011, 0b0010111, 0b0100011,
                             0b0110111, 0b1100011, 0b0110011, 0b1100111, 0b1110011,
                             0b1101111]
        # Default legal opcodes for RV32C instructions
        self.legal_c00_opcode = [0b000, 0b010, 0b110]
        self.legal_c10_opcode = [0b000, 0b010, 0b100, 0b110]
        isa = rcs.supported_isa
        if riscv_instr_group_t.RV32F in isa or riscv_instr_group_t.RV32D in isa:
            self.legal_opcode.extend((0b0000111, 0b0100111, 0b1000011, 0b1000111,
                                      0b1001011, 0b1001111, 0b1010011))
        if riscv_instr_group_t.RV64I in isa:
            self.legal_opcode.append(0b0011011)
        if riscv_instr_group_t.RV32A in isa:
            self.legal_opcode.append(0b0101111)
        if riscv_instr_group_t.RV64I in isa or riscv_instr_group_t.RV64M in isa:
            self.legal_opcode.append(0b0111011)
        if riscv_instr_group_t.RV64I in isa:
            self.legal_c00_opcode.extend((0b011, 0b111))
            self.legal_c10_opcode.extend((0b011, 0b111))

    @vsc.constraint
    def exception_dist_c(self):
        vsc.dist(self.exception,
                 [vsc.weight(illegal_instr_type_e.kIllegalOpcode, 3),
                  vsc.weight(illegal_instr_type_e.kIllegalCompressedOpcode, 1),
                  vsc.weight(illegal_instr_type_e.kIllegalFunc3, 1),
                  vsc.weight(illegal_instr_type_e.kIllegalFunc7, 1),
                  vsc.weight(illegal_instr_type_e.kReservedCompressedInstr, 1),
                  vsc.weight(illegal_instr_type_e.kHintInstr, 3),
                  vsc.weight(illegal_instr_type_e.kIllegalSystemInstr, 3)
                  ])
        if not supports_compressed():
            self.exception != illegal_instr_type_e.kHintInstr
            self.compressed == 0

    @vsc.constraint
    def instr_bit_assignment_c(self):
        vsc.solve_order(self.opcode, self.instr_bin)
        vsc.solve_order(self.func3, self.instr_bin)
        vsc.solve_order(self.func7, self.instr_bin)
        vsc.solve_order(self.c_msb, self.instr_bin)
        vsc.solve_order(self.c_op, self.instr_bin)
        with vsc.if_then(self.compressed == 1):
            self.instr_bin[1:0] == self.c_op
            self.instr_bin[15:13] == self.c_msb
        with vsc.else_then():
            self.instr_bin[6:0] == self.opcode
            with vsc.if_then(self.has_func7 == 1):
                self.instr_bin[31:25] == self.func7
            with vsc.if_then(self.has_func3 == 1):
                self.instr_bin[14:12] == self.func3

    # Invalid SYSTEM instructions
    @vsc.constraint
    def system_instr_c(self):
        with vsc.if_then(self.exception == illegal_instr_type_e.kIllegalSystemInstr):
            self.opcode == 0b1110011
            # ECALL/EBREAK/xRET/WFI
            with vsc.if_then(self.func3 == 0):
                # Constrain RS1 and RD to be non-zero
                self.instr_bin[19:15] != 0
                self.instr_bin[11:7] != 0
                # Valid SYSTEM instructions considered by this
                # Constrain the upper 12 bits to be invalid
                self.instr_bin[31:20].not_inside(
                    vsc.rangelist(0x0, 0x1, 0x2, 0x102, 0x302, 0x7b2, 0x105))
            with vsc.else_then():
                # Invalid CSR instructions
                self.instr_bin[31:20].not_inside(vsc.rangelist(self.csrs))

    @vsc.constraint
    def legal_rv32_c_slli(self):
        if rcs.XLEN == 32:
            with vsc.if_then((self.c_msb == 0b000) & (self.c_op == 0b10)):
                with vsc.if_then(self.exception == illegal_instr_type_e.kReservedCompressedInstr):
                    self.instr_bin[12] == 1
                with vsc.else_then():
                    self.instr_bin[12] == 0

    @vsc.constraint
    def exception_type_c(self):
        with vsc.if_then(self.compressed == 1):
            self.exception.inside(vsc.rangelist(illegal_instr_type_e.kReservedCompressedInstr,
                                                illegal_instr_type_e.kIllegalCompressedOpcode,
                                                illegal_instr_type_e.kHintInstr))
        with vsc.else_then():
            self.exception.inside(vsc.rangelist(illegal_instr_type_e.kIllegalOpcode,
                                                illegal_instr_type_e.kIllegalFunc3,
                                                illegal_instr_type_e.kIllegalFunc7,
                                                illegal_instr_type_e.kIllegalSystemInstr))
        with vsc.if_then(self.has_func7 == 0):
            self.exception != illegal_instr_type_e.kIllegalFunc7
        with vsc.if_then(self.has_func3 == 0):
            self.exception != illegal_instr_type_e.kIllegalFunc3

    @vsc.constraint
    def compressed_instr_op_c(self):
        self.c_op != 0b11

    # Avoid generating illegal func3/func7 errors for opcodes used by B-extension for now
    # TODO: add support for generating illegal B-extension instructions
    @vsc.constraint
    def b_extension_c(self):
        if riscv_instr_group_t.RV32B in rcs.supported_isa:
            with vsc.if_then(self.exception.inside(
                    vsc.rangelist(illegal_instr_type_e.kIllegalFunc3,
                                  illegal_instr_type_e.kIllegalFunc7))):
                self.opcode.not_inside(vsc.rangelist(0b0110011, 0b0010011, 0b0111011))

    @vsc.constraint
    def illegal_compressed_op_c(self):
        with vsc.if_then(self.exception == illegal_instr_type_e.kIllegalCompressedOpcode):
            self.c_op != 0b01
            if len(self.legal_c00_opcode) == 8:
                self.c_op != 0b00
            else:
                self.c_msb.not_inside(vsc.rangelist(self.legal_c00_opcode))
            if len(self.legal_c10_opcode) == 8:
                self.c_op != 0b10
            else:
                self.c_msb.not_inside(vsc.rangelist(self.legal_c10_opcode))

    @vsc.constraint
    def reserved_compressed_instr_c(self):
        vsc.solve_order(self.exception, self.reserved_c)
        vsc.solve_order(self.exception, self.opcode)
        vsc.solve_order(self.reserved_c, self.instr_bin)
        vsc.solve_order(self.reserved_c, self.c_msb)
        vsc.solve_order(self.reserved_c, self.c_op)
        if rcs.XLEN == 32:
            # c.addiw is RV64/RV128 only instruction, the encoding is used for C.JAL for RV32C
            self.reserved_c != reserved_c_instr_e.kReservedAddiw
        with vsc.if_then(self.exception == illegal_instr_type_e.kReservedCompressedInstr):
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kIllegalCompressed):
                self.instr_bin[15:0] == 0
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedAddispn):
                (self.instr_bin[15:5] == 0) & (self.c_op == 0b00)
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedAddiw):
                (self.c_msb == 0b001) & (self.c_op == 0b01) & (self.instr_bin[11:7] == 0)
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedC0):
                ((self.instr_bin[15:10] == 0b100111) & (self.instr_bin[6:5] == 0b10) &
                 (self.c_op == 0b01))
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedC1):
                ((self.instr_bin[15:10] == 0b100111) & (self.instr_bin[6:5] == 0b11) &
                 (self.c_op == 0b01))
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedC2):
                (self.c_msb == 0b100) & (self.c_op == 0b00)
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedAddi16sp):
                ((self.c_msb == 0b011) & (self.c_op == 0b01) & (self.instr_bin[11:7] == 2) &
                 (self.instr_bin[12] == 0) & (self.instr_bin[6:2] == 0))
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedLui):
                ((self.c_msb == 0b011) & (self.c_op == 0b01) & (self.instr_bin[12] == 0) &
                 (self.instr_bin[6:2] == 0))
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedJr):
                self.instr_bin[15:0] == 0b1000_0000_0000_0010
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedLqsp):
                (self.c_msb == 0b001) & (self.c_op == 0b10) & (self.instr_bin[11:7] == 0)
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedLwsp):
                (self.c_msb == 0b010) & (self.c_op == 0b10) & (self.instr_bin[11:7] == 0)
            with vsc.implies(self.reserved_c == reserved_c_instr_e.kReservedLdsp):
                (self.c_msb == 0b011) & (self.c_op == 0b10) & (self.instr_bin[11:7] == 0)

    @vsc.constraint
    def hint_instr_c(self):
        with vsc.if_then(self.exception == illegal_instr_type_e.kHintInstr):
            (((self.c_msb == 0b000) & (self.c_op == 0b01) & (self.instr_bin[12] == 0) &
              (self.instr_bin[6:2] == 0)) |
             ((self.c_msb == 0b010) & (self.c_op == 0b01) & (self.instr_bin[11:7] == 0)) |
             ((self.c_msb == 0b100) & (self.c_op == 0b01) & (self.instr_bin[12:11] == 0) &
              (self.instr_bin[6:2] == 0)) |
             ((self.c_msb == 0b100) & (self.c_op == 0b10) & (self.instr_bin[11:7] == 0) &
              (self.instr_bin[6:2] != 0)) |
             ((self.c_msb == 0b011) & (self.c_op == 0b01) & (self.instr_bin[11:7] == 0) &
              ((self.instr_bin[12] != 0) | (self.instr_bin[6:2] != 0))) |
             ((self.c_msb == 0b000) & (self.c_op == 0b10) & (self.instr_bin[11:7] == 0)) |
             ((self.c_msb == 0b000) & (self.c_op == 0b10) & (self.instr_bin[11:7] != 0) &
              (self.instr_bin[12] == 0) & (self.instr_bin[6:2] == 0)) |
             ((self.c_msb == 0b100) & (self.c_op == 0b10) & (self.instr_bin[11:7] == 0) &
              (self.instr_bin[12] == 1) & (self.instr_bin[6:2] != 0)))

    @vsc.constraint
    def illegal_opcode_c(self):
        vsc.solve_order(self.opcode, self.instr_bin)
        with vsc.if_then(self.exception == illegal_instr_type_e.kIllegalOpcode):
            self.opcode.not_inside(vsc.rangelist(self.legal_opcode))
            self.opcode[1:0] == 0b11
        with vsc.else_then():
            self.opcode.inside(vsc.rangelist(self.legal_opcode))

    # TODO: Enable atomic instruction
    @vsc.constraint
    def no_atomic_c(self):
        with vsc.if_then(self.exception != illegal_instr_type_e.kIllegalOpcode):
            self.opcode != 0b0101111

    @vsc.constraint
    def illegal_func3_c(self):
        vsc.solve_order(self.opcode, self.func3)
        with vsc.if_then(self.compressed == 0):
            with vsc.if_then(self.exception == illegal_instr_type_e.kIllegalFunc3):
                with vsc.implies(self.opcode == 0b1100111):
                    self.func3 != 0b000
                with vsc.implies(self.opcode == 0b1100011):
                    self.func3.inside(vsc.rangelist(0b010, 0b011))
                if rcs.XLEN == 32:
                    with vsc.implies(self.opcode == 0b0100011):
                        self.func3 >= 0b011
                    with vsc.implies(self.opcode == 0b0000011):
                        self.func3.inside(vsc.rangelist(0b011, 0b111))
                else:
                    with vsc.implies(self.opcode == 0b0100011):
                        self.func3 > 0b011
                    with vsc.implies(self.opcode == 0b0000011):
                        self.func3 == 0b111
                with vsc.implies(self.opcode == 0b0001111):
                    self.func3.not_inside(vsc.rangelist(0b000, 0b001))
                with vsc.implies(self.opcode == 0b1110011):
                    self.func3 == 0b100
                with vsc.implies(self.opcode == 0b0011011):
                    self.func3.not_inside(vsc.rangelist(0b000, 0b001, 0b101))
                with vsc.implies(self.opcode == 0b0111011):
                    self.func3.inside(vsc.rangelist(0b010, 0b011))
                self.opcode.inside(vsc.rangelist(0b1100111, 0b1100011, 0b0000011, 0b0100011,
                                                 0b0001111, 0b1110011, 0b0011011, 0b0111011))
            with vsc.else_then():
                with vsc.implies(self.opcode == 0b1100111):
                    self.func3 == 0b000
                with vsc.implies(self.opcode == 0b1100011):
                    self.func3.not_inside(vsc.rangelist(0b010, 0b011))
                if rcs.XLEN == 32:
                    with vsc.implies(self.opcode == 0b0100011):
                        self.func3 < 0b011
                    with vsc.implies(self.opcode == 0b0000011):
                        self.func3.not_inside(vsc.rangelist(0b011, 0b111))
                else:
                    with vsc.implies(self.opcode == 0b0100011):
                        self.func3 <= 0b011
                    with vsc.implies(self.opcode == 0b0000011):
                        self.func3 != 0b111
                with vsc.implies(self.opcode == 0b0001111):
                    self.func3.inside(vsc.rangelist(0b000, 0b001))
                with vsc.implies(self.opcode == 0b1110011):
                    self.func3 != 0b100
                with vsc.implies(self.opcode == 0b0011011):
                    self.func3.inside(vsc.rangelist(0b000, 0b001, 0b101))
                with vsc.implies(self.opcode == 0b0111011):
                    self.func3.not_inside(vsc.rangelist(0b010, 0b011))

    @vsc.constraint
    def has_func7_c(self):
        vsc.solve_order(self.opcode, self.func7)
        with vsc.if_then(((self.opcode == 0b0010011) &
                          ((self.func3 == 0b001) | (self.func3 == 0b101))) |
                         (self.opcode == 0b0110011) | (self.opcode == 0b0111011)):
            self.has_func7 == 1
        with vsc.else_then():
            self.has_func7 == 0

    @vsc.constraint
    def has_func3_c(self):
        vsc.solve_order(self.opcode, self.func7)
        with vsc.if_then(self.opcode.inside(vsc.rangelist(0b0110111, 0b1101111, 0b0010111))):
            self.has_func3 == 0
        with vsc.else_then():
            self.has_func3 == 1

    @vsc.constraint
    def illegal_func7_c(self):
        with vsc.if_then(self.compressed == 0):
            with vsc.if_then(self.exception == illegal_instr_type_e.kIllegalFunc7):
                self.func7.not_inside(vsc.rangelist(0b0000000, 0b0100000, 0b0000001))
                # SLLI, SRLI, SRAI (the SV source compares with 7'b0001001, a typo)
                with vsc.if_then(self.opcode == 0b0010011):
                    self.func7[6:1].not_inside(vsc.rangelist(0b000000, 0b010000))
            with vsc.else_then():
                self.func7.inside(vsc.rangelist(0b0000000, 0b0100000, 0b0000001))

    # Draws a non-HINT type with the weights of exception_dist_c, leaving out the compressed
    # types when the target has no C extension
    def pick_exception(self):
        weights = {illegal_instr_type_e.kIllegalOpcode: 3,
                   illegal_instr_type_e.kIllegalFunc3: 1,
                   illegal_instr_type_e.kIllegalFunc7: 1,
                   illegal_instr_type_e.kIllegalSystemInstr: 3}
        if supports_compressed():
            weights[illegal_instr_type_e.kIllegalCompressedOpcode] = 1
            weights[illegal_instr_type_e.kReservedCompressedInstr] = 1
        return random.choices(list(weights), weights=list(weights.values()))[0]

    # The lists are built in __init__; kept for the callers of the SV init()
    def initialize(self):
        pass

    def get_bin_str(self):
        if self.compressed == 1:
            bin_str = "0x{:04x}".format(self.instr_bin & 0xffff)
        else:
            bin_str = "0x{:08x}".format(self.instr_bin)
        logging.info("Illegal instruction type: {}, illegal instruction: {}".format(
            self.exception.name, bin_str))
        return bin_str

    def post_randomize(self):
        self.comment = self.exception.name
        if self.exception == illegal_instr_type_e.kReservedCompressedInstr:
            self.comment += " {}".format(self.reserved_c.name)
        elif self.exception == illegal_instr_type_e.kIllegalOpcode:
            self.comment += " {:07b}".format(self.opcode)
