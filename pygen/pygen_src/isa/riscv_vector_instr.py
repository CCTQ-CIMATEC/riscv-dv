"""
Copyright 2020 Google LLC
Copyright 2020 Andes Technology Co., Ltd.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""

import logging
import random
import sys
import vsc
from pygen_src.isa.riscv_instr import riscv_instr
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.riscv_instr_pkg import (pkg_ins, riscv_instr_format_t, riscv_instr_category_t,
                                       va_variant_t, imm_t)


# Base class for the RISC-V vector extension, following RVV 1.0
# Port of the SV riscv_vector_instr (spec v0.8), restricted to integer instructions
# Vector registers are 5-bit fields printed as v<n>, so the group constraints can use
# arithmetic on them
@vsc.randobj
class riscv_vector_instr(riscv_instr):
    def __init__(self):
        super().__init__()
        self.vs1 = vsc.rand_bit_t(5)
        self.vs2 = vsc.rand_bit_t(5)
        self.vs3 = vsc.rand_bit_t(5)
        self.vd = vsc.rand_bit_t(5)
        # Drawn in pre_randomize: the solver strongly favours the first va_variant_t value
        self.va_variant = vsc.enum_t(va_variant_t)
        self.vm = vsc.rand_bit_t(1)
        # Variants from the instruction definition
        self.va_variants = []
        self.has_va_variant = vsc.bit_t(1)
        # Register group size of each operand, set by update_operand_groups
        self.vd_grp = vsc.uint8_t(1)
        self.vs1_grp = vsc.uint8_t(1)
        self.vs2_grp = vsc.uint8_t(1)
        self.vs3_grp = vsc.uint8_t(1)
        # vm = 0 is allowed
        self.mask_enable = vsc.bit_t(1, 1)
        # The destination may overlap v0 when masked: mask and scalar-like results
        self.vd_may_overlap_v0 = vsc.bit_t(1)
        # Element width of loads and stores, encoded in the mnemonic
        self.eew = 8
        self.sub_extension = ""
        self.is_reduction_instr = 0

    # RVV 1.0 section 3.4.2: a register group starts at a register aligned to its size
    @vsc.constraint
    def operand_group_c(self):
        self.vd % self.vd_grp == 0
        self.vs1 % self.vs1_grp == 0
        self.vs2 % self.vs2_grp == 0
        self.vs3 % self.vs3_grp == 0

    # RVV 1.0 section 5.3: a masked destination cannot overlap v0, except for mask results and the
    # scalar destination of reductions
    @vsc.constraint
    def vmask_c(self):
        with vsc.if_then(self.mask_enable == 0):
            self.vm == 1
        with vsc.if_then(self.vm == 0):
            with vsc.if_then(self.vd_may_overlap_v0 == 0):
                self.vd != 0

    def set_rand_mode(self):
        self.has_rs1 = 0
        self.has_rs2 = 0
        self.has_rd = 0
        self.has_imm = 0
        # vred*.vs and vwred*.vs
        if self.instr_name.name.endswith("_VS"):
            self.is_reduction_instr = 1
            self.vd_may_overlap_v0 = 1
        if len(self.va_variants) > 0:
            self.has_va_variant = 1
        if self.format == riscv_instr_format_t.VA_FORMAT:
            # Superset of all variants: rs1 for .vx and imm for .vi
            self.has_rs1 = 1
            self.has_imm = int(va_variant_t.VI in self.va_variants)
        if self.category in [riscv_instr_category_t.LOAD, riscv_instr_category_t.STORE]:
            # Base address register
            self.has_rs1 = 1
            self.mask_enable = cfg.vector_cfg.mask_load_store
            self.eew = cfg.vector_cfg.legal_eew[0]
        self.update_operand_groups()

    def set_imm_len(self):
        # simm5, or uimm5 for shifts
        self.imm_len = 5
        if self.category == riscv_instr_category_t.SHIFT:
            self.imm_type = imm_t.UIMM
        self.imm_mask = (self.imm_mask << self.imm_len) & self.shift_t

    # Register group sizes from the vtype of the test and, for loads and stores, the EEW
    def update_operand_groups(self):
        lmul = cfg.vector_cfg.vlmul
        vd_grp = vs1_grp = vs2_grp = vs3_grp = lmul
        if self.is_reduction_instr:
            # RVV 1.0 section 14: vd and vs1 hold a single element
            vd_grp = vs1_grp = 1
        if self.category in [riscv_instr_category_t.LOAD, riscv_instr_category_t.STORE]:
            # RVV 1.0 section 7.3: EMUL = (EEW / SEW) * LMUL, a fractional EMUL takes one
            # register
            emul = max(1, self.eew * lmul // cfg.vector_cfg.vsew)
            vd_grp = vs3_grp = emul
        self.vd_grp = vd_grp
        self.vs1_grp = vs1_grp
        self.vs2_grp = vs2_grp
        self.vs3_grp = vs3_grp

    def set_eew(self, eew):
        if eew not in cfg.vector_cfg.legal_eew:
            logging.critical("EEW %0d is not legal for e%0d m%0d", eew,
                             cfg.vector_cfg.vsew, cfg.vector_cfg.vlmul)
            sys.exit(1)
        self.eew = eew
        self.update_operand_groups()

    def pre_randomize(self):
        super().pre_randomize()
        if self.has_va_variant:
            self.va_variant = random.choice(self.va_variants)

    def get_instr_name(self):
        name = super().get_instr_name()
        if self.category in [riscv_instr_category_t.LOAD, riscv_instr_category_t.STORE]:
            # Add eew before the ".V" suffix: VLE.V -> VLE32.V
            name = "{}{}.V".format(name[:-2], self.eew)
        return name

    # Convert the instruction to assembly code
    def convert2asm(self, prefix = ""):
        vd = "v{}".format(self.vd)
        vs1 = "v{}".format(self.vs1)
        vs2 = "v{}".format(self.vs2)
        vs3 = "v{}".format(self.vs3)
        if self.format == riscv_instr_format_t.VA_FORMAT:
            if self.has_va_variant:
                asm_str = pkg_ins.format_string("{}.{}".format(self.get_instr_name(),
                                                               self.va_variant.name),
                                                pkg_ins.MAX_INSTR_STR_LEN)
                if self.va_variant == va_variant_t.VV:
                    operands = [vd, vs2, vs1]
                elif self.va_variant == va_variant_t.VX:
                    operands = [vd, vs2, self.rs1.name]
                elif self.va_variant == va_variant_t.VI:
                    operands = [vd, vs2, self.get_imm()]
                else:
                    logging.critical("Unsupported va_variant %0s of %0s",
                                     self.va_variant.name, self.instr_name.name)
                    sys.exit(1)
            else:
                asm_str = pkg_ins.format_string(self.get_instr_name(),
                                                pkg_ins.MAX_INSTR_STR_LEN)
                operands = [vd, vs2, vs1]
        elif self.format == riscv_instr_format_t.VL_FORMAT:
            asm_str = pkg_ins.format_string(self.get_instr_name(), pkg_ins.MAX_INSTR_STR_LEN)
            operands = [vd, "({})".format(self.rs1.name)]
        elif self.format == riscv_instr_format_t.VS_FORMAT:
            asm_str = pkg_ins.format_string(self.get_instr_name(), pkg_ins.MAX_INSTR_STR_LEN)
            operands = [vs3, "({})".format(self.rs1.name)]
        else:
            logging.critical("Unsupported vector format %0s of %0s",
                             self.format.name, self.instr_name.name)
            sys.exit(1)
        if not self.vm:
            operands.append("v0.t")
        asm_str = "{} {}".format(asm_str, ", ".join(operands))
        if self.comment != "":
            asm_str = asm_str + " #" + self.comment
        return asm_str.lower()
