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
                                       riscv_instr_name_t, va_variant_t, imm_t)


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
        # vm = 0 is part of the encoding: vmerge, with v0 as an operand instead of a mask
        self.mask_required = vsc.bit_t(1)
        # The destination may overlap v0 when masked: mask and scalar-like results
        self.vd_may_overlap_v0 = vsc.bit_t(1)
        # vd has a narrower EEW than vs2 or vs1: compare masks and narrowing results
        self.vd_narrower_vs2 = vsc.bit_t(1)
        self.vd_narrower_vs1 = vsc.bit_t(1)
        # vd has a wider EEW than vs2 or vs1: widening and integer extension results
        self.vd_wider_vs2 = vsc.bit_t(1)
        self.vd_wider_vs1 = vsc.bit_t(1)
        # The narrower source has a fractional EMUL and takes part of one register
        self.src_emul_fractional = vsc.bit_t(1)
        # vs1 is an operand of the drawn variant, set in pre_randomize
        self.has_vs1 = vsc.bit_t(1, 1)
        # Element width of loads and stores, encoded in the mnemonic
        self.eew = 8
        self.sub_extension = ""
        self.is_reduction_instr = 0
        self.is_widening_instr = 0
        self.is_narrowing_instr = 0
        # Source EEW divisor of vzext/vsext: 2 for .vf2, 4 for .vf4
        self.ext_factor = 1
        # Multiply-add operand order: vd, vs1/rs1, vs2
        self.is_multiply_add_instr = 0
        # The result goes to the x register rd: vcpop.m, vfirst.m and vmv.x.s
        self.has_scalar_rd = 0
        # Mask register logical instructions, vm*.mm
        self.is_mask_logical_instr = 0

    # RVV 1.0 section 3.4.2: a register group starts at a register aligned to its size
    @vsc.constraint
    def operand_group_c(self):
        self.vd % self.vd_grp == 0
        self.vs1 % self.vs1_grp == 0
        self.vs2 % self.vs2_grp == 0
        self.vs3 % self.vs3_grp == 0

    # RVV 1.0 section 5.2: a destination with a narrower EEW may overlap a source group only
    # in its lowest-numbered registers, so vd either starts the group or stays out of it
    @vsc.constraint
    def narrow_dest_overlap_c(self):
        with vsc.if_then(self.vd_narrower_vs2 == 1):
            ((self.vd == self.vs2) | (self.vd + self.vd_grp <= self.vs2) |
             (self.vd >= self.vs2 + self.vs2_grp))
        with vsc.if_then((self.vd_narrower_vs1 == 1) & (self.has_vs1 == 1)):
            ((self.vd == self.vs1) | (self.vd + self.vd_grp <= self.vs1) |
             (self.vd >= self.vs1 + self.vs1_grp))

    # RVV 1.0 section 5.2: a destination with a wider EEW may overlap a source group only in
    # its highest-numbered registers, and only when the source EMUL is at least 1
    @vsc.constraint
    def wide_dest_overlap_c(self):
        with vsc.if_then(self.vd_wider_vs2 == 1):
            ((self.vs2 + self.vs2_grp <= self.vd) | (self.vs2 >= self.vd + self.vd_grp) |
             ((self.src_emul_fractional == 0) &
              (self.vs2 + self.vs2_grp == self.vd + self.vd_grp)))
        with vsc.if_then((self.vd_wider_vs1 == 1) & (self.has_vs1 == 1)):
            ((self.vs1 + self.vs1_grp <= self.vd) | (self.vs1 >= self.vd + self.vd_grp) |
             (self.vs1 + self.vs1_grp == self.vd + self.vd_grp))

    # RVV 1.0 section 5.3: a masked destination cannot overlap v0, except for mask results and the
    # scalar destination of reductions
    @vsc.constraint
    def vmask_c(self):
        with vsc.if_then(self.mask_enable == 0):
            self.vm == 1
        with vsc.if_then(self.mask_required == 1):
            self.vm == 0
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
        name = self.instr_name.name
        if self.category == riscv_instr_category_t.COMPARE:
            self.vd_narrower_vs2 = 1
            self.vd_narrower_vs1 = 1
            self.vd_may_overlap_v0 = 1
        # vw* except the vwred*.vs reductions
        if name.startswith("VW") and not self.is_reduction_instr:
            self.is_widening_instr = 1
            self.vd_wider_vs1 = 1
        if name in ["VNSRL", "VNSRA", "VNCLIPU", "VNCLIP"]:
            self.is_narrowing_instr = 1
            self.vd_narrower_vs2 = 1
        if name.startswith("VZEXT") or name.startswith("VSEXT"):
            self.ext_factor = int(name[-1])
        if name in ["VMACC", "VNMSAC", "VMADD", "VNMSUB",
                    "VWMACCU", "VWMACC", "VWMACCSU", "VWMACCUS"]:
            self.is_multiply_add_instr = 1
        if name.endswith("_MM"):
            self.is_mask_logical_instr = 1
            self.mask_enable = 0
        if name in ["VCPOP_M", "VFIRST_M", "VMV_X_S"]:
            self.has_scalar_rd = 1
            self.has_rd = 1
        if name in ["VMV_X_S", "VMV_S_X"]:
            self.mask_enable = 0
        if len(self.va_variants) > 0:
            self.has_va_variant = 1
        if self.format == riscv_instr_format_t.VA_FORMAT and not self.has_scalar_rd:
            # Superset of all variants: rs1 for .vx and imm for .vi
            self.has_rs1 = 1
            self.has_imm = int(any(v in self.va_variants for v in
                                   [va_variant_t.VI, va_variant_t.VIM, va_variant_t.WI]))
        if self.instr_name == riscv_instr_name_t.VMV:
            self.mask_enable = 0
        if self.instr_name == riscv_instr_name_t.VMERGE:
            self.mask_required = 1
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

    # Filter the instructions the vtype of the test makes illegal
    def is_supported(self, cfg):
        vector_cfg = cfg.vector_cfg
        # 2*SEW results or sources need 2*SEW <= ELEN and 2*LMUL <= 8
        if ((self.is_widening_instr or self.is_narrowing_instr) and
                not vector_cfg.vec_narrowing_widening):
            return 0
        # RVV 1.0 section 11.3: the source EEW SEW/vf2 or SEW/vf4 is at least 8
        if vector_cfg.vsew // self.ext_factor < 8:
            return 0
        # RVV 1.0 section 14.2: vwredsum(u) accumulates 2*SEW elements in vd and vs1
        if (self.is_reduction_instr and self.instr_name.name.startswith("VW") and
                2 * vector_cfg.vsew > vector_cfg.ELEN):
            return 0
        return 1

    # Register group sizes from the vtype of the test, the drawn variant and, for loads and
    # stores, the EEW
    def update_operand_groups(self):
        lmul = cfg.vector_cfg.vlmul
        vd_grp = vs1_grp = vs2_grp = vs3_grp = lmul
        vd_wider_vs2 = 0
        if self.is_widening_instr:
            vd_grp = 2 * lmul
            # .wv and .wx take a 2*SEW vs2, with the EEW of vd
            if self.va_variant in [va_variant_t.WV, va_variant_t.WX]:
                vs2_grp = 2 * lmul
            else:
                vd_wider_vs2 = 1
        if self.is_narrowing_instr:
            vs2_grp = 2 * lmul
        if self.ext_factor > 1:
            vs2_grp = max(1, lmul // self.ext_factor)
            vd_wider_vs2 = 1
            self.src_emul_fractional = int(lmul < self.ext_factor)
        self.vd_wider_vs2 = vd_wider_vs2
        if self.is_reduction_instr:
            # RVV 1.0 section 14: vd and vs1 hold a single element
            vd_grp = vs1_grp = 1
        if self.category == riscv_instr_category_t.COMPARE:
            # One mask register holds the result of a whole group
            vd_grp = 1
        if self.is_mask_logical_instr or self.instr_name in [riscv_instr_name_t.VCPOP_M,
                                                             riscv_instr_name_t.VFIRST_M]:
            # Masks take one register whatever LMUL is
            vd_grp = vs1_grp = vs2_grp = 1
        if self.instr_name in [riscv_instr_name_t.VMV_X_S, riscv_instr_name_t.VMV_S_X]:
            # RVV 1.0 section 16.1: element 0 of a single register, LMUL is ignored
            vd_grp = vs2_grp = 1
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
            self.has_vs1 = int(self.va_variant in [va_variant_t.VV, va_variant_t.VVM,
                                                   va_variant_t.WV])
            self.update_operand_groups()

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
        if self.instr_name == riscv_instr_name_t.VMV_X_S:
            asm_str = pkg_ins.format_string(self.get_instr_name(), pkg_ins.MAX_INSTR_STR_LEN)
            operands = [self.rd.name, vs2]
        elif self.instr_name == riscv_instr_name_t.VMV_S_X:
            asm_str = pkg_ins.format_string(self.get_instr_name(), pkg_ins.MAX_INSTR_STR_LEN)
            operands = [vd, self.rs1.name]
        elif self.format == riscv_instr_format_t.VA_FORMAT:
            if self.has_va_variant:
                # Last source operand: a vector register, x register or immediate
                if self.va_variant in [va_variant_t.VV, va_variant_t.VVM, va_variant_t.WV]:
                    src = vs1
                elif self.va_variant in [va_variant_t.VX, va_variant_t.VXM, va_variant_t.WX]:
                    src = self.rs1.name
                elif self.va_variant in [va_variant_t.VI, va_variant_t.VIM, va_variant_t.WI]:
                    src = self.get_imm()
                else:
                    logging.critical("Unsupported va_variant %0s of %0s",
                                     self.va_variant.name, self.instr_name.name)
                    sys.exit(1)
                if self.instr_name == riscv_instr_name_t.VMV:
                    # vmv.v.v, vmv.v.x and vmv.v.i have no vs2
                    name = "{}.V.{}".format(self.get_instr_name(), self.va_variant.name[1])
                    operands = [vd, src]
                elif self.is_multiply_add_instr:
                    name = "{}.{}".format(self.get_instr_name(), self.va_variant.name)
                    operands = [vd, src, vs2]
                else:
                    name = "{}.{}".format(self.get_instr_name(), self.va_variant.name)
                    operands = [vd, vs2, src]
                asm_str = pkg_ins.format_string(name, pkg_ins.MAX_INSTR_STR_LEN)
            else:
                asm_str = pkg_ins.format_string(self.get_instr_name(),
                                                pkg_ins.MAX_INSTR_STR_LEN)
                operands = [vd, vs2, vs1]
        elif self.format == riscv_instr_format_t.VS2_FORMAT:
            asm_str = pkg_ins.format_string(self.get_instr_name(), pkg_ins.MAX_INSTR_STR_LEN)
            operands = [self.rd.name if self.has_scalar_rd else vd, vs2]
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
            operands.append("v0" if self.mask_required else "v0.t")
        asm_str = "{} {}".format(asm_str, ", ".join(operands))
        if self.comment != "":
            asm_str = asm_str + " #" + self.comment
        return asm_str.lower()
