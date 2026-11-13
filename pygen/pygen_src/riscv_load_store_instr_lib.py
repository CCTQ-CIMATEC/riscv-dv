
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

import sys
import random
import logging
import vsc
from enum import IntEnum, auto
from importlib import import_module
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.isa.riscv_instr import riscv_instr
from pygen_src.riscv_directed_instr_lib import riscv_mem_access_stream
from pygen_src.riscv_instr_pkg import (riscv_reg_t, riscv_instr_name_t, riscv_instr_group_t,
                                       riscv_instr_category_t)
rcs = import_module("pygen_src.target." + cfg.argv.target + ".riscv_core_setting")


class locality_e(IntEnum):
    NARROW = 0
    HIGH = auto()
    MEDIUM = auto()
    SPARSE = auto()


# Base class for all load/store instruction stream
@vsc.randobj
class riscv_load_store_base_instr_stream(riscv_mem_access_stream):
    def __init__(self):
        super().__init__()
        self.num_load_store = vsc.rand_uint32_t()
        self.num_mixed_instr = vsc.rand_uint32_t()
        self.base = vsc.rand_int32_t()
        self.offset = []
        self.addr = []
        self.load_store_instr = []
        self.data_page_id = vsc.rand_uint32_t()
        self.rs1_reg = vsc.rand_enum_t(riscv_reg_t)
        self.locality = vsc.rand_enum_t(locality_e)
        self.max_load_store_offset = vsc.rand_int32_t()
        self.use_sp_as_rs1 = vsc.rand_bit_t()

    @vsc.constraint
    def sp_rnd_order_c(self):
        vsc.solve_order(self.use_sp_as_rs1, self.rs1_reg)

    @vsc.constraint
    def sp_c(self):
        vsc.dist(self.use_sp_as_rs1, [vsc.weight(1, 1), vsc.weight(0, 2)])
        with vsc.if_then(self.use_sp_as_rs1 == 1):
            self.rs1_reg == riscv_reg_t.SP

    # TODO Getting pyvsc error -- > rs1 has not been build yet
    '''@vsc.constraint
    def rs1_c(self):
        self.rs1_reg.not_inside(vsc.rangelist(cfg.reserved_regs,
                                              self.reserved_rd, riscv_reg_t.ZERO))'''

    @vsc.constraint
    def addr_c(self):
        # TODO solve_order
        # vsc.solve_order(self.data_page_id, self.max_load_store_offset)
        # vsc.solve_order(self.max_load_store_offset, self.base)
        self.data_page_id < self.max_data_page_id
        with vsc.foreach(self.data_page, idx = True) as i:
            with vsc.if_then(i == self.data_page_id):
                self.max_load_store_offset == self.data_page[i].size_in_bytes
        self.base in vsc.rangelist(vsc.rng(0, self.max_load_store_offset - 1))

    def randomize_offset(self):
        self.offset = [0] * self.num_load_store
        self.addr = [0] * self.num_load_store
        for i in range(self.num_load_store):
            self.offset[i], self.addr[i] = self.rand_offset_addr()

    # One access, as the SV randomize_offset: addr == base + offset, inside the
    # data page; the locality range (soft there) is clipped to the page. It
    # always holds offset 0, since base is inside the page.
    def rand_offset_addr(self):
        lo, hi = {locality_e.NARROW: (-16, 16),
                  locality_e.HIGH: (-64, 64),
                  locality_e.MEDIUM: (-256, 256),
                  locality_e.SPARSE: (-2048, 2047)}[self.locality]
        offset_ = random.randint(max(lo, -self.base),
                                 min(hi, self.max_load_store_offset - 1 - self.base))
        return offset_, self.base + offset_

    def pre_randomize(self):
        super().pre_randomize()
        if(riscv_reg_t.SP in [cfg.reserved_regs, self.reserved_rd]):
            self.use_sp_as_rs1 = 0
            with vsc.raw_mode():
                self.use_sp_as_rs1.rand_mode = False
            self.sp_rnd_order_c.constraint_mode(False)

    def post_randomize(self):
        # rs1_c is not a pyvsc constraint (see the TODO above)
        excluded = [riscv_reg_t.ZERO, *cfg.reserved_regs, *self.reserved_rd]
        if self.rs1_reg in excluded:
            self.rs1_reg = random.choice([reg for reg in riscv_reg_t if reg not in excluded])
        self.randomize_offset()
        # rs1 cannot be modified by other instructions
        if not(self.rs1_reg in self.reserved_rd):
            self.reserved_rd.append(self.rs1_reg)
        self.gen_load_store_instr()
        self.add_mixed_instr(self.num_mixed_instr)
        self.add_rs1_init_la_instr(self.rs1_reg, self.data_page_id, self.base)
        super().post_randomize()

    # Generate each load/store instruction
    def gen_load_store_instr(self):
        allowed_instr = []
        enable_compressed_load_store = 0
        self.randomize_avail_regs()
        if ((self.rs1_reg in [riscv_reg_t.S0, riscv_reg_t.S1, riscv_reg_t.A0, riscv_reg_t.A1,
                              riscv_reg_t.A2, riscv_reg_t.A3, riscv_reg_t.A4, riscv_reg_t.A5,
                              riscv_reg_t.SP]) and not(cfg.disable_compressed_instr)):
            enable_compressed_load_store = 1
        for i in range(len(self.addr)):
            # Assign the allowed load/store instructions based on address alignment
            # This is done separately rather than a constraint to improve the randomization
            # performance. The list restarts for each access, as in the SV
            # generator: one that grows would let an aligned access's LH/LW
            # through for a later misaligned address.
            allowed_instr = [riscv_instr_name_t.LB, riscv_instr_name_t.LBU, riscv_instr_name_t.SB]
            if not cfg.enable_unaligned_load_store:
                if (self.addr[i] & 1) == 0:
                    allowed_instr.extend(
                        [riscv_instr_name_t.LH, riscv_instr_name_t.LHU, riscv_instr_name_t.SH])
                if self.addr[i] % 4 == 0:
                    allowed_instr.extend([riscv_instr_name_t.LW, riscv_instr_name_t.SW])
                    if cfg.enable_floating_point:
                        allowed_instr.extend([riscv_instr_name_t.FLW,
                                              riscv_instr_name_t.FSW])
                    if ((self.offset[i] in range(128)) and (self.offset[i] % 4 == 0) and
                        (riscv_instr_group_t.RV32C in rcs.supported_isa) and
                            (enable_compressed_load_store)):
                        if self.rs1_reg == riscv_reg_t.SP:
                            logging.info("Add LWSP/SWSP to allowed instr")
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LWSP, riscv_instr_name_t.C_SWSP])
                        else:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LW, riscv_instr_name_t.C_SW])
                            if (cfg.enable_floating_point and
                                    riscv_instr_group_t.RV32FC in rcs.supported_isa):
                                allowed_instr.extend(
                                    [riscv_instr_name_t.C_FLW, riscv_instr_name_t.C_FSW])
                if (rcs.XLEN >= 64) and (self.addr[i] % 8 == 0):
                    allowed_instr.extend([riscv_instr_name_t.LWU,
                                          riscv_instr_name_t.LD,
                                          riscv_instr_name_t.SD])
                    if (cfg.enable_floating_point and
                            (riscv_instr_group_t.RV32D in rcs.supported_isa)):
                        allowed_instr.extend([riscv_instr_name_t.FLD,
                                              riscv_instr_name_t.FSD])
                    if (self.offset[i] in range(256) and (self.offset[i] % 8 == 0) and
                        (riscv_instr_group_t.RV64C in rcs.supported_isa) and
                            enable_compressed_load_store):
                        if self.rs1_reg == riscv_reg_t.SP:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LDSP, riscv_instr_name_t.C_SDSP])
                        else:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LD, riscv_instr_name_t.C_SD])
                            if (cfg.enable_floating_point and
                                    (riscv_instr_group_t.RV32DC in rcs.supported_isa)):
                                allowed_instr.extend(
                                    [riscv_instr_name_t.C_FLD, riscv_instr_name_t.C_FSD])
            else:  # unaligned load/store (enable_unaligned_load_store)
                allowed_instr.extend([riscv_instr_name_t.LW, riscv_instr_name_t.SW,
                                      riscv_instr_name_t.LH, riscv_instr_name_t.LHU,
                                      riscv_instr_name_t.SH])
                # Compressed load/store still needs to be alligned
                if (self.offset[i] in range(128) and (self.offset[i] % 4 == 0) and
                    (riscv_instr_group_t.RV32C in rcs.supported_isa) and
                        enable_compressed_load_store):
                    if self.rs1_reg == riscv_reg_t.SP:
                        allowed_instr.extend(
                            [riscv_instr_name_t.C_LWSP, riscv_instr_name_t.C_SWSP])
                    else:
                        allowed_instr.extend(
                            [riscv_instr_name_t.C_LW, riscv_instr_name_t.C_SW])
                if rcs.XLEN >= 64:
                    allowed_instr.extend(
                        [riscv_instr_name_t.LWU, riscv_instr_name_t.LD, riscv_instr_name_t.SD])
                    if (self.offset[i] in range(256) and (self.offset[i] % 8 == 0) and
                        (riscv_instr_group_t.RV64C in rcs.supported_isa) and
                            enable_compressed_load_store):
                        if self.rs1_reg == riscv_reg_t.SP:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LWSP, riscv_instr_name_t.C_SWSP])
                        else:
                            allowed_instr.extend(
                                [riscv_instr_name_t.C_LD, riscv_instr_name_t.C_SD])
            instr = riscv_instr.get_load_store_instr(allowed_instr)
            instr.has_rs1 = 0
            instr.has_imm = 0
            self.randomize_gpr(instr)
            instr.rs1 = self.rs1_reg
            instr.imm_str = str(instr.uintToInt(self.offset[i]))
            instr.process_load_store = 0
            self.instr_list.append(instr)
            self.load_store_instr.append(instr)


# A single load/store instruction
@vsc.randobj
class riscv_single_load_store_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()

    @vsc.constraint
    def legal_c(self):
        self.num_load_store == 1
        self.num_mixed_instr < 5


# Back to back load/store instructions
@vsc.randobj
class riscv_load_store_stress_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()
        self.max_instr_cnt = 30
        self.min_instr_cnt = 10

    @vsc.constraint
    def legal_c(self):
        self.num_load_store.inside(vsc.rangelist(vsc.rng(self.min_instr_cnt, self.max_instr_cnt)))
        self.num_mixed_instr == 0


# Back to back load/store instructions
@vsc.randobj
class riscv_load_store_shared_mem_stream(riscv_load_store_stress_instr_stream):
    def __init__(self):
        super().__init__()

    def pre_randomize(self):
        self.load_store_shared_memory = 1
        super().pre_randomize()


# Random load/store sequence
# A random mix of load/store instructions and other instructions
@vsc.randobj
class riscv_load_store_rand_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()

    @vsc.constraint
    def legal_c(self):
        self.num_load_store.inside(vsc.rangelist(vsc.rng(10, 30)))
        self.num_mixed_instr.inside(vsc.rangelist(vsc.rng(10, 30)))


# Use a small set of GPR to create various WAW, RAW, WAR hazard scenario
@vsc.randobj
class riscv_load_store_hazard_instr_stream(riscv_load_store_base_instr_stream):
    def __init__(self):
        super().__init__()
        self.hazard_ratio = vsc.rand_int32_t()

    @vsc.constraint
    def hazard_ratio_c(self):
        self.hazard_ratio.inside(vsc.rangelist(vsc.rng(20, 100)))

    @vsc.constraint
    def legal_c(self):
        self.num_load_store.inside(vsc.rangelist(vsc.rng(10, 20)))
        self.num_mixed_instr.inside(vsc.rangelist(vsc.rng(1, 7)))

    def randomize_offset(self):
        self.offset = [0] * self.num_load_store
        self.addr = [0] * self.num_load_store
        rand_num = random.randrange(0, 100)
        for i in range(self.num_load_store):
            if (i > 0) and (rand_num < self.hazard_ratio):
                self.offset[i] = self.offset[i - 1]
                self.addr[i] = self.addr[i - 1]
            else:
                self.offset[i], self.addr[i] = self.rand_offset_addr()


# Vector loads and stores from one data page, port of the SV riscv_vector_load_store_instr_stream
# for unit-stride accesses
# The base address is a multiple of cfg.vector_cfg.load_store_alignment and the accessed
# bytes, vl * EEW / 8, stay inside the page
@vsc.randobj
class riscv_vector_load_store_instr_stream(riscv_mem_access_stream):
    def __init__(self):
        super().__init__()
        self.rs1_reg = riscv_reg_t.ZERO
        self.load_store_instr = None

    def post_randomize(self):
        vector_cfg = cfg.vector_cfg
        allowed_instr = [name for name, instr in riscv_instr.instr_template.items()
                         if instr.group == riscv_instr_group_t.RVV and
                         instr.category in [riscv_instr_category_t.LOAD,
                                            riscv_instr_category_t.STORE]]
        if not allowed_instr:
            logging.critical("No vector load/store instruction for %0s", self.name)
            sys.exit(1)
        self.load_store_instr = riscv_instr.get_instr(random.choice(allowed_instr))
        eew = random.choice(vector_cfg.legal_eew)
        self.load_store_instr.set_eew(eew)
        data_page_id = random.randrange(0, self.max_data_page_id)
        page_size = self.data_page[data_page_id].size_in_bytes
        num_bytes = vector_cfg.vl * eew // 8
        align = vector_cfg.load_store_alignment
        if num_bytes > page_size:
            logging.critical("Vector access of %0d bytes does not fit data page %0d",
                             num_bytes, data_page_id)
            sys.exit(1)
        base = random.randrange(0, (page_size - num_bytes) // align + 1) * align
        excluded = [riscv_reg_t.ZERO, *cfg.reserved_regs, *self.reserved_rd]
        self.rs1_reg = random.choice([reg for reg in riscv_reg_t if reg not in excluded])
        # Mixed instructions must keep the base register
        self.reserved_rd.append(self.rs1_reg)
        with self.load_store_instr.randomize_with() as it:
            it.rs1 == self.rs1_reg
        self.instr_list.append(self.load_store_instr)
        self.add_mixed_instr(random.randrange(0, 11))
        self.add_rs1_init_la_instr(self.rs1_reg, data_page_id, base)
        super().post_randomize()
