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
import vsc
from pygen_src.riscv_instr_pkg import vxrm_t


# Vector state used by every vector instruction of a test, set once by the init section
# Port of the SV riscv_vector_cfg for RVV 1.0: no EDIV, and LMUL is integral (1, 2, 4, 8)
@vsc.randobj
class riscv_vector_cfg:
    def __init__(self, rcs):
        self.VLEN = rcs.VLEN
        self.ELEN = vsc.uint32_t(rcs.ELEN)
        self.MAX_LMUL = vsc.uint32_t(rcs.MAX_LMUL)

        self.vsew = vsc.rand_uint32_t()
        self.vlmul = vsc.rand_uint32_t()
        self.vxrm = vsc.rand_enum_t(vxrm_t)
        self.vxsat = vsc.rand_bit_t(1)
        # Allow widening and narrowing instructions, which need 2*SEW <= ELEN and LMUL < 8
        self.vec_narrowing_widening = vsc.rand_bit_t(1)

        # Derived in post_randomize
        self.vl = 0
        self.legal_eew = []
        # Vector registers the random instructions must not write
        self.reserved_vregs = []

        # Masked vector loads and stores, 0 generates them all with vm = 1
        self.mask_load_store = getattr(rcs, "support_masked_vector_load_store", 1)
        # Base address alignment of vector loads and stores, in bytes
        self.load_store_alignment = getattr(rcs, "vector_load_store_alignment", 1)

    @vsc.constraint
    def vsew_c(self):
        self.vsew.inside(vsc.rangelist(8, 16, 32))
        self.vsew <= self.ELEN
        with vsc.if_then(self.vec_narrowing_widening == 1):
            self.vsew < self.ELEN

    @vsc.constraint
    def vlmul_c(self):
        self.vlmul.inside(vsc.rangelist(1, 2, 4, 8))
        self.vlmul <= self.MAX_LMUL
        with vsc.if_then(self.vec_narrowing_widening == 1):
            self.vlmul < 8

    def post_randomize(self):
        # Bringup setting of the SV class: vl = VLMAX, vstart = 0
        self.vl = self.VLEN * self.vlmul // self.vsew
        # RVV 1.0 section 7.3: EMUL = (EEW / SEW) * LMUL must stay within [1/8, 8]
        self.legal_eew = [eew for eew in (8, 16, 32)
                          if eew <= self.ELEN and
                          1 / 8 <= eew / self.vsew * self.vlmul <= 8]
        logging.info("vector_cfg: e%0d m%0d vl=%0d legal_eew=%s",
                     self.vsew, self.vlmul, self.vl, self.legal_eew)

    # vtype operands of vsetvli/vsetivli
    # Tail and mask undisturbed: with agnostic policies the ISS may fill those elements with
    # ones while the core keeps them, and later reads or stores of the group would diverge
    def vtype_str(self, sew = None, lmul = None):
        sew = self.vsew if sew is None else sew
        lmul = self.vlmul if lmul is None else lmul
        return "e{}, m{}, tu, mu".format(sew, lmul)
