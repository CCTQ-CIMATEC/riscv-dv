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

from pygen_src.riscv_defines import DEFINE_VA_INSTR
from pygen_src.riscv_instr_pkg import (riscv_instr_name_t, riscv_instr_format_t,
                                       riscv_instr_category_t, riscv_instr_group_t,
                                       va_variant_t)

# RVV 1.0 integer instructions of the Zve32x subset, named after the group by create_instr

# RVV 1.0 section 11.1: vector single-width integer add and subtract
DEFINE_VA_INSTR(riscv_instr_name_t.VADD, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VSUB, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VRSUB, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VX, va_variant_t.VI], g=globals())

# RVV 1.0 section 11.5: vector bitwise logical instructions
DEFINE_VA_INSTR(riscv_instr_name_t.VAND, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.LOGICAL, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VOR, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.LOGICAL, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VXOR, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.LOGICAL, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())

# RVV 1.0 section 11.6: vector single-width shift instructions
DEFINE_VA_INSTR(riscv_instr_name_t.VSLL, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.SHIFT, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VSRL, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.SHIFT, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VSRA, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.SHIFT, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())

# RVV 1.0 section 11.8: vector integer compare instructions, the .vi forms of vmslt(u)
# and the .vv forms of vmsgt(u) do not exist
DEFINE_VA_INSTR(riscv_instr_name_t.VMSEQ, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMSNE, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMSLTU, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMSLT, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMSLEU, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMSLE, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMSGTU, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VX, va_variant_t.VI], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMSGT, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.COMPARE, riscv_instr_group_t.RVV,
                [va_variant_t.VX, va_variant_t.VI], g=globals())

# RVV 1.0 section 11.9: vector integer min/max instructions
DEFINE_VA_INSTR(riscv_instr_name_t.VMINU, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMIN, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMAXU, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMAX, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX], g=globals())

# RVV 1.0 sections 11.15 and 11.16: vector integer merge and move instructions
DEFINE_VA_INSTR(riscv_instr_name_t.VMERGE, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VVM, va_variant_t.VXM, va_variant_t.VIM], g=globals())
DEFINE_VA_INSTR(riscv_instr_name_t.VMV, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())

# RVV 1.0 section 14.1: vector single-width integer reduction instructions
DEFINE_VA_INSTR(riscv_instr_name_t.VREDSUM_VS, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV, g=globals())

# RVV 1.0 section 7.4: vector unit-stride loads and stores
DEFINE_VA_INSTR(riscv_instr_name_t.VLE_V, riscv_instr_format_t.VL_FORMAT,
                riscv_instr_category_t.LOAD, riscv_instr_group_t.RVV, g=globals())
