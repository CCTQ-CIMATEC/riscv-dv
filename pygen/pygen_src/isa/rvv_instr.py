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

# Vector integer arithmetic instructions
DEFINE_VA_INSTR(riscv_instr_name_t.VADD, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV,
                [va_variant_t.VV, va_variant_t.VX, va_variant_t.VI], g=globals())

# Vector reduction instructions
DEFINE_VA_INSTR(riscv_instr_name_t.VREDSUM_VS, riscv_instr_format_t.VA_FORMAT,
                riscv_instr_category_t.ARITHMETIC, riscv_instr_group_t.RVV, g=globals())

# RVV 1.0 section 7.4: vector unit-stride loads and stores
DEFINE_VA_INSTR(riscv_instr_name_t.VLE_V, riscv_instr_format_t.VL_FORMAT,
                riscv_instr_category_t.LOAD, riscv_instr_group_t.RVV, g=globals())
