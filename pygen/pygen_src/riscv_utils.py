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
import os
import sys
import inspect
import logging
import importlib.util
import pandas as pd
from tabulate import tabulate
from pygen_src.riscv_instr_gen_config import cfg
from pygen_src.riscv_loop_instr import riscv_loop_instr
from pygen_src.riscv_directed_instr_lib import (riscv_directed_instr_stream,
                                                riscv_int_numeric_corner_stream,
                                                riscv_jal_instr, riscv_mem_access_stream,
                                                riscv_vector_vsetvli_instr_stream,
                                                riscv_vector_hazard_instr_stream,
                                                riscv_vector_scalar_return_stream)
from pygen_src.riscv_instr_stream import riscv_instr_stream
from pygen_src.riscv_amo_instr_lib import (riscv_lr_sc_instr_stream, riscv_amo_instr_stream)
from pygen_src.riscv_load_store_instr_lib import (riscv_load_store_rand_instr_stream,
                                                  riscv_load_store_hazard_instr_stream,
                                                  riscv_load_store_stress_instr_stream,
                                                  riscv_single_load_store_instr_stream,
                                                  riscv_vector_load_store_instr_stream)


# ----------------------------------------------------------
# pyflow commmon utility helpers functions
# ----------------------------------------------------------

# Instruction stream classes of the +stream_module files, by class name, loaded on the first
# factory call
user_streams = {}
user_streams_loaded = False


# Imports each file of +stream_module (comma-separated, environment variables expanded, e.g.
# $PROJECT/dv/streams/core_streams.py) and registers the instruction stream
# classes it defines, so directed_instr_N can name them like the built-in streams
def load_stream_modules():
    for path in [p.strip() for p in cfg.argv.stream_module.split(",") if p.strip()]:
        path = os.path.abspath(os.path.expandvars(path))
        if not os.path.isfile(path):
            logging.critical("Cannot find stream module %s", path)
            sys.exit(1)
        name = "user_stream_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        # Every stream class in the module namespace, imported ones included: vsc.randobj
        # moves __module__ to vsc.rand_obj, and an imported built-in maps to the same class
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, riscv_instr_stream):
                user_streams[cls_name] = cls
                logging.info("Registered stream %s from %s", cls_name, path)


# Instruction stream classes by name, built-in and from +stream_module
def stream_classes():
    global user_streams_loaded
    if not user_streams_loaded:
        load_stream_modules()
        user_streams_loaded = True
    objs = {
        "riscv_directed_instr_stream": riscv_directed_instr_stream,
        "riscv_int_numeric_corner_stream": riscv_int_numeric_corner_stream,
        "riscv_jal_instr": riscv_jal_instr,
        "riscv_mem_access_stream": riscv_mem_access_stream,
        "riscv_lr_sc_instr_stream": riscv_lr_sc_instr_stream,
        "riscv_amo_instr_stream": riscv_amo_instr_stream,
        "riscv_load_store_rand_instr_stream": riscv_load_store_rand_instr_stream,
        "riscv_load_store_hazard_instr_stream": riscv_load_store_hazard_instr_stream,
        "riscv_load_store_stress_instr_stream": riscv_load_store_stress_instr_stream,
        "riscv_single_load_store_instr_stream": riscv_single_load_store_instr_stream,
        "riscv_vector_load_store_instr_stream": riscv_vector_load_store_instr_stream,
        "riscv_vector_vsetvli_instr_stream": riscv_vector_vsetvli_instr_stream,
        "riscv_vector_hazard_instr_stream": riscv_vector_hazard_instr_stream,
        "riscv_vector_scalar_return_stream": riscv_vector_scalar_return_stream,
        "riscv_loop_instr": riscv_loop_instr
    }
    objs.update(user_streams)
    return objs


def factory(obj_of):
    objs = stream_classes()
    try:
        return objs[obj_of]()
    except KeyError:
        logging.critical("Cannot Create object of %s", obj_of)
        sys.exit(1)


def gen_config_table():
    data = []
    for key, value in cfg.__dict__.items():
        # Ignoring the unneccesary attributes
        if key in ["_ro_int", "_int_field_info", "argv", "mem_region",
                   "amo_region", "s_mem_region", "args_dict"]:
            continue
        else:
            try:  # Fields values for the pyvsc data types
                data.append([key, type(key), sys.getsizeof(key), value.get_val()])
            except Exception:
                data.append([key, type(key), sys.getsizeof(key), value])
    df = pd.DataFrame(data, columns=['Name', 'Type', 'Size', 'Value'])
    df['Value'] = df['Value'].apply(str)
    logging.info('\n' + tabulate(df, headers='keys', tablefmt='psql'))
