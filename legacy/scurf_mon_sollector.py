#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Slurm cluster statistics collector.

Compatible with:
    Python 3.6+
    Python 3.13+

Requires:
    PyYAML


============================================================
SLURM COMMAND PATHS
============================================================

All Slurm commands use absolute paths.
"""

from __future__ import print_function

import argparse
import datetime
import re
import subprocess
import sys
import time

from collections import OrderedDict

import yaml


# ============================================================
# Absolute Slurm command paths
# ============================================================

SLURM_SCONTROL = (
    "/cm/shared/apps/slurm/current/bin/scontrol"
)

SLURM_SQUEUE = (
    "/cm/shared/apps/slurm/current/bin/squeue"
)


# ============================================================
# High-memory threshold
# ============================================================

HIGH_MEMORY_THRESHOLD_GB = 800

HIGH_MEMORY_THRESHOLD_MB = (
    HIGH_MEMORY_THRESHOLD_GB * 1024
)


# ============================================================
# Slurm node states considered unavailable
# ============================================================

DOWN_STATES = set([
    "down",
    "drain",
    "drained",
    "draining",
    "fail",
    "failing",
    "future",
    "maint",
    "perfctrs",
    "planned",
    "power_down",
    "power_up",
    "reserved",
    "unknown"
])

DOWN_STATES_STRICT = set([
    "down",
    "fail",
    "failing",
    "future",
    "maint",
    "perfctrs",
    "planned",
    "power_down",
    "power_up",
    "reserved",
    "unknown"
])

# ============================================================
# YAML support
# ============================================================

def represent_ordered_dict(dumper, data):

    return dumper.represent_dict(
        data.items()
    )


yaml.add_representer(
    OrderedDict,
    represent_ordered_dict,
    Dumper=yaml.SafeDumper
)



# ============================================================
# Run command
# ============================================================

def run_command(command):
    """
    Run a command and return stdout.

    Raises RuntimeError on failure.
    """

    try:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        stdout, stderr = process.communicate()

    except OSError as exc:

        raise RuntimeError(
            "Unable to execute {}: {}".format(
                command[0],
                exc
            )
        )

    if process.returncode != 0:

        raise RuntimeError(
            "Command failed: {}\n{}".format(
                " ".join(command),
                stderr.strip()
            )
        )

    return stdout.strip()


# ============================================================
# Basic helpers
# ============================================================

def parse_number(value):
    """
    Extract the first integer.

    Examples:

        123
        123G
        123/456
    """

    if not value:
        return 0

    match = re.match(
        r"^\s*(\d+)",
        value
    )

    if match:

        return int(
            match.group(1)
        )

    return 0


def count_lines(output):
    """
    Count non-empty lines.
    """

    if not output:
        return 0

    return sum(
        1
        for line in output.splitlines()
        if line.strip()
    )


def normalize_gpu_type(gpu_type):
    """
    Normalize GPU type names.

    Examples:

        A100       -> a100
        H200       -> h200
        RTX6000    -> rtx6000
    """

    if not gpu_type:

        return "generic"

    if gpu_type == (
        "nvidia_rtx_pro_6000_blackwell"
    ):

        gpu_type = "6000"

    return gpu_type.strip().lower()


# ============================================================
# Parse scontrol key=value record
# ============================================================

def parse_key_value_record(record):
    """
    Parse one line from:

        scontrol show node -o
    """

    result = {}

    if not record:
        return result

    for item in record.split():

        if "=" not in item:
            continue

        key, value = item.split(
            "=",
            1
        )

        result[key] = value

    return result


# ============================================================
# Compute node filter
# ============================================================

def is_compute_node(node):
    """
    Determine whether a node is a compute node.

    A valid Partitions= field identifies compute nodes.
    """

    partitions = node.get(
        "Partitions",
        ""
    )

    if not partitions:
        return False

    partitions = partitions.strip()

    if partitions in (
        "",
        "(null)",
        "N/A",
        "None",
        "none",
        "-"
    ):

        return False

    return True


# ============================================================
# Parse GPU GRES
# ============================================================

def parse_gpu_gres(gres):
    """
    Parse GPU information from Gres or GresUsed.

    Examples:

        gpu:8
        gpu:a100:8
        gpu:h200:8
        gpu:rtx6000:4
        gpu:a100:4,gpu:h200:4
    """

    result = OrderedDict()

    if not gres:
        return result

    gres = gres.strip()

    if gres in (
        "",
        "(null)",
        "N/A",
        "None",
        "none",
        "-"
    ):

        return result

    for item in gres.split(","):

        item = item.strip()

        if not item:
            continue

        item = item.split(
            "(",
            1
        )[0]

        fields = item.split(":")

        # ----------------------------------------------------
        # gpu:8
        # ----------------------------------------------------

        if len(fields) == 2:

            if fields[0].lower() != "gpu":
                continue

            try:

                count = int(
                    fields[1]
                )

            except ValueError:

                continue

            if count > 0:

                result["generic"] = (
                    result.get(
                        "generic",
                        0
                    )
                    + count
                )

            continue

        # ----------------------------------------------------
        # gpu:a100:8
        # gpu:h200:8
        # ----------------------------------------------------

        if len(fields) >= 3:

            if fields[0].lower() != "gpu":
                continue

            gpu_type = normalize_gpu_type(
                fields[1]
            )

            try:

                count = int(
                    fields[2]
                )

            except ValueError:

                continue

            if count <= 0:
                continue

            result[gpu_type] = (
                result.get(
                    gpu_type,
                    0
                )
                + count
            )

    return result


# ============================================================
# GPU node test
# ============================================================

def is_gpu_node(gres):

    return bool(
        parse_gpu_gres(
            gres
        )
    )


# ============================================================
# Parse Slurm memory value
# ============================================================

def parse_slurm_real_memory(value):
    """
    Parse Slurm RealMemory.

    Slurm normally reports RealMemory in MB when there
    is no suffix.

    Examples:

        1024000       -> 1024000 MB
        1000G         -> 1024000 MB
        1T            -> 1048576 MB
    """

    if not value:
        return 0

    value = str(value).strip()

    if not value:
        return 0

    if re.match(
        r"^\d+$",
        value
    ):

        return int(value)

    match = re.match(
        r"^(\d+(?:\.\d+)?)([KMGTP])(?:B)?$",
        value,
        re.IGNORECASE
    )

    if not match:
        return 0

    number = float(
        match.group(1)
    )

    unit = match.group(2).upper()

    multipliers = {
        "K": 1.0 / 1024.0,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
        "P": 1024 * 1024 * 1024
    }

    return int(
        number * multipliers[unit]
    )


# ============================================================
# Format memory
# ============================================================

def format_memory_gb(memory_mb):

    return round(
        float(memory_mb) / 1024.0,
        2
    )


# ============================================================
# Human-readable wait time
# ============================================================

def format_duration(seconds):
    """
    Format wait time using the largest suitable unit.

    Examples:

        3.1 days
        2.0 hours
        0.1 hours
        45.0 minutes
        12.0 seconds
    """

    if seconds <= 0:

        return "0.0 seconds"

    if seconds >= 86400:

        return "{:.1f} days".format(
            float(seconds) / 86400.0
        )

    if seconds >= 3600:

        return "{:.1f} hours".format(
            float(seconds) / 3600.0
        )

    if seconds >= 60:

        return "{:.1f} minutes".format(
            float(seconds) / 60.0
        )

    return "{:.1f} seconds".format(
        float(seconds)
    )


# ============================================================
# Parse Slurm datetime
# ============================================================

def parse_slurm_datetime(value):
    """
    Parse squeue %V submission time.

    Expected:

        YYYY-MM-DDTHH:MM:SS
    """

    if not value:
        return 0

    value = value.strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M"
    ]

    for fmt in formats:

        try:

            dt = datetime.datetime.strptime(
                value,
                fmt
            )

            return time.mktime(
                dt.timetuple()
            )

        except ValueError:

            continue

    return 0


# ============================================================
# Parse memory allocation from TRES
# ============================================================

def parse_memory_from_tres(value):
    """
    Extract memory allocation from AllocTRES.

    Examples:

        cpu=4,mem=256G,node=1
        cpu=8,mem=102400M
        mem=512G
        mem=1048576K

    Returns MB.
    """

    if not value:
        return 0

    match = re.search(
        r"(?:^|,)mem=([^,]+)",
        value
    )

    if not match:
        return 0

    return parse_slurm_real_memory(
        match.group(1)
    )


# ============================================================
# PART 2 OF 5
#
# GPU allocation
# Node discovery
# ============================================================


# ============================================================
# Parse GPU information from TRES
# ============================================================

def parse_gpu_tres(value):

    result = OrderedDict()

    if not value:
        return result

    value = value.strip()

    if not value:
        return result

    typed_pattern = re.compile(
        r"(?:gres/)?gpu:([^:=,]+)=(\d+)"
    )

    for match in typed_pattern.finditer(
        value
    ):

        gpu_type = normalize_gpu_type(
            match.group(1)
        )

        count = int(
            match.group(2)
        )

        if count <= 0:
            continue

        result[gpu_type] = (
            result.get(
                gpu_type,
                0
            )
            + count
        )

    if result:
        return result

    generic_match = re.search(
        r"(?:gres/)?gpu=(\d+)",
        value
    )

    if generic_match:

        count = int(
            generic_match.group(1)
        )

        if count > 0:

            result["generic"] = count

    return result


# ============================================================
# Resolve GPU allocation
# ============================================================

def resolve_gpu_allocation(
    total_gpu_info,
    allocated_gpu_info
):

    if not allocated_gpu_info:
        return {}

    result = OrderedDict()

    configured_types = [
        gpu_type
        for gpu_type in total_gpu_info
        if gpu_type != "generic"
    ]

    for gpu_type, count in (
        allocated_gpu_info.items()
    ):

        if count <= 0:
            continue

        if gpu_type != "generic":

            result[gpu_type] = (
                result.get(
                    gpu_type,
                    0
                )
                + count
            )

            continue

        if len(configured_types) == 1:

            real_gpu_type = (
                configured_types[0]
            )

            result[real_gpu_type] = (
                result.get(
                    real_gpu_type,
                    0
                )
                + count
            )

        else:

            result["generic"] = (
                result.get(
                    "generic",
                    0
                )
                + count
            )

    return result


# ============================================================
# Get compute nodes
# ============================================================

def get_node_info():

    output = run_command([
        SLURM_SCONTROL,
        "show",
        "node",
        "-o"
    ])

    nodes = []

    if not output:
        return nodes

    for line in output.splitlines():

        line = line.strip()

        if not line:
            continue

        record = parse_key_value_record(
            line
        )

        if not record:
            continue

        node_name = record.get(
            "NodeName"
        )

        if not node_name:
            continue

        if not is_compute_node(
            record
        ):
            continue

        nodes.append(
            record
        )

    return nodes


# ============================================================
# Node and CPU statistics
# ============================================================

def get_node_stats(nodes):

    stats = OrderedDict()

    stats["nodes"] = OrderedDict()

    stats["nodes"]["total"] = 0
    stats["nodes"]["idle"] = 0
    stats["nodes"]["allocated"] = 0
    stats["nodes"]["down"] = 0
    stats["nodes"]["available"] = 0

    stats["cpu"] = OrderedDict()

    stats["cpu"]["total"] = 0
    stats["cpu"]["allocated"] = 0
    stats["cpu"]["idle"] = 0
    stats["cpu"]["down"] = 0

    for node in nodes:

        stats["nodes"]["total"] += 1

        state = node.get(
            "State",
            ""
        ).lower()

        state_tokens = set(
            token.strip().lower()
            for token in state.split("+")
        )

        is_down = bool(
            state_tokens.intersection(
                DOWN_STATES
            )
        )

        # ----------------------------------------------------
        # Down takes priority.
        # ----------------------------------------------------

        if is_down:

            stats["nodes"]["down"] += 1

        elif (
            "allocated" in state_tokens
            or
            "alloc" in state_tokens
            or
            "mixed" in state_tokens
        ):

            stats["nodes"]["allocated"] += 1

        elif "idle" in state_tokens:

            stats["nodes"]["idle"] += 1

        if not is_down:

            stats["nodes"]["available"] += 1

        cpu_total = parse_number(
            node.get(
                "CPUTot",
                "0"
            )
        )

        cpu_allocated = parse_number(
            node.get(
                "CPUAlloc",
                "0"
            )
        )

        stats["cpu"]["total"] += (
            cpu_total
        )

        stats["cpu"]["allocated"] += (
            cpu_allocated
        )

        if is_down:

            stats["cpu"]["down"] += (
                cpu_total
            )

    stats["cpu"]["idle"] = max(
        0,
        stats["cpu"]["total"]
        -
        stats["cpu"]["allocated"]
    )

    return stats


# ============================================================
# Detailed node-state statistics
# ============================================================

def get_node_state_stats(nodes):

    stats = OrderedDict()

    stats["total"] = 0
    stats["idle"] = 0
    stats["mixed"] = 0
    stats["allocated"] = 0
    stats["draining"] = 0
    stats["down"] = 0

    for node in nodes:

        stats["total"] += 1
        state = node.get(
            "State",
            ""
        ).lower()

        state_tokens = set(
            token.strip().lower()
            for token in state.split("+")
        )

        is_down = bool(
            state_tokens.intersection(
                DOWN_STATES_STRICT #DOWN_STATES
            )
        )

        # ----------------------------------------------------
        # Down has highest priority.
        # ----------------------------------------------------

        if is_down:

            stats["down"] += 1

        elif (
            "drain" in state_tokens
            or
            "draining" in state_tokens
            or
            "drained" in state_tokens
        ):

            stats["draining"] += 1

        elif "mixed" in state_tokens:

            stats["mixed"] += 1

        elif (
            "allocated" in state_tokens
            or
            "alloc" in state_tokens
        ):

            stats["allocated"] += 1

        elif "idle" in state_tokens:

            stats["idle"] += 1

    return stats


# ============================================================
# PART 3 OF 5
#
# GPU statistics
# ============================================================


def get_gpu_stats(nodes):
    """
    Calculate GPU statistics.

    GPU total:
        Gres=

    GPU allocation:
        GresUsed=

    Fallback:
        AllocTRES=

    Generic GPU allocation is resolved from the GPU type
    configured on the same node.
    """

    gpu = OrderedDict()

    gpu["total"] = 0
    gpu["allocated"] = 0
    gpu["idle"] = 0
    gpu["down"] = 0

    gpu["types"] = OrderedDict()

    for node in nodes:

        node_name = node.get(
            "NodeName",
            "unknown"
        )

        state = node.get(
            "State",
            ""
        ).lower()

        state_tokens = set(
            token.strip().lower()
            for token in state.split("+")
        )

        is_down = bool(
            state_tokens.intersection(
                DOWN_STATES
            )
        )

        gres = node.get(
            "Gres",
            ""
        )

        if not is_gpu_node(
            gres
        ):
            continue

        # ----------------------------------------------------
        # Total GPUs.
        # ----------------------------------------------------

        total_gpu_info = parse_gpu_gres(
            gres
        )

        # ----------------------------------------------------
        # Preferred allocation:
        #
        # GresUsed=
        # ----------------------------------------------------

        allocated_gpu_info = parse_gpu_gres(
            node.get(
                "GresUsed",
                ""
            )
        )

        # ----------------------------------------------------
        # Fallback:
        #
        # AllocTRES=
        # ----------------------------------------------------

        if not allocated_gpu_info:

            allocated_gpu_info = parse_gpu_tres(
                node.get(
                    "AllocTRES",
                    ""
                )
            )

        # ----------------------------------------------------
        # Resolve generic allocation.
        # ----------------------------------------------------

        allocated_gpu_info = (
            resolve_gpu_allocation(
                total_gpu_info,
                allocated_gpu_info
            )
        )

        # ====================================================
        # GPU totals.
        # ====================================================

        for gpu_type, count in (
            total_gpu_info.items()
        ):

            gpu_type = normalize_gpu_type(
                gpu_type
            )

            if gpu_type not in gpu["types"]:

                gpu["types"][gpu_type] = (
                    OrderedDict()
                )

                gpu["types"][gpu_type]["total"] = 0
                gpu["types"][gpu_type]["allocated"] = 0
                gpu["types"][gpu_type]["idle"] = 0
                gpu["types"][gpu_type]["down"] = 0

            gpu["types"][gpu_type]["total"] += (
                count
            )

            gpu["total"] += count

            if is_down:

                gpu["types"][gpu_type]["down"] += (
                    count
                )

                gpu["down"] += count

        # ====================================================
        # GPU allocations.
        # ====================================================

        for gpu_type, count in (
            allocated_gpu_info.items()
        ):

            gpu_type = normalize_gpu_type(
                gpu_type
            )

            if gpu_type not in gpu["types"]:

                gpu["types"][gpu_type] = (
                    OrderedDict()
                )

                gpu["types"][gpu_type]["total"] = 0
                gpu["types"][gpu_type]["allocated"] = 0
                gpu["types"][gpu_type]["idle"] = 0
                gpu["types"][gpu_type]["down"] = 0

            gpu["types"][gpu_type]["allocated"] += (
                count
            )

            gpu["allocated"] += count

        # ====================================================
        # Sanity check.
        # ====================================================

        total_node_gpus = sum(
            total_gpu_info.values()
        )

        allocated_node_gpus = sum(
            allocated_gpu_info.values()
        )

        if allocated_node_gpus > total_node_gpus:

            sys.stderr.write(
                "WARNING: node {} reports {} allocated "
                "GPUs but only {} total GPUs\n".format(
                    node_name,
                    allocated_node_gpus,
                    total_node_gpus
                )
            )

    # ========================================================
    # GPU idle per type.
    # ========================================================

    for gpu_type in gpu["types"]:

        data = gpu["types"][gpu_type]

        data["idle"] = max(
            0,
            data["total"]
            -
            data["allocated"]
        )

    # ========================================================
    # Overall GPU idle.
    # ========================================================

    gpu["idle"] = max(
        0,
        gpu["total"]
        -
        gpu["allocated"]
    )

    return gpu


# ============================================================
# PART 4 OF 5
#
# High-memory node statistics
# Partition/job statistics
# ============================================================


# ============================================================
# High-memory configuration
# ============================================================

HIGH_MEMORY_THRESHOLD_GB = 800

HIGH_MEMORY_THRESHOLD_MB = (
    HIGH_MEMORY_THRESHOLD_GB * 1024
)


# ============================================================
# Parse Slurm RealMemory
# ============================================================

def parse_slurm_real_memory(value):
    """
    Parse Slurm RealMemory.

    Slurm normally reports RealMemory in MB when no
    suffix is present.

    Examples:

        1024000       -> 1024000 MB
        1024000M      -> 1024000 MB
        1000G         -> 1024000 MB
        1T            -> 1048576 MB
    """

    if not value:
        return 0

    value = str(value).strip()

    if not value:
        return 0

    # --------------------------------------------------------
    # No suffix = MB.
    # --------------------------------------------------------

    if re.match(
        r"^\d+$",
        value
    ):

        return int(value)

    # --------------------------------------------------------
    # Explicit suffix.
    # --------------------------------------------------------

    match = re.match(
        r"^(\d+(?:\.\d+)?)([KMGTP])(?:B)?$",
        value,
        re.IGNORECASE
    )

    if not match:
        return 0

    number = float(
        match.group(1)
    )

    unit = match.group(2).upper()

    multipliers = {
        "K": 1.0 / 1024.0,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
        "P": 1024 * 1024 * 1024
    }

    return int(
        number * multipliers[unit]
    )


# ============================================================
# Parse memory allocation from AllocTRES
# ============================================================

def parse_memory_from_tres(value):
    """
    Extract allocated memory from AllocTRES.

    Examples:

        cpu=32,mem=500G
        cpu=64,mem=256000M
        cpu=16,gres/gpu=4,mem=128G

    Returns memory in MB.
    """

    if not value:
        return 0

    # --------------------------------------------------------
    # Find mem= value.
    # --------------------------------------------------------

    match = re.search(
        r"(?:^|,)mem=([^,]+)",
        value
    )

    if not match:
        return 0

    memory_value = (
        match.group(1).strip()
    )

    if not memory_value:
        return 0

    # --------------------------------------------------------
    # Parse memory with suffix.
    # --------------------------------------------------------

    return parse_slurm_real_memory(
        memory_value
    )


# ============================================================
# Format memory as GB
# ============================================================

def format_memory_gb(memory_mb):
    """
    Convert MB to GB.
    """

    return round(
        float(memory_mb) / 1024.0,
        2
    )


# ============================================================
# High-memory node statistics
# ============================================================

def get_high_memory_stats(nodes):
    """
    Calculate memory utilization for nodes with more than
    800 GB RAM.

    Physical memory:

        RealMemory=

    Allocated memory:

        AllocTRES=...mem=...

    Fallback:

        MemAlloc=

    Only nodes with:

        RealMemory > 800 GB

    are included.
    """

    stats = OrderedDict()

    stats["threshold_gb"] = (
        HIGH_MEMORY_THRESHOLD_GB
    )

    stats["nodes"] = OrderedDict()

    stats["nodes"]["total"] = 0
    stats["nodes"]["idle"] = 0
    stats["nodes"]["allocated"] = 0
    stats["nodes"]["mixed"] = 0
    stats["nodes"]["down"] = 0

    stats["memory"] = OrderedDict()

    stats["memory"]["total_gb"] = 0
    stats["memory"]["allocated_gb"] = 0
    stats["memory"]["idle_gb"] = 0
    stats["memory"]["down_gb"] = 0

    stats["nodes_detail"] = []

    # ========================================================
    # Process nodes.
    # ========================================================

    for node in nodes:

        real_memory_mb = (
            parse_slurm_real_memory(
                node.get(
                    "RealMemory",
                    "0"
                )
            )
        )

        # ----------------------------------------------------
        # Only nodes > 800 GB.
        # ----------------------------------------------------

        if real_memory_mb <= (
            HIGH_MEMORY_THRESHOLD_MB
        ):
            continue

        node_name = node.get(
            "NodeName",
            "unknown"
        )

        state = node.get(
            "State",
            ""
        ).lower()

        state_tokens = set(
            token.strip().lower()
            for token in state.split("+")
        )

        is_down = bool(
            state_tokens.intersection(
                DOWN_STATES
            )
        )

        # ----------------------------------------------------
        # Determine node state.
        # ----------------------------------------------------

        if is_down:

            node_state = "down"

        elif "mixed" in state_tokens:

            node_state = "mixed"

        elif (
            "allocated" in state_tokens
            or
            "alloc" in state_tokens
        ):

            node_state = "allocated"

        elif "idle" in state_tokens:

            node_state = "idle"

        else:

            node_state = "other"

        # ====================================================
        # Memory allocation.
        # ====================================================

        allocated_mb = (
            parse_memory_from_tres(
                node.get(
                    "AllocTRES",
                    ""
                )
            )
        )

        # ----------------------------------------------------
        # Fallback to MemAlloc.
        # ----------------------------------------------------

        if allocated_mb == 0:

            allocated_mb = (
                parse_slurm_real_memory(
                    node.get(
                        "MemAlloc",
                        "0"
                    )
                )
            )

        # ----------------------------------------------------
        # Do not allow allocation > physical memory.
        # ----------------------------------------------------

        allocated_mb = min(
            allocated_mb,
            real_memory_mb
        )

        idle_mb = max(
            0,
            real_memory_mb
            -
            allocated_mb
        )

        # ====================================================
        # Node totals.
        # ====================================================

        stats["nodes"]["total"] += 1

        if is_down:

            stats["nodes"]["down"] += 1

        elif node_state == "idle":

            stats["nodes"]["idle"] += 1

        elif node_state == "allocated":

            stats["nodes"]["allocated"] += 1

        elif node_state == "mixed":

            stats["nodes"]["mixed"] += 1

        # ====================================================
        # Memory totals.
        # ====================================================

        stats["memory"]["total_gb"] += (
            format_memory_gb(
                real_memory_mb
            )
        )

        stats["memory"]["allocated_gb"] += (
            format_memory_gb(
                allocated_mb
            )
        )

        if is_down:

            stats["memory"]["down_gb"] += (
                format_memory_gb(
                    real_memory_mb
                )
            )

        # ====================================================
        # Node detail.
        # ====================================================

        node_detail = OrderedDict()

        node_detail["node"] = node_name
        node_detail["state"] = node_state

        node_detail["memory_gb"] = (
            format_memory_gb(
                real_memory_mb
            )
        )

        node_detail["allocated_gb"] = (
            format_memory_gb(
                allocated_mb
            )
        )

        node_detail["idle_gb"] = (
            format_memory_gb(
                idle_mb
            )
        )

        if real_memory_mb > 0:

            node_detail[
                "utilization_percent"
            ] = round(
                (
                    float(allocated_mb)
                    /
                    float(real_memory_mb)
                )
                * 100.0,
                2
            )

        else:

            node_detail[
                "utilization_percent"
            ] = 0

        stats["nodes_detail"].append(
            node_detail
        )

    # ========================================================
    # Overall idle memory.
    # ========================================================

    stats["memory"]["idle_gb"] = max(
        0,
        round(
            stats["memory"]["total_gb"]
            -
            stats["memory"]["allocated_gb"]
            -
            stats["memory"]["down_gb"],
            2
        )
    )

    # --------------------------------------------------------
    # Round totals.
    # --------------------------------------------------------

    stats["memory"]["total_gb"] = round(
        stats["memory"]["total_gb"],
        2
    )

    stats["memory"]["allocated_gb"] = round(
        stats["memory"]["allocated_gb"],
        2
    )

    stats["memory"]["idle_gb"] = round(
        stats["memory"]["idle_gb"],
        2
    )

    stats["memory"]["down_gb"] = round(
        stats["memory"]["down_gb"],
        2
    )

    # --------------------------------------------------------
    # Overall memory utilization.
    # --------------------------------------------------------

    if stats["memory"]["total_gb"] > 0:

        stats["memory"]["utilization_percent"] = round(
            (
                float(
                    stats["memory"]["allocated_gb"]
                )
                /
                float(
                    stats["memory"]["total_gb"]
                )
            )
            * 100.0,
            2
        )

    else:

        stats["memory"]["utilization_percent"] = 0

    return stats


# ============================================================
# Parse Slurm submission time
# ============================================================

def parse_slurm_datetime(value):
    """
    Parse Slurm squeue %V submission time.

    Expected format:

        YYYY-MM-DDTHH:MM:SS

    Returns Unix timestamp.
    """

    if not value:
        return 0

    value = value.strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M"
    ]

    for fmt in formats:

        try:

            dt = datetime.datetime.strptime(
                value,
                fmt
            )

            return time.mktime(
                dt.timetuple()
            )

        except ValueError:

            continue

    return 0


# ============================================================
# Format pending wait time
# ============================================================

def format_duration(seconds):
    """
    Format wait time using the largest useful unit.

    Examples:

        3.1 days
        2.0 hours
        0.1 hours
        45.0 minutes
        12.0 seconds
    """

    if seconds <= 0:
        return "0.0 seconds"

    if seconds >= 86400:

        return "{:.1f} days".format(
            float(seconds) / 86400.0
        )

    if seconds >= 3600:

        return "{:.1f} hours".format(
            float(seconds) / 3600.0
        )

    if seconds >= 60:

        return "{:.1f} minutes".format(
            float(seconds) / 60.0
        )

    return "{:.1f} seconds".format(
        float(seconds)
    )


# ============================================================
# Overall job statistics
# ============================================================

def get_job_stats():
    """
    Get overall job counts.

    -a includes all partitions.
    """

    # --------------------------------------------------------
    # ALL jobs
    # --------------------------------------------------------

    total_output = run_command([
        SLURM_SQUEUE,
        "-a",
        "-h",
        "-o",
        "%i"
    ])

    # --------------------------------------------------------
    # RUNNING jobs
    # --------------------------------------------------------

    running_output = run_command([
        SLURM_SQUEUE,
        "-a",
        "-h",
        "-t",
        "RUNNING",
        "-o",
        "%i"
    ])

    # --------------------------------------------------------
    # PENDING jobs
    # --------------------------------------------------------

    pending_output = run_command([
        SLURM_SQUEUE,
        "-a",
        "-h",
        "-t",
        "PENDING",
        "-o",
        "%i"
    ])

    jobs = OrderedDict()

    jobs["total"] = count_lines(
        total_output
    )

    jobs["running"] = count_lines(
        running_output
    )

    jobs["pending"] = count_lines(
        pending_output
    )

    return jobs


# ============================================================
# Per-partition statistics
# ============================================================


def get_partition_stats():
    """
    Get independent statistics for each Slurm partition.

    A job with:

        Partition=b40x4,b40x4-long,h200x4

    is counted independently in:

        b40x4
        b40x4-long
        h200x4

    It is NOT stored under the combined partition name.

    Output fields:

        total
        running
        pending
        longest_pending_wait_minutes
        longest_pending_job_id
        average_pending_wait_minutes
        median_pending_wait_minutes
    """

    output = run_command([
        SLURM_SQUEUE,
        "-a",
        "-h",
        "-o",
        "%i|%F|%K|%P|%T|%V"
    ])

    partitions = OrderedDict()

    if not output:
        return partitions

    now = time.time()

    for line in output.splitlines():

        line = line.strip()

        if not line:
            continue

        fields = line.split("|")

        if len(fields) < 6:
            continue
        job_id = fields[0].strip()
        array_job_id = fields[1].strip()
        array_task_id = fields[2].strip()
        partition_field = fields[3].strip()
        state = fields[4].strip().upper()
        submit_time = fields[5].strip()

        if (
            array_job_id
            and array_task_id
            and array_task_id != "N/A"
            and array_task_id != "NO_VAL"
        ):

            job_id = "{}_[{}]".format(
             array_job_id,
                array_task_id
            )

#        job_id = fields[0].strip()
#        partition_field = fields[1].strip()
#        state = fields[2].strip().upper()
#        submit_time = fields[3].strip()

        if not partition_field:

            partition_field = "unknown"

        # ----------------------------------------------------
        # Split multi-partition jobs.
        # ----------------------------------------------------

        job_partitions = []

        for partition in (
            partition_field.split(",")
        ):

            partition = partition.strip()

            if not partition:
                continue

            if partition not in job_partitions:

                job_partitions.append(
                    partition
                )

        if not job_partitions:

            job_partitions = [
                "unknown"
            ]

        # ====================================================
        # Process each partition independently.
        # ====================================================

        for partition in job_partitions:

            if partition not in partitions:

                partitions[partition] = (
                    OrderedDict()
                )

                partitions[partition]["total"] = 0
                partitions[partition]["running"] = 0
                partitions[partition]["pending"] = 0

                # ------------------------------------------------
                # Minute-based output fields.
                # ------------------------------------------------

                partitions[partition][
                    "longest_pending_wait_minutes"
                ] = 0.0

                partitions[partition][
                    "longest_pending_job_id"
                ] = ""

                partitions[partition][
                    "average_pending_wait_minutes"
                ] = 0.0

                partitions[partition][
                    "median_pending_wait_minutes"
                ] = 0.0

                # ------------------------------------------------
                # Internal calculation values.
                # These are removed before YAML output.
                # ------------------------------------------------

                partitions[partition][
                    "_longest_pending_wait_seconds"
                ] = 0

                partitions[partition][
                    "_pending_waits"
                ] = []

            data = partitions[partition]

            # ------------------------------------------------
            # Total.
            # ------------------------------------------------

            data["total"] += 1

            # ------------------------------------------------
            # Running.
            # ------------------------------------------------

            if state in (
                "RUNNING",
                "COMPLETING"
            ):

                data["running"] += 1

            # ------------------------------------------------
            # Pending.
            # ------------------------------------------------

            elif state == "PENDING":

                data["pending"] += 1

                submit_epoch = (
                    parse_slurm_datetime(
                        submit_time
                    )
                )

                if submit_epoch > 0:

                    wait_seconds = max(
                        0,
                        int(
                            now - submit_epoch
                        )
                    )

                    # ----------------------------------------
                    # Save for average and median.
                    # ----------------------------------------

                    data[
                        "_pending_waits"
                    ].append(
                        wait_seconds
                    )

                    # ----------------------------------------
                    # Longest pending job.
                    # ----------------------------------------

                    if (
                        wait_seconds
                        >
                        data[
                            "_longest_pending_wait_seconds"
                        ]
                    ):

                        data[
                            "_longest_pending_wait_seconds"
                        ] = wait_seconds

                        data[
                            "longest_pending_wait_minutes"
                        ] = round(
                            float(wait_seconds) / 60.0,
                            2
                        )

                        data[
                            "longest_pending_job_id"
                        ] = job_id

    # ========================================================
    # Calculate average / median.
    # ========================================================

    for partition in partitions:

        data = partitions[partition]

        waits = data[
            "_pending_waits"
        ]

        if waits:

            # ------------------------------------------------
            # Average pending wait.
            # ------------------------------------------------

            average_wait = (
                float(sum(waits))
                /
                float(len(waits))
            )

            data[
                "average_pending_wait_minutes"
            ] = round(
                average_wait / 60.0,
                2
            )

            # ------------------------------------------------
            # Median pending wait.
            # ------------------------------------------------

            sorted_waits = sorted(
                waits
            )

            count = len(
                sorted_waits
            )

            middle = count // 2

            if count % 2 == 0:

                median_wait = (
                    (
                        sorted_waits[
                            middle - 1
                        ]
                        +
                        sorted_waits[
                            middle
                        ]
                    )
                    /
                    2.0
                )

            else:

                median_wait = (
                    sorted_waits[
                        middle
                    ]
                )

            data[
                "median_pending_wait_minutes"
            ] = round(
                median_wait / 60.0,
                2
            )

        else:

            data[
                "average_pending_wait_minutes"
            ] = 0.0

            data[
                "median_pending_wait_minutes"
            ] = 0.0

        # ----------------------------------------------------
        # Remove internal fields.
        # ----------------------------------------------------

        del data[
            "_pending_waits"
        ]

        del data[
            "_longest_pending_wait_seconds"
        ]

    return partitions


# ============================================================
# PART 5 OF 5
#
# Statistics assembly
# YAML output
# Main program
# ============================================================


# ============================================================
# Collect statistics
# ============================================================

def collect_stats():

    # --------------------------------------------------------
    # Get compute nodes.
    # --------------------------------------------------------

    nodes = get_node_info()

    # --------------------------------------------------------
    # Node / CPU statistics.
    # --------------------------------------------------------

    node_stats = get_node_stats(
        nodes
    )

    # --------------------------------------------------------
    # Detailed node-state statistics.
    #
    # idle / mixed / allocated / draining / down
    # --------------------------------------------------------

    node_state_stats = get_node_state_stats(
        nodes
    )

    # --------------------------------------------------------
    # GPU statistics.
    # --------------------------------------------------------

    gpu_stats = get_gpu_stats(
        nodes
    )

    # --------------------------------------------------------
    # High-memory statistics.
    # --------------------------------------------------------

    high_memory_stats = get_high_memory_stats(
        nodes
    )

    # --------------------------------------------------------
    # Overall job statistics.
    # --------------------------------------------------------

    job_stats = get_job_stats()

    # --------------------------------------------------------
    # Per-partition job statistics.
    # --------------------------------------------------------

    partition_stats = get_partition_stats()

    # ========================================================
    # Final YAML structure.
    # ========================================================

    stats = OrderedDict()

    # --------------------------------------------------------
    # Timestamp.
    # --------------------------------------------------------

    stats["timestamp"] = (
        datetime.datetime.now(
            datetime.timezone.utc
        ).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    # --------------------------------------------------------
    # Nodes.
    # --------------------------------------------------------

    stats["nodes"] = (
        node_stats["nodes"]
    )

    # --------------------------------------------------------
    # Node states.
    # --------------------------------------------------------

    stats["node_states"] = (
        node_state_stats
    )

    # --------------------------------------------------------
    # CPU.
    # --------------------------------------------------------

    stats["cpu"] = (
        node_stats["cpu"]
    )

    # --------------------------------------------------------
    # GPU.
    # --------------------------------------------------------

    stats["gpu"] = (
        gpu_stats
    )

    # --------------------------------------------------------
    # High-memory nodes.
    # --------------------------------------------------------

    stats["high_memory"] = (
        high_memory_stats
    )

    # --------------------------------------------------------
    # Overall jobs.
    # --------------------------------------------------------

    stats["jobs"] = (
        job_stats
    )

    # --------------------------------------------------------
    # Per-partition jobs.
    # --------------------------------------------------------

    stats["partitions"] = (
        partition_stats
    )

    return stats


# ============================================================
# Write YAML
# ============================================================

def write_yaml(
    data,
    filename
):
    """
    Write statistics to YAML.
    """

    with open(
        filename,
        "w"
    ) as output:

        yaml.safe_dump(
            data,
            output,
            default_flow_style=False
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Collect Slurm cluster statistics "
            "for compute nodes and write them to YAML."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        default="slurm_stats.yaml",
        help=(
            "Output YAML file "
            "(default: slurm_stats.yaml)"
        )
    )

    args = parser.parse_args()

    try:

        stats = collect_stats()

        write_yaml(
            stats,
            args.output
        )

        print(
            "Slurm statistics written to {}".format(
                args.output
            )
        )

    except RuntimeError as exc:

        print(
            "ERROR: {}".format(
                exc
            ),
            file=sys.stderr
        )

        sys.exit(1)


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()


