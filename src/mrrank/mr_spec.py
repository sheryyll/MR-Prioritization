"""
Declarative MR specs for MR-Rank (user-defined MRs, no uploaded code).

A spec is plain JSON-able data:

    {"id": "U01", "name": "Rotate 2deg", "type": "rotate", "params": {"angle": 2.0}}
    {"id": "U02", "name": "Flip + noise", "type": "composite",
     "steps": [{"type": "flip"}, {"type": "noise", "params": {"sigma": 0.004}}]}

`build_mr_from_spec` validates type and parameter ranges, then maps the spec
onto the existing pure transforms in mr_engine. It never evaluates user code.

Torch-free: safe to import from the lean web app.
"""

from __future__ import annotations

import math
import re
from functools import partial
from typing import Any

from mrrank import mr_engine as E

MAX_COMPOSITE_STEPS = 3
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,16}$")

# type -> (category, {param: (min, max, default)}, mr_type used in the encoding tables)
# Categories/mr_types mirror how the canonical 20 MRs are labelled.
_TYPES: dict[str, dict[str, Any]] = {
    "flip":        {"category": "geometric",   "mr_type": "flip",        "params": {}},
    "rotate":      {"category": "geometric",   "mr_type": "rotate",      "params": {"angle": (-45.0, 45.0, None)}},
    "translation": {"category": "geometric",   "mr_type": "translation", "params": {"dx": (-8.0, 8.0, 0.0), "dy": (-8.0, 8.0, 0.0)}},
    "shear":       {"category": "geometric",   "mr_type": "shear",       "params": {"shear_factor": (-0.5, 0.5, None)}},
    "scale":       {"category": "geometric",   "mr_type": "scale",       "params": {"factor": (0.5, 1.5, None)}},
    "noise":       {"category": "noise",       "mr_type": "noise",       "params": {"sigma": (0.0, 0.2, None)}},
    "salt_pepper": {"category": "noise",       "mr_type": "salt_pepper", "params": {"amount": (0.0, 0.1, None)}},
    "sharpen":     {"category": "noise",       "mr_type": "sharpen",     "params": {"amount": (0.0, 3.0, None), "ksize": (3, 5, 3)}},
    "brightness":  {"category": "photometric", "mr_type": "brightness",  "params": {"delta": (-100.0, 100.0, None)}},
    "contrast":    {"category": "photometric", "mr_type": "contrast",    "params": {"factor": (0.5, 2.0, None)}},
    "saturation":  {"category": "photometric", "mr_type": "saturation",  "params": {"factor": (0.0, 2.0, None)}},
    "posterize":   {"category": "photometric", "mr_type": "posterize",   "params": {"bits": (1, 8, None)}},
    "jpeg":        {"category": "photometric", "mr_type": "jpeg",        "params": {"quality": (1, 100, None)}},
}
ALLOWED_TYPES = sorted(_TYPES) + ["composite"]


class MRSpecError(ValueError):
    """Raised for any invalid user-supplied MR spec."""


def _num(name: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MRSpecError(f"param '{name}' must be a number, got {value!r}")
    if not math.isfinite(value):
        raise MRSpecError(f"param '{name}' must be finite")
    if not lo <= value <= hi:
        raise MRSpecError(f"param '{name}'={value} outside allowed range [{lo}, {hi}]")
    return float(value)


def _clean_params(type_: str, raw: Any) -> dict[str, float]:
    if type_ not in _TYPES:
        raise MRSpecError(f"unknown type '{type_}'. Allowed: {ALLOWED_TYPES}")
    raw = raw or {}
    if not isinstance(raw, dict):
        raise MRSpecError("'params' must be an object")
    schema = _TYPES[type_]["params"]
    extra = set(raw) - set(schema)
    if extra:
        raise MRSpecError(f"type '{type_}' does not accept params {sorted(extra)}; allowed: {sorted(schema)}")
    out = {}
    for name, (lo, hi, default) in schema.items():
        if name in raw:
            out[name] = _num(name, raw[name], lo, hi)
        elif default is None:
            raise MRSpecError(f"type '{type_}' requires param '{name}'")
        else:
            out[name] = float(default)
    if type_ == "sharpen" and int(out["ksize"]) not in (3, 5):
        raise MRSpecError("sharpen ksize must be 3 or 5")
    return out


def _step_fn(type_: str, p: dict[str, float], mr_id: str, global_seed: int):
    """Map a validated (type, params) pair onto the existing pure transforms."""
    if type_ == "flip":
        return E._hflip
    if type_ == "rotate":
        return partial(E._rotate, angle=p["angle"])
    if type_ == "translation":
        return partial(E._translate, dx=p["dx"], dy=p["dy"])
    if type_ == "shear":
        return partial(E._shear, shear_factor=p["shear_factor"])
    if type_ == "scale":
        return partial(E._scale, factor=p["factor"])
    if type_ == "noise":  # sigma is a fraction of 255, as in the canonical MRs
        return partial(E._gaussian_noise, sigma_uint8=p["sigma"] * 255, mr_id=mr_id, global_seed=global_seed)
    if type_ == "salt_pepper":
        return partial(E._salt_pepper, amount=p["amount"], mr_id=mr_id, global_seed=global_seed)
    if type_ == "sharpen":
        return partial(E._unsharp_mask, amount=p["amount"], ksize=int(p["ksize"]))
    if type_ == "brightness":
        return partial(E._brightness, delta=p["delta"])
    if type_ == "contrast":
        return partial(E._contrast, factor=p["factor"])
    if type_ == "saturation":
        return partial(E._saturation, factor=p["factor"])
    if type_ == "posterize":
        return partial(E._posterize, bits=int(p["bits"]))
    if type_ == "jpeg":
        return partial(E._jpeg_compress, quality=int(p["quality"]))
    raise MRSpecError(f"unknown type '{type_}'")  # unreachable after validation


def _magnitude(type_: str, p: dict[str, float]) -> float:
    if type_ == "flip":
        return 0
    if type_ == "translation":
        return max(abs(p["dx"]), abs(p["dy"]))
    return next(iter(p.values()))  # first (primary) parameter


def build_mr_from_spec(spec: dict, global_seed: int | None = None,
                       taken_ids: set[str] | None = None) -> E.MetamorphicRelation:
    """Validate `spec` and return a MetamorphicRelation. Raises MRSpecError."""
    if global_seed is None:
        from mrrank import config
        global_seed = config.SEED
    if not isinstance(spec, dict):
        raise MRSpecError("spec must be an object")

    mr_id = spec.get("id")
    if not isinstance(mr_id, str) or not ID_PATTERN.match(mr_id):
        raise MRSpecError("'id' must match [A-Za-z0-9_-]{1,16}")
    if taken_ids is not None and mr_id in taken_ids:
        raise MRSpecError(f"id '{mr_id}' already in use")
    name = str(spec.get("name") or mr_id)[:80]
    type_ = spec.get("type")

    if type_ == "composite":
        steps = spec.get("steps")
        if not isinstance(steps, list) or not 2 <= len(steps) <= MAX_COMPOSITE_STEPS:
            raise MRSpecError(f"composite needs 2..{MAX_COMPOSITE_STEPS} steps")
        fns = []
        for s in steps:
            if not isinstance(s, dict) or s.get("type") in (None, "composite"):
                raise MRSpecError("each composite step needs a non-composite 'type'")
            p = _clean_params(s["type"], s.get("params"))
            fns.append(_step_fn(s["type"], p, mr_id, global_seed))

        def fn(img, _fns=tuple(fns)):
            for f in _fns:
                img = f(img)
            return img

        return E.MetamorphicRelation(mr_id, name, "composite", "composite", 0, True, fn)

    p = _clean_params(type_, spec.get("params"))
    t = _TYPES[type_]
    return E.MetamorphicRelation(mr_id, name, t["category"], t["mr_type"],
                                 _magnitude(type_, p), False,
                                 _step_fn(type_, p, mr_id, global_seed))


def build_mrs_from_specs(specs: list[dict], global_seed: int | None = None,
                         max_mrs: int = 20) -> list[E.MetamorphicRelation]:
    """Build a batch; ids must be unique and must not collide with the canonical 20."""
    if not isinstance(specs, list) or not 1 <= len(specs) <= max_mrs:
        raise MRSpecError(f"provide between 1 and {max_mrs} MR specs")
    taken = set(E.MR_BY_ID)
    out = []
    for s in specs:
        mr = build_mr_from_spec(s, global_seed, taken)
        taken.add(mr.id)
        out.append(mr)
    return out
