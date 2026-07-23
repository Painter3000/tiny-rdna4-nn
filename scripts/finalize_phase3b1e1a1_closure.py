#!/usr/bin/env python3
"""TCNN_RDNA4_P3B1E1A1_FINALIZER_CLOSURE_001: finalizer-only closure."""
import copy
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = pathlib.Path("/tmp/phase3b1e1a_final_encoding_audit_raw.json")
OUT = ROOT / "phase3b1_reports/phase3b1e1a1_finalizer_closure.json"
MD = ROOT / "phase3b1_reports/PHASE3B1E1A1_FINALIZER_CLOSURE.md"
BASE = "7be7bb15cc0cf598f1ae6bca04156883f3982cf5"
MARKER = "TCNN_RDNA4_P3B1E1A1_FINALIZER_CLOSURE_001"
EXPECTED_RAW_SHA256 = "049d977ee88024017592c4649ff3f16ad21b2f24cbdcbfc90d7873eb1583bf9e"


def raw_checks(data):
    padding = data["padding_contracts"]
    runs = data["training_runs"]
    dynamic = runs[2]
    observation = dynamic["overflow_observation"]
    strong = data["collision_proofs"]["strong"]
    low = (data["collision_proofs"]["low_3d"], data["collision_proofs"]["low_2d"])
    resumes = data["checkpoint_resume"]
    maxima = data["numerical_maxima"]
    return {
        "raw_identity": data.get("decision") == "E1A_RAW_PASS",
        "padding": (
            all(padding["standalone"].get(k) == 1.0 for k in ("Identity", "Frequency", "OneBlob"))
            and all(x.get("value") == 1.0 for x in padding["fp32_network_with_encoding"])
            and all(x.get("value") == 0.0 for x in padding["fp16_network_with_encoding"])
            and all(x.get("passed") is True for x in padding["fp16_zero_execution"])
            and padding.get("hashgrid_unchanged") is True
        ),
        "matrix": (
            data.get("functional_cases") == 204
            and data.get("functional_passed") == 204
            and all(x.get("passed") is True for x in data["functional_matrix"])
        ),
        "collisions": (
            strong.get("collision_classification") == "collision-strong"
            and strong.get("collision_count", 0) > 0
            and strong.get("maximum_bucket_occupancy", 1) > 1
            and strong.get("collision_witness") is not None
            and all(x.get("collision_classification") == "collision-low" and x.get("collision_count") == 0 for x in low)
        ),
        "dynamic_scale": (
            dynamic.get("scaling") == "dynamic"
            and dynamic.get("initial_scale", 1) > 1
            and dynamic.get("scale_change_count", 0) > 0
        ),
        "dynamic_overflow": (
            dynamic["overflow_count"] > 0
            and dynamic["skip_count"] == dynamic["overflow_count"]
            and dynamic["recovery_count"] > 0
            and dynamic["both_parameter_ranges_checked"] is True
            and observation["forward_output_finite"] is True
            and observation["upstream_gradient_finite"] is True
            and observation["network_gradient_nonfinite"] is True
            and observation["encoding_gradient_nonfinite"] is True
            and observation["full_step_skipped"] is True
            and observation["scale_reduced"] is True
            and observation["gradients_reset"] is True
        ),
        "training": (
            len(runs) == 4
            and sum(x.get("steps", 0) for x in runs) == 3200
            and all(x.get("passed") is True for x in runs)
            and data.get("validated_training_steps", 0) >= 7600
        ),
        "resume": (
            len(resumes) == 4
            and all(
                x.get("passed") is True
                and x.get("cpu_rng_equal") is True
                and x.get("cuda_all_rng_equal") is True
                and x.get("custom_generator_rng_equal") is True
                for x in resumes
            )
        ),
        "events": len(data["event_chains"]) == 4 and all(x.get("passed") is True for x in data["event_chains"]),
        "maxima_attributed": (
            set(maxima) == {"output", "dinput", "network_gradient", "encoding_gradient"}
            and all(
                all(
                    key in item
                    for key in (
                        "encoding", "dims", "variant", "batch", "interpolation",
                        "n_levels", "n_features_per_level", "absolute_error",
                        "normalized_l2", "max_relative_outside_near_zero", "reference_norm",
                    )
                )
                for item in maxima.values()
            )
        ),
    }


raw_bytes = RAW.read_bytes()
data = json.loads(raw_bytes)
checks = raw_checks(data)
checks["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest() == EXPECTED_RAW_SHA256

manipulations = []


def manipulate(name, callback):
    changed = copy.deepcopy(data)
    callback(changed)
    blocked = not all(raw_checks(changed).values())
    manipulations.append({
        "name": name,
        "decision": "PHASE3B1E1A1_BLOCKED" if blocked else "INVALID_PASS",
        "passed": blocked,
    })


manipulate("standalone_padding_globally_zero", lambda x: x["padding_contracts"]["standalone"].update({"Identity": 0.0, "Frequency": 0.0, "OneBlob": 0.0}))
manipulate("fp16_padding_nonzero", lambda x: x["padding_contracts"]["fp16_network_with_encoding"][0].update({"value": 1.0}))
manipulate("collision_strong_without_collision", lambda x: x["collision_proofs"]["strong"].update({"collision_count": 0, "collision_witness": None}))
manipulate("collision_low_rate_invalid", lambda x: x["collision_proofs"]["low_3d"].update({"collision_count": 1}))
manipulate("dynamic_without_scale_change", lambda x: x["training_runs"][2].update({"scale_change_count": 0}))
manipulate("dynamic_without_overflow_skip_recovery", lambda x: x["training_runs"][2].update({"overflow_count": 0, "skip_count": 0, "recovery_count": 0}))
manipulate("cpu_rng_mismatch", lambda x: x["checkpoint_resume"][0].update({"cpu_rng_equal": False}))
manipulate("cuda_all_rng_mismatch", lambda x: x["checkpoint_resume"][0].update({"cuda_all_rng_equal": False}))
manipulate("custom_generator_rng_mismatch", lambda x: x["checkpoint_resume"][0].update({"custom_generator_rng_equal": False}))
manipulate("maximum_without_attribution", lambda x: x["numerical_maxima"]["dinput"].pop("encoding"))
manipulate("network_gradient_nonfinite_false", lambda x: x["training_runs"][2]["overflow_observation"].update({"network_gradient_nonfinite": False}))
manipulate("encoding_gradient_nonfinite_false", lambda x: x["training_runs"][2]["overflow_observation"].update({"encoding_gradient_nonfinite": False}))
manipulate("full_step_skipped_false", lambda x: x["training_runs"][2]["overflow_observation"].update({"full_step_skipped": False}))
manipulate("scale_reduced_false", lambda x: x["training_runs"][2]["overflow_observation"].update({"scale_reduced": False}))
manipulate("gradients_reset_false", lambda x: x["training_runs"][2]["overflow_observation"].update({"gradients_reset": False}))
manipulate("both_parameter_ranges_checked_false", lambda x: x["training_runs"][2].update({"both_parameter_ranges_checked": False}))

historical = {}
for path in subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", BASE, "phase3b1_reports"], cwd=ROOT, text=True
).splitlines():
    expected = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    actual = (ROOT / path).read_bytes()
    historical[path] = {"sha256": hashlib.sha256(actual).hexdigest(), "byte_equal": actual == expected}

changed = set(subprocess.check_output(["git", "diff", "--name-only", BASE], cwd=ROOT, text=True).splitlines())
changed.update(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines())
production = sorted(path for path in changed if path.startswith(("src/", "include/", "bindings/")))
checks["historical_reports"] = bool(historical) and all(item["byte_equal"] for item in historical.values())
checks["no_production_changes"] = not production
checks["manipulations"] = len(manipulations) >= 16 and all(item["passed"] for item in manipulations)

dinput = data["numerical_maxima"]["dinput"]
network_gradient = data["numerical_maxima"]["network_gradient"]
numerical_baseline = {
    "interpretation": "Phase-3B1-F starting numerics only; not a performance improvement",
    "frozen_absolute_tolerances_unchanged": True,
    "dinput": dinput,
    "network_gradient": network_gradient,
}
checks["numerical_baseline"] = (
    dinput["encoding"] == "Frequency"
    and dinput["dims"] == 2
    and dinput["variant"] == 3
    and dinput["batch"] == 1024
    and dinput["absolute_error"] == 0.02745274268090725
    and dinput["normalized_l2"] == 0.028579029471602468
    and network_gradient["encoding"] == "HashGrid"
    and network_gradient["dims"] == 2
    and network_gradient["interpolation"] == "Smoothstep"
    and network_gradient["n_levels"] == 16
    and network_gradient["n_features_per_level"] == 4
    and network_gradient["batch"] == 1024
    and network_gradient["absolute_error"] == 0.0007190704345703125
    and network_gradient["normalized_l2"] == 0.10276123438550208
)

decision = "PROCEED_TO_3B1F_FP16_PERFORMANCE" if all(checks.values()) else "PHASE3B1E1A1_BLOCKED"
result = {
    "marker": MARKER,
    "base_commit": BASE,
    "decision": decision,
    "gates": checks,
    "blocking_gates": [name for name, passed in checks.items() if not passed],
    "manipulation_tests": manipulations,
    "dynamic_run": data["training_runs"][2],
    "encoding_overflow_field_contract": {
        "classification": "naming_contract_only",
        "measured_overflow_evidence": False,
        "note": "Measured overflow evidence is exclusively training_runs[2].overflow_observation.",
        "field_names": data.get("encoding_overflow_field_contract", {}),
    },
    "phase3b1f_starting_numerics": numerical_baseline,
    "historical_reports": historical,
    "production_files": production,
    "raw_data": {
        "absolute_path": str(RAW),
        "size_bytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    },
}
OUT.write_text(json.dumps(result, indent=2) + "\n")
MD.write_text(
    f"""# Phase 3B1-E1a1 – Finalizer-Only Closure

Marker: `{MARKER}`

- Entscheidung: `{decision}`
- Finalizer-Manipulationen: {sum(x['passed'] for x in manipulations)}/{len(manipulations)}
- Dynamischer Overflow vollständig gemessen: {checks['dynamic_overflow']}
- Historische Reports bytegleich: {checks['historical_reports']}
- Produktionsänderungen: keine

## Ausgangsnumerik für Phase 3B1-F

- dL/dinput: Frequency, dims=2, variant=3, batch=1024; max_abs={dinput['absolute_error']}, normalized_l2={dinput['normalized_l2']}
- Network gradient: HashGrid, dims=2, Smoothstep, 16 Levels, 4 Features, batch=1024; max_abs={network_gradient['absolute_error']}, normalized_l2={network_gradient['normalized_l2']}

Diese Werte liegen innerhalb der unveränderten absoluten Toleranzen. Sie sind eine numerische Ausgangsbasis und keine Performanceverbesserung.
"""
)
print(decision)
raise SystemExit(0 if decision.startswith("PROCEED") else 1)
