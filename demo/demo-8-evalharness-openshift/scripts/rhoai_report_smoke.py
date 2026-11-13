"""Exercise the report component with synthetic, non-fault test results."""

from kfp import dsl

from pipeline import generate_report, preflight_models, reset_environment


@dsl.component(
    base_image="image-registry.openshift-image-registry.svc:5000/agentic-aiops-harness/aiops-harness:latest"
)
def synthetic_cell(agent: str, detected: bool) -> dict:
    return {"approach": agent, "fault_flag": "test", "detected": detected,
            "mttd_seconds": 0 if detected else None, "rca_correct": detected,
            "remediation_executed": False, "agent_error": False}


@dsl.pipeline(name="aiops-rhoai-report-smoke")
def report_smoke_pipeline():
    first = synthetic_cell(agent="synthetic-a", detected=True).set_caching_options(False)
    second = synthetic_cell(agent="synthetic-b", detected=False).set_caching_options(False)
    third = synthetic_cell(agent="synthetic-c", detected=True).set_caching_options(False)
    fourth = synthetic_cell(agent="synthetic-d", detected=False).set_caching_options(False)
    generate_report(
        result_1=first.output,
        result_2=second.output,
        result_3=third.output,
        result_4=fourth.output,
        run_name="synthetic-smoke-not-a-harness-run",
    ).set_caching_options(False)


@dsl.pipeline(name="aiops-preflight-reset-smoke")
def preflight_reset_smoke_pipeline():
    preflight = preflight_models().set_caching_options(False)
    reset_environment().set_caching_options(False).after(preflight)
