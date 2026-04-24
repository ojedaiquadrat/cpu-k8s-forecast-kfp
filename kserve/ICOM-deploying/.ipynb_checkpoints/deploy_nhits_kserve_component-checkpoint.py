from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import kserve
from kserve import KServeClient
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from mlflow import MlflowException
from mlflow.tracking import MlflowClient


LOGGER = logging.getLogger("deploy_nhits_kserve")


@dataclass
class DeploymentResult:
    model_name: str
    model_alias: str
    service_name: str
    namespace: str
    runtime: str
    run_id: str
    experiment_id: str
    model_version: str
    storage_uri: str
    status: str
    url: Optional[str]


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )


def _load_kube_config() -> None:
    try:
        k8s_config.load_incluster_config()
        LOGGER.info("Loaded in-cluster Kubernetes configuration.")
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
        LOGGER.info("Loaded local kubeconfig.")


def _sanitize_k8s_name(value: str) -> str:
    name = value.lower().strip()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    if not name:
        raise ValueError("The generated InferenceService name is empty.")
    return name[:63].rstrip("-")


def _build_storage_uri(
    experiment_id: str,
    run_id: str,
    s3_prefix: str = "s3://mlflow",
    model_subpath: str = "model",
) -> str:
    prefix = s3_prefix.rstrip("/")
    model_path = model_subpath.strip("/")
    return f"{prefix}/{experiment_id}/{run_id}/artifacts/{model_path}"


def _resolve_model_version(
    mlflow_tracking_uri: str,
    registered_model_name: str,
    model_alias: str,
) -> Dict[str, Any]:
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    try:
        model_version = client.get_model_version_by_alias(
            name=registered_model_name,
            alias=model_alias,
        )
    except MlflowException as exc:
        raise RuntimeError(
            f"Unable to resolve model '{registered_model_name}' with alias '{model_alias}'."
        ) from exc

    if not model_version.run_id:
        run_id_from_tags = getattr(model_version, "tags", {}).get("run_id")
        if run_id_from_tags:
            model_version.run_id = run_id_from_tags

    if not model_version.run_id and getattr(model_version, "source", None):
        match = re.search(r"/([0-9a-f]{32})/artifacts", model_version.source)
        if match:
            model_version.run_id = match.group(1)

    if not model_version.run_id:
        raise RuntimeError(
            f"Model '{registered_model_name}' alias '{model_alias}' does not have a run_id."
        )

    run = client.get_run(model_version.run_id)
    experiment_id = run.info.experiment_id

    return {
        "model_version": model_version,
        "run_id": model_version.run_id,
        "experiment_id": experiment_id,
    }


def _build_inferenceservice(
    service_name: str,
    namespace: str,
    storage_uri: str,
    runtime_name: str,
    service_account_name: str,
    min_replicas: int,
    max_replicas: int,
    cpu_request: str,
    cpu_limit: str,
    memory_request: str,
    memory_limit: str,
) -> kserve.V1beta1InferenceService:
    predictor = kserve.V1beta1PredictorSpec(
        service_account_name=service_account_name,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        model=kserve.V1beta1ModelSpec(
            name=service_name,
            model_format={"name": "mlflow"},
            runtime=runtime_name,
            storage_uri=storage_uri,
            resources=k8s_client.V1ResourceRequirements(
                requests={"cpu": cpu_request, "memory": memory_request},
                limits={"cpu": cpu_limit, "memory": memory_limit},
            ),
        ),
    )

    return kserve.V1beta1InferenceService(
        api_version="serving.kserve.io/v1beta1",
        kind="InferenceService",
        metadata=k8s_client.V1ObjectMeta(name=service_name, namespace=namespace),
        spec=kserve.V1beta1InferenceServiceSpec(predictor=predictor),
    )


def deploy_nhits_model(
    registered_model_name: str = "nhits-model",
    model_alias: str = "champion",
    service_name: str = "nhits-model",
    namespace: str = "lstm-iqu",
    mlflow_tracking_uri: str = "http://sunrise-mlflow-tracking.mlflow.svc.cluster.local:5080",
    runtime_name: str = "kserve-mlserver-nhits",
    service_account_name: str = "sa-private-mlflow",
    s3_prefix: str = "s3://mlflow",
    model_subpath: str = "model",
    min_replicas: int = 1,
    max_replicas: int = 1,
    cpu_request: str = "500m",
    cpu_limit: str = "2",
    memory_request: str = "2Gi",
    memory_limit: str = "6Gi",
    timeout_seconds: int = 900,
) -> Dict[str, Any]:
    """
    Resolve the champion alias for the NHITS model in MLflow and deploy it as a KServe InferenceService.

    This function is intended to run from a Kubeflow notebook or Python component container that can:
    - reach the MLflow tracking service
    - authenticate to the Kubernetes API
    - access the target namespace `lstm-iqu`
    """
    _configure_logging()
    _load_kube_config()

    service_name = _sanitize_k8s_name(service_name)
    LOGGER.info(
        "Resolving MLflow model alias '%s' for model '%s'.",
        model_alias,
        registered_model_name,
    )

    try:
        resolved = _resolve_model_version(
            mlflow_tracking_uri=mlflow_tracking_uri,
            registered_model_name=registered_model_name,
            model_alias=model_alias,
        )
        storage_uri = _build_storage_uri(
            experiment_id=resolved["experiment_id"],
            run_id=resolved["run_id"],
            s3_prefix=s3_prefix,
            model_subpath=model_subpath,
        )
    except Exception as exc:
        LOGGER.exception("Failed while resolving MLflow model metadata.")
        raise RuntimeError(
            f"Model resolution failed for '{registered_model_name}@{model_alias}': {exc}"
        ) from exc

    LOGGER.info("Resolved run_id=%s storage_uri=%s", resolved["run_id"], storage_uri)

    isvc = _build_inferenceservice(
        service_name=service_name,
        namespace=namespace,
        storage_uri=storage_uri,
        runtime_name=runtime_name,
        service_account_name=service_account_name,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        memory_request=memory_request,
        memory_limit=memory_limit,
    )

    kserve_client = KServeClient()

    try:
        kserve_client.get(service_name, namespace=namespace)
        LOGGER.info(
            "InferenceService '%s' already exists in namespace '%s'; patching it.",
            service_name,
            namespace,
        )
        kserve_client.patch(service_name, isvc, namespace=namespace)
    except Exception:
        LOGGER.info(
            "InferenceService '%s' does not exist yet in namespace '%s'; creating it.",
            service_name,
            namespace,
        )
        try:
            kserve_client.create(isvc, namespace=namespace)
        except Exception as exc:
            LOGGER.exception("KServe create operation failed.")
            raise RuntimeError(
                f"Failed to create InferenceService '{service_name}': {exc}"
            ) from exc
    try:
        LOGGER.info(
            "Waiting up to %s seconds for InferenceService '%s' to become Ready.",
            timeout_seconds,
            service_name,
        )
        kserve_client.wait_isvc_ready(
            service_name,
            namespace=namespace,
            timeout_seconds=timeout_seconds,
        )
        final_isvc = kserve_client.get(service_name, namespace=namespace)
    except Exception as exc:
        LOGGER.exception("InferenceService did not become ready.")
        raise RuntimeError(
            f"InferenceService '{service_name}' was created but did not become Ready: {exc}"
        ) from exc

    final_status = getattr(final_isvc, "status", None)
    final_url = getattr(final_status, "url", None) if final_status else None

    result = DeploymentResult(
        model_name=registered_model_name,
        model_alias=model_alias,
        service_name=service_name,
        namespace=namespace,
        runtime=runtime_name,
        run_id=resolved["run_id"],
        experiment_id=resolved["experiment_id"],
        model_version=str(resolved["model_version"].version),
        storage_uri=storage_uri,
        status="Ready",
        url=final_url,
    )

    LOGGER.info("Deployment completed successfully: %s", json.dumps(asdict(result), indent=2))
    return asdict(result)


if __name__ == "__main__":
    deployment = deploy_nhits_model()
    print(json.dumps(deployment, indent=2))
