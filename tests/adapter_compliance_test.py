from __future__ import annotations

from fastapi.testclient import TestClient

from vla_lens.capture import (
    FakeDatasetEpisodeAdapter,
    FakeEnvironmentAdapter,
    FakeModelCaptureAdapter,
    validate_lerobot_v3_dataset,
    write_fake_adapter_lerobot_dataset,
)
from vla_lens.server.fastapi_app import create_dashboard_app


def test_fake_adapters_write_lerobot_v3_overlay_without_pi05_dependencies(tmp_path):
    dataset = write_fake_adapter_lerobot_dataset(tmp_path, episode_count=2, length=4)
    validation = validate_lerobot_v3_dataset(tmp_path)

    assert validation.valid, validation.to_dict()
    assert [bundle.manifest.trace_id for bundle in dataset.bundles] == ["fake_000", "fake_001"]
    assert dataset.bundle("fake_000").actions().shape == (4, 2)
    assert dataset.bundle("fake_000").cameras() == ["main"]
    assert dataset.bundle("fake_000").array("action_chunks").shape == (2, 2, 2)
    assert set(dataset.model_site_index["name"].astype(str)) == {
        "fake.action_head.output",
        "fake.backbone.layers.0.hidden",
    }
    model_site_names = dataset.model_site_index["name"].astype(str)
    assert not any(name.startswith("pi05.") for name in model_site_names)


def test_fake_adapters_satisfy_protocol_shapes():
    dataset_adapter = FakeDatasetEpisodeAdapter(episode_count=1, length=4)
    environment_adapter = FakeEnvironmentAdapter()
    model_adapter = FakeModelCaptureAdapter()

    episode = dataset_adapter.load_episode(dataset_adapter.episode_ids()[0])
    trace = model_adapter.capture_episode(episode)

    assert dataset_adapter.descriptor.dataset_family == "fake_robot"
    assert environment_adapter.metadata()["env_family"] == "fake_env"
    assert model_adapter.capture_spec.model_family == "fake_vla"
    assert model_adapter.capture_spec.action_generator is not None
    assert len(model_adapter.capture_spec.sites) == 2
    assert trace.descriptor.model_family == "fake_vla"
    assert [site.name for site in trace.model_arrays] == [
        "fake.backbone.layers.0.hidden",
        "fake.action_head.output",
    ]
    assert trace.policy_calls[0].metadata["model_family"] == "fake_vla"


def test_fake_adapter_dataset_capabilities_are_generic_not_pi05(tmp_path):
    dataset = write_fake_adapter_lerobot_dataset(tmp_path, episode_count=1, length=4)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/dataset")
    payload = response.json()

    assert response.status_code == 200
    assert payload["capabilities"]["flags"]["robot_episodes"] is True
    assert payload["capabilities"]["flags"]["cameras"] is True
    assert payload["capabilities"]["flags"]["policy_calls"] is True
    assert payload["capabilities"]["flags"]["model_sites"] is True
    assert payload["capabilities"]["flags"]["token_spaces"] is True
    assert payload["capabilities"]["flags"]["action_chunks"] is True
    assert payload["capabilities"]["flags"]["action_generation"] is True
    assert payload["capabilities"]["flags"]["attention_maps"] is False
    assert payload["capabilities"]["model_families"] == ["fake_vla"]
    assert payload["capabilities"]["model_site_prefixes"] == ["fake"]
