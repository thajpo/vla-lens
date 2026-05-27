# ruff: noqa: F403,F405
from tests._support.vla_lens_trace_mvp import *


def test_richer_trace_tables_are_cataloged_and_queryable(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)

    tables = {table.table_id: table.to_dict() for table in table_catalog(dataset)}
    generation_steps = query_table(
        dataset,
        table="generation_steps",
        filters={"generation_step": [1]},
        columns=["trace_id", "policy_call_index", "generation_step", "t"],
    )
    streams = query_table(dataset, table="streams", columns=["trace_id", "stream_id", "modality"])
    token_spaces = query_table(
        dataset,
        table="token_spaces",
        columns=["trace_id", "token_space_id", "token_count"],
    )
    context = query_table(
        dataset,
        table="context",
        filters={"context_table": ["robot_state"]},
        columns=["trace_id", "context_table", "field_name", "array_id"],
    )

    assert {"generation_steps", "streams", "token_spaces", "context"}.issubset(tables)
    assert tables["context"]["provenance"]["context_tables"]
    assert generation_steps["rows"][0]["generation_step"] == 1
    assert any(row["stream_id"] == "synthetic.action" for row in streams["rows"])
    assert any(row["token_space_id"] == "synthetic.action_suffix" for row in token_spaces["rows"])
    assert context["rows"][0]["context_table"] == "robot_state"


def test_model_site_catalog_exposes_richer_schema_fields(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    bundle = dataset.bundles[0]
    assert bundle.overlay_bundle is not None
    table_path = bundle.overlay_bundle.path / TraceBundle.MODEL_SITES
    sites = pd.read_parquet(table_path)
    sites.loc[0, "family"] = "expert"
    sites.loc[0, "role"] = "action_generation"
    sites.loc[0, "segment"] = "action_head"
    sites.loc[0, "materialization"] = "summary"
    sites.loc[0, "exactness"] = "approximate"
    sites.loc[0, "token_space_id"] = "action_tokens"
    sites.loc[0, "query_token_space_id"] = "action_queries"
    sites.loc[0, "key_token_space_id"] = "vlm_keys"
    sites.loc[0, "parent_site_id"] = "expert.parent"
    sites.loc[0, "summary_type"] = "mean"
    sites.to_parquet(table_path, index=False)
    manifest = workbench_manifest(TraceDataset.open(dataset.root))
    site = next(item for item in manifest["model_sites"] if item["family"] == "expert")

    assert site["site_type"] == "action_generation"
    assert site["segment"] == "action_head"
    assert site["materialization"] == "summary"
    assert site["exactness"] == "approximate"
    assert site["token_space_id"] == "action_tokens"
    assert site["refs"]["query_token_space_id"] == "action_queries"
    assert site["refs"]["key_token_space_id"] == "vlm_keys"
    assert site["refs"]["parent_site_id"] == "expert.parent"
    assert site["summary"]["row_count"] >= 1


def test_validation_rejects_token_space_reference_errors(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "bad_tokens",
        tokens={
            "token_space_id": ["missing_space"],
            "token_index": [0],
            "token_kind": ["action"],
        },
        token_spaces={
            "token_space_id": ["known_space"],
            "stream_id": ["action"],
            "token_count": [1],
        },
        streams={"stream_id": ["action"], "name": ["action"], "modality": ["action"]},
    )

    result = validate_trace_bundle(bundle)

    assert not result.valid
    assert any(error["code"] == "invalid_reference" for error in result.errors)


def test_validation_requires_exact_raw_full_sites(tmp_path):
    model_sites = [
        TraceModelSiteSpec(
            name=f"pi05.full.{role}",
            array=np.zeros((1,), dtype=np.float32),
            axes=["scalar"],
            module=f"pi05/{role}",
            tensor_type=role,
            role=role,
            materialization="raw",
            exactness="exact",
        )
        for role in FULL_REQUIRED_MODEL_SITE_ROLES
    ]
    role = FULL_REQUIRED_MODEL_SITE_ROLES[0]
    model_sites[0] = TraceModelSiteSpec(
        name=f"pi05.full.{role}",
        array=np.zeros((1,), dtype=np.float32),
        axes=["scalar"],
        module=f"pi05/{role}",
        tensor_type=role,
        role=role,
        materialization="summary",
        exactness="lossy_summary",
    )
    bundle = _make_minimal_trace(
        tmp_path / "partial_full",
        profile="full",
        model_sites=model_sites,
    )

    result = validate_trace_bundle(bundle)

    assert not result.valid
    assert any(error["code"] == "profile_full_missing_raw_sites" for error in result.errors)


def test_validation_accepts_complete_exact_raw_full_sites(tmp_path):
    model_sites = [
        TraceModelSiteSpec(
            name=f"pi05.full.{role}",
            array=np.zeros((1,), dtype=np.float32),
            axes=["scalar"],
            module=f"pi05/{role}",
            tensor_type=role,
            role=role,
            materialization="raw",
            exactness="exact",
        )
        for role in FULL_REQUIRED_MODEL_SITE_ROLES
    ]
    bundle = _make_minimal_trace(
        tmp_path / "complete_full",
        profile="full",
        model_sites=model_sites,
    )

    result = validate_trace_bundle(bundle)

    assert result.valid


def test_validation_accepts_audit_windowed_profile(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "audit_windowed_validation",
        profile="audit_windowed",
        model_sites=[
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.prefix.hidden_tokens",
                array=np.zeros((1, 2, 3), dtype=np.float32),
                axes=["policy_call", "token", "channel"],
                module="pi05.vlm.layers.0",
                layer=0,
                tensor_type="hidden_tokens",
            ),
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.prefix.attention",
                array=np.zeros((1, 1, 2, 2), dtype=np.float32),
                axes=["policy_call", "head", "query_token", "key_token"],
                module="pi05.vlm.layers.0.attention",
                layer=0,
                tensor_type="attention",
            ),
        ],
    )

    result = validate_trace_bundle(bundle)

    assert result.valid
    assert not any(warning["code"] == "unknown_capture_profile" for warning in result.warnings)


def test_activation_sites_payload_includes_runtime_kv_collection(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "kv_collection",
        model_sites=[
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.kv_cache.key",
                array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                module="pi05.vlm.layers.0.attention",
                layer=0,
                tensor_type="kv_cache",
                token_kind="prefix",
                family="cache",
                role="kv_cache_key",
                token_space_id="pi05.prefix",
            ),
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.kv_cache.value",
                array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                module="pi05.vlm.layers.0.attention",
                layer=0,
                tensor_type="kv_cache",
                token_kind="prefix",
                family="cache",
                role="kv_cache_value",
                token_space_id="pi05.prefix",
            ),
        ],
    )

    payload = _activation_sites_payload(bundle)

    assert payload["runtime_collections"][0]["id"] == "pi05.vlm.past_key_values"
    assert payload["runtime_collections"][0]["materialized"] is False
    assert payload["runtime_collections"][0]["aggregation"] == "none"
    assert {member["site_name"] for member in payload["runtime_collections"][0]["members"]} == {
        "pi05.vlm.layers.0.kv_cache.key",
        "pi05.vlm.layers.0.kv_cache.value",
    }


def test_activation_sites_payload_includes_per_layer_kv_architecture_edges(tmp_path):
    model_sites = []
    for layer in (0, 4):
        model_sites.extend(
            [
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.key",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_key",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.value",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_value",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.expert.layers.{layer}.by_step.attention",
                    array=np.zeros((1, 1, 1, 2, 4), dtype=np.float32),
                    axes=["policy_call", "generation_step", "head", "query_token", "key_token"],
                    module=f"pi05.expert.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="attention",
                    token_kind="action",
                    family="attention",
                    role="attention_probs",
                    segment="action_expert",
                    query_token_space_id="pi05.action_suffix",
                    key_token_space_id="pi05.expert_context",
                ),
            ]
        )
    bundle = _make_minimal_trace(tmp_path / "kv_architecture", model_sites=model_sites)

    payload = _activation_sites_payload(bundle)

    edges = payload["architecture"]["edges"]
    assert [edge["id"] for edge in edges] == [
        "pi05.vlm.layers.0.kv_to_expert.layers.0",
        "pi05.vlm.layers.4.kv_to_expert.layers.4",
    ]
    assert [edge["layer"] for edge in edges] == [0, 4]
    assert all(edge["kind"] == "per_layer_kv_conditioning" for edge in edges)
    assert all(edge["source_token_space"] == "pi05.prefix" for edge in edges)
    assert all(edge["query_token_space"] == "pi05.action_suffix" for edge in edges)
    assert all(edge["key_token_space"] == "pi05.expert_context" for edge in edges)
    assert all(edge["materialized"] is False for edge in edges)
    assert edges[0]["source_sites"] == [
        "pi05.vlm.layers.0.kv_cache.key",
        "pi05.vlm.layers.0.kv_cache.value",
    ]
    assert any(node["id"] == "pi05.vlm.layers.4" for node in payload["architecture"]["nodes"])
    assert any(node["id"] == "pi05.expert.layers.4" for node in payload["architecture"]["nodes"])


def test_activation_sites_payload_includes_audit_windowed_kv_edges(tmp_path):
    model_sites = []
    for layer in AUDIT_WINDOWED_LAYERS:
        model_sites.extend(
            [
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.key",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_key",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.value",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_value",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.expert.layers.{layer}.by_step.attention",
                    array=np.zeros((1, 1, 1, 2, 4), dtype=np.float32),
                    axes=["policy_call", "generation_step", "head", "query_token", "key_token"],
                    module=f"pi05.expert.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="attention",
                    token_kind="action",
                    family="attention",
                    role="attention_probs",
                    segment="action_expert",
                    query_token_space_id="pi05.action_suffix",
                    key_token_space_id="pi05.expert_context",
                ),
            ]
        )
    bundle = _make_minimal_trace(
        tmp_path / "audit_windowed_kv_architecture",
        model_sites=model_sites,
    )

    payload = _activation_sites_payload(bundle)

    members = payload["runtime_collections"][0]["members"]
    edges = payload["architecture"]["edges"]
    assert len(members) == 20
    assert {int(member["layer"]) for member in members} == set(AUDIT_WINDOWED_LAYERS)
    assert [edge["layer"] for edge in edges] == list(AUDIT_WINDOWED_LAYERS)
    assert len(edges) == 10
    assert all(edge["kind"] == "per_layer_kv_conditioning" for edge in edges)
    assert all(edge["source"].endswith(str(edge["layer"])) for edge in edges)
    assert all(edge["target"].endswith(str(edge["layer"])) for edge in edges)


def test_activation_sites_payload_keeps_empty_architecture_for_non_pi05_sites(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "generic_activation",
        model_sites=[
            TraceModelSiteSpec(
                name="toy.layers.0.hidden",
                array=np.zeros((1, 2), dtype=np.float32),
                axes=["token", "channel"],
                module="toy.layers.0",
                layer=0,
                tensor_type="hidden_tokens",
            )
        ],
    )

    payload = _activation_sites_payload(bundle)

    assert payload["architecture"] == {}


def test_expert_token_details_project_attention_to_image_and_prompt_tokens(tmp_path):
    hidden = np.array([[[[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]]]], dtype=np.float32)
    attention = np.array(
        [
            [
                [
                    [
                        [0.05, 0.10, 0.20, 0.25, 0.30, 0.05, 0.05],
                        [0.10, 0.20, 0.40, 0.10, 0.15, 0.03, 0.02],
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )
    bundle = _make_minimal_trace(
        tmp_path / "attention_details",
        include_frames=True,
        camera_state=pd.DataFrame.from_records(
            [{"camera_id": "main", "name": "main", "width": 16, "height": 16}]
        ),
        streams={
            "stream_id": ["prefix", "image_main", "language", "action_suffix"],
            "name": ["prefix", "main", "language", "action_suffix"],
            "modality": ["multimodal", "image", "language", "action"],
        },
        token_spaces={
            "token_space_id": ["pi05.prefix", "pi05.action_suffix"],
            "stream_id": ["prefix", "action_suffix"],
            "token_count": [5, 2],
        },
        tokens={
            "token_space_id": ["pi05.prefix"] * 5 + ["pi05.action_suffix"] * 2,
            "token_index": [0, 1, 2, 3, 4, 0, 1],
            "token_kind": ["image", "image", "image", "image", "language", "action", "action"],
            "token_type": ["image_patch"] * 4 + ["text", "continuous_action", "continuous_action"],
            "camera_id": ["main", "main", "main", "main", None, None, None],
            "patch_row": [0, 0, 1, 1, None, None, None],
            "patch_col": [0, 1, 0, 1, None, None, None],
            "token_id": [None, None, None, None, 42, None, None],
            "token_piece": [None, None, None, None, "cube", None, None],
            "attention_mask": [None, None, None, None, True, None, None],
            "policy_call_index": [0] * 7,
        },
        model_sites=[
            TraceModelSiteSpec(
                name="pi05.vlm.prefix.image_hidden_tokens",
                array=np.zeros((1, 4, 3), dtype=np.float32),
                axes=["policy_call", "token", "channel"],
                module="pi05.vlm.prefix",
                tensor_type="hidden_tokens",
                token_kind="image",
                family="representation",
                role="hidden_state",
                token_space_id="pi05.prefix",
                metadata={"patches_per_image": 4, "image_slots": 1, "grid_size": 2},
            ),
            TraceModelSiteSpec(
                name="pi05.expert.layers.0.by_step.hidden_tokens",
                array=hidden,
                axes=["policy_call", "generation_step", "token", "channel"],
                module="pi05.expert.layers.0",
                layer=0,
                tensor_type="hidden_tokens",
                token_kind="action",
                family="representation",
                role="hidden_state",
                token_space_id="pi05.action_suffix",
            ),
            TraceModelSiteSpec(
                name="pi05.expert.layers.0.attention.attention_probs",
                array=attention,
                axes=["policy_call", "generation_step", "head", "query_token", "key_token"],
                module="pi05.expert.layers.0",
                layer=0,
                tensor_type="attention_probs",
                family="attention",
                role="attention_probs",
                query_token_space_id="pi05.action_suffix",
                key_token_space_id="pi05.prefix",
            ),
        ],
    )

    details = _expert_token_details_payload(
        bundle,
        {
            "name": ["pi05.expert.layers.0.by_step.hidden_tokens"],
            "call_index": ["0"],
            "generation_step": ["0"],
            "token_index": ["1"],
            "feature": ["2"],
        },
    )
    prompt = _prompt_attention_payload(bundle, {"call_index": ["0"], "generation_step": ["0"]})

    assert details["available"] is True
    assert details["attention_site"] == "pi05.expert.layers.0.attention.attention_probs"
    assert np.isclose(details["attention_coarse"]["image"], 0.8)
    assert np.isclose(details["attention_coarse"]["prompt"], 0.15)
    assert np.isclose(details["attention_coarse"]["action_suffix"], 0.05)
    assert np.allclose(details["maps"]["main"]["values"], [[0.1, 0.2], [0.4, 0.1]])
    assert details["top_image_patches"][0]["camera"] == "main"
    assert details["top_image_patches"][0]["row"] == 1
    assert details["top_image_patches"][0]["col"] == 0
    assert details["top_image_patches"][0]["token_index"] == 2
    assert np.isclose(details["top_image_patches"][0]["attention"], 0.4)
    assert details["top_prompt_tokens"][0]["token_piece"] == "cube"
    assert np.isclose(details["top_prompt_tokens"][0]["attention"], 0.15)
    assert prompt["available"] is True
    assert np.isclose(prompt["expert_coarse"]["prompt"], 0.225)
    selected_map = _attention_map_payload(
        bundle,
        {
            "kind": ["expert"],
            "call_index": ["0"],
            "generation_step": ["0"],
            "head": ["0"],
            "query_token": ["1"],
        },
    )
    averaged_map = _attention_map_payload(
        bundle,
        {"kind": ["expert"], "call_index": ["0"], "generation_step": ["0"]},
    )
    assert selected_map["available"] is True
    assert selected_map["head_mode"] == "selected"
    assert selected_map["query_mode"] == "selected"
    assert selected_map["head"] == 0
    assert selected_map["query_token"] == 1
    assert np.allclose(selected_map["maps"]["main"]["values"], [[0.1, 0.2], [0.4, 0.1]])
    assert averaged_map["query_mode"] == "average"
    assert np.allclose(averaged_map["maps"]["main"]["values"], [[0.075, 0.15], [0.3, 0.175]])


def test_lens_array_dims_exist_in_axis_registry(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    manifest = workbench_manifest(dataset)
    axes = set(manifest["axes"])

    for array in manifest["lens_arrays"]:
        assert set(array["dims"]).issubset(axes)
