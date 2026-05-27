from dataset import SOCIAL_CALENDAR_CKPT_BASENAME


def test_resolve_social_calendar_table_path_prefers_ckpt_sidecar(tmp_path):
    from infer import resolve_social_calendar_table_path

    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    sidecar = model_dir / SOCIAL_CALENDAR_CKPT_BASENAME
    sidecar.write_text("date,holiday_type,promo_id\n2024-01-01,0,0\n", encoding="utf-8")
    path = resolve_social_calendar_table_path({}, str(model_dir), str(tmp_path))
    assert path == str(sidecar)


def test_infer_main_builds_dataset_after_resolve_model_cfg():
    import inspect
    from infer import main

    src = inspect.getsource(main)
    idx_cfg = src.find("resolve_model_cfg")
    idx_ds = src.find("PCVRParquetDataset(")
    assert idx_cfg != -1 and idx_ds != -1 and idx_cfg < idx_ds
