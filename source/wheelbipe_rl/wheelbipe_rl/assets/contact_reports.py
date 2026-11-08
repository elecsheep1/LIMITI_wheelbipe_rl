def spawn_with_contact_reports(
    prim_path,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """生成机器人，并为嵌套刚体启用接触报告。"""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics, PhysxSchema

    root = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            report_api = (
                PhysxSchema.PhysxContactReportAPI.Apply(
                    prim
                )
            )
            report_api.CreateThresholdAttr().Set(0.0)

    return root
