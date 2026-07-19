from handumi.dataset.canonical import canonical_joint_layout
from handumi.robots.registry import load_embodiment


def test_piper_layout_is_six_arm_joints_plus_one_gripper_per_side():
    layout = canonical_joint_layout(load_embodiment("piper"))

    assert len(layout.names) == 14
    assert layout.names[:7] == [
        "left_joint1.pos",
        "left_joint2.pos",
        "left_joint3.pos",
        "left_joint4.pos",
        "left_joint5.pos",
        "left_joint6.pos",
        "left_gripper.width_m",
    ]
    assert layout.names[13] == "right_gripper.width_m"


def test_two_finger_grippers_count_as_one_logical_width_per_side():
    for embodiment in ("trlc_dk1", "yam"):
        layout = canonical_joint_layout(load_embodiment(embodiment))

        assert len(layout.names) == 14
        assert layout.names.count("left_gripper.width_m") == 1
        assert layout.names.count("right_gripper.width_m") == 1


def test_embodiment_without_declared_gripper_keeps_yaml_arm_joints():
    layout = canonical_joint_layout(load_embodiment("axol"))

    assert len(layout.names) == 14
    assert "left_gripper.width_m" not in layout.names
