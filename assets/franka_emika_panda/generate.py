"""Derive HandUMI's URDF and meter-controlled MJCF from Menagerie's panda.xml.

Run with the project's Python environment. No meshes are regenerated. The
compiled model supplies inertias/limits; visual and collision origins stay in
the source mesh coordinates (MuJoCo recenters meshes internally).
"""

# mujoco ships no py.typed/stubs (its classes come from native bindings that
# pyright cannot introspect), so attribute checks are meaningless here.
# pyright: reportAttributeAccessIssue=false

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def numbers(values):
    return " ".join(f"{v:.12g}" for v in values)


def origin(parent, position, quaternion=(1, 0, 0, 0)):
    w, x, y, z = quaternion
    rpy = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
    ET.SubElement(parent, "origin", xyz=numbers(position), rpy=numbers(rpy))


def generate():
    source = ET.parse(ROOT / "panda.xml")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "panda.xml"))
    robot = ET.Element("robot", name="panda")
    ET.SubElement(robot, "link", name="world")
    meshes = {
        mesh.get("name", Path(mesh.attrib["file"]).stem): mesh.attrib["file"]
        for mesh in source.findall("./asset/mesh")
    }
    for material in source.findall("./asset/material"):
        node = ET.SubElement(robot, "material", name=material.attrib["name"])
        ET.SubElement(node, "color", rgba=material.attrib["rgba"])
    defaults = {}

    def collect_defaults(node, inherited):
        values = dict(inherited)
        geom = node.find("geom")
        if geom is not None:
            values.update(geom.attrib)
        defaults[node.get("class", "")] = values
        for child in node.findall("default"):
            collect_defaults(child, values)

    collect_defaults(source.getroot().find("default"), {})

    def add_body(body, parent):
        name = body.attrib["name"]
        link_name = f"right_{name}"
        link = ET.SubElement(robot, "link", name=link_name)
        bid = model.body(name).id
        inertial = ET.SubElement(link, "inertial")
        origin(inertial, model.body_ipos[bid], model.body_iquat[bid])
        ET.SubElement(inertial, "mass", value=str(model.body_mass[bid]))
        ix, iy, iz = model.body_inertia[bid]
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(ix),
            iyy=str(iy),
            izz=str(iz),
            ixy="0",
            ixz="0",
            iyz="0",
        )
        for geom in body.findall("geom"):
            attrs = defaults[geom.get("class", "panda")] | geom.attrib
            tag = "visual" if attrs.get("group") == "2" else "collision"
            node = ET.SubElement(link, tag)
            origin(
                node,
                np.fromstring(attrs.get("pos", "0 0 0"), sep=" "),
                np.fromstring(attrs.get("quat", "1 0 0 0"), sep=" "),
            )
            geometry = ET.SubElement(node, "geometry")
            if attrs.get("type") == "box":
                size = 2 * np.fromstring(attrs["size"], sep=" ")
                ET.SubElement(geometry, "box", size=numbers(size))
            else:
                ET.SubElement(
                    geometry, "mesh", filename=f"assets/{meshes[attrs['mesh']]}"
                )
            if tag == "visual" and attrs.get("material"):
                ET.SubElement(node, "material", name=attrs["material"])
        raw_joint = body.find("joint")
        if raw_joint is None:
            joint_name = f"{link_name}_fixed"
            jid = None
            kind = "fixed"
        else:
            joint_name = f"right_{raw_joint.attrib['name']}"
            jid = model.joint(raw_joint.attrib["name"]).id
            kind = (
                "prismatic"
                if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
                else "revolute"
            )
        joint = ET.SubElement(robot, "joint", name=joint_name, type=kind)
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=link_name)
        origin(joint, model.body_pos[bid], model.body_quat[bid])
        if raw_joint is not None and jid is not None:
            assert np.allclose(model.jnt_pos[jid], 0), (
                "Nonzero joint anchor needs a split link"
            )
            ET.SubElement(joint, "axis", xyz=numbers(model.jnt_axis[jid]))
            lo, hi = model.jnt_range[jid]
            # Menagerie supplies force bounds, but no velocity limits. These
            # conservative simulation limits are not a physical control spec.
            finger = kind == "prismatic"
            effort = (
                100
                if finger
                else (
                    87
                    if raw_joint.attrib["name"]
                    in {"joint1", "joint2", "joint3", "joint4"}
                    else 12
                )
            )
            ET.SubElement(
                joint,
                "limit",
                lower=str(lo),
                upper=str(hi),
                effort=str(effort),
                velocity="0.2" if finger else "2.0",
            )
            if raw_joint.attrib["name"] == "finger_joint2":
                ET.SubElement(
                    joint,
                    "mimic",
                    joint="right_finger_joint1",
                    multiplier="1",
                    offset="0",
                )
        for child in body.findall("body"):
            add_body(child, link_name)

    for body in source.findall("./worldbody/body"):
        add_body(body, "world")
    ET.SubElement(robot, "link", name="right_tcp")
    tcp = ET.SubElement(robot, "joint", name="right_tcp_fixed", type="fixed")
    ET.SubElement(tcp, "parent", link="right_hand")
    ET.SubElement(tcp, "child", link="right_tcp")
    # Midpoint of the opposing fingertip pads: 0.0584 + 0.0445 m.
    origin(tcp, [0, 0, 0.1029])
    ET.indent(robot)
    ET.ElementTree(robot).write(
        ROOT / "panda.urdf", encoding="unicode", xml_declaration=True
    )

    # Keep the upstream MJCF untouched. Adapt only actuator naming and the
    # gripper's control units for HandUMI's common physics interface.
    for actuator in source.findall("./actuator/general"):
        if "joint" in actuator.attrib:
            actuator.set("name", actuator.attrib["joint"])
        else:
            actuator.set("name", "finger_joint1")
            actuator.set("ctrlrange", "0 0.04")
            actuator.set("gainprm", "100 0 0")
    key = source.find("./keyframe/key")
    assert key is not None, "panda.xml lost its home keyframe"
    key.set("ctrl", "0 0 0 -1.57079 0 1.57079 -0.7853 0.04")
    hand = source.find(".//body[@name='hand']")
    assert hand is not None, "panda.xml lost its hand body"
    ET.SubElement(hand, "site", name="right_tcp", pos="0 0 0.1029", size="0.003")
    ET.indent(source)
    source.write(ROOT / "panda_handumi.xml", encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    generate()
