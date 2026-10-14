import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
PART_DIR = ROOT / "assets" / "prusa-core-one" / "parts"

# Welded part meshes. Fasteners keep about half their triangles. The CAD
# assembly (CORE_One_assembly.obj / .step) is not loaded.
PARTS = ("frame", "door", "plate")
PART_COLOR = {
    "frame": (0.72, 0.72, 0.74, 1.0),
    "door": (0.55, 0.75, 0.82, 1.0),
    "plate": (0.82, 0.62, 0.28, 1.0),
}
# Millimeter AABB of the part meshes. Same box as the original assembly.
_MESH_MIN_MM = (-207.747528, -263.121307, -64.313202)
_MESH_MAX_MM = (207.748535, 285.699707, 490.937531)


def _floor_offset(
    minimum: tuple[float, float, float], maximum: tuple[float, float, float]
) -> tuple[float, float, float]:
    # Source vertices are millimeters. 1 mm of lift keeps the feet off the
    # floor plane so they do not z-fight.
    center_x = (minimum[0] + maximum[0]) / 2
    center_y = (minimum[1] + maximum[1]) / 2
    return (-center_x / 1000, -center_y / 1000, -minimum[2] / 1000 + 0.001)


def _add_visual(body: mujoco.MjsBody, name: str) -> None:
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = name
    # Visual only. A convex hull of the shell would fill the build volume.
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 2
    geom.rgba[:] = PART_COLOR[name]


def load_model() -> mujoco.MjModel:
    missing = [name for name in PARTS if not (PART_DIR / f"{name}.obj").is_file()]
    if missing:
        raise FileNotFoundError(f"missing part meshes in {PART_DIR}: {', '.join(missing)}")

    spec = mujoco.MjSpec()
    spec.modelname = "prusa_core_one"
    spec.meshdir = str(PART_DIR)
    # Mouse spring. The red wireframe cube is this target, not a part of the printer.
    spec.visual.map.stiffness = 400
    spec.visual.map.stiffnessrot = 2000
    for name in PARTS:
        mesh = spec.add_mesh()
        mesh.name = name
        mesh.file = f"{name}.obj"
        mesh.scale[:] = [0.001, 0.001, 0.001]

    key = spec.worldbody.add_light()
    key.name = "key"
    key.pos[:] = [0.8, -1.0, 1.2]
    key.dir[:] = [-0.8, 1.0, -0.9]
    fill = spec.worldbody.add_light()
    fill.name = "fill"
    fill.pos[:] = [-0.7, 0.8, 0.9]
    fill.dir[:] = [0.7, -0.8, -0.6]

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[:] = [2, 2, 0.1]
    floor.rgba[:] = [0.85, 0.85, 0.85, 1]

    frame = spec.worldbody.add_body()
    frame.name = "frame"
    frame.pos[:] = _floor_offset(_MESH_MIN_MM, _MESH_MAX_MM)
    _add_visual(frame, "frame")

    # The plate stays welded. The door swings on the real pin, so a mouse
    # drag has a joint it can move. Both hinge leaves are still in this mesh.
    door = frame.add_body()
    door.name = "door"
    hinge = door.add_joint()
    hinge.name = "door_hinge"
    hinge.type = mujoco.mjtJoint.mjJNT_HINGE
    hinge.pos[:] = [-0.1805, -0.2067, 0.232]
    hinge.axis[:] = [0, 0, 1]
    hinge.range[:] = [-110, 0]
    hinge.damping[:] = 0.2
    # Light explicit mass. The mesh convex hull would make the mouse spring too weak.
    door.mass = 0.4
    door.ipos[:] = [-0.07, -0.208, 0.23]
    door.inertia[:] = [0.02, 0.015, 0.008]
    door.explicitinertial = True
    _add_visual(door, "door")
    plate = frame.add_body()
    plate.name = "plate"
    _add_visual(plate, "plate")
    model = spec.compile()
    for index, name in enumerate(PARTS):
        print(f"{name:8s} {model.mesh_facenum[index]:8d} faces  {model.mesh_vertnum[index]:8d} vertices")
    return model


def main() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with viewer.lock():
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
