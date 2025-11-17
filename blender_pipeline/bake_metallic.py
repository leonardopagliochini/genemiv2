import os
import shutil
from pathlib import Path

import addon_utils
import bpy
import numpy as np


PROJECT_NAME = "BrainTumor"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SHADERS_DIR = SCRIPT_DIR / "Shaders"
OUTPUT_BASE_DIR = SCRIPT_DIR / "Output"
DEFAULT_TIMESTEP = int(os.environ.get("BRAIN_TIMESTEP", 108))
DEFAULT_SURFACES_DIR = REPO_ROOT / "build" / f"extracted_{DEFAULT_TIMESTEP:03d}"
DEFAULT_TUMOR_SHADER = SHADERS_DIR / "badguy_2.blend"
DEFAULT_TUMOR_SHADER_MAT = os.environ.get("BRAIN_TUMOR_SHADER_MAT", "badguy_2_mat")
TEXTURE_SIZE = int(os.environ.get("BRAIN_TEXTURE_SIZE", 2048))
TUMOR_REMESH_VOXEL_SIZE = float(os.environ.get("BRAIN_TUMOR_VOXEL_SIZE", 1.5))
TUMOR_REMESH_SMOOTH = os.environ.get("BRAIN_TUMOR_REMESH_SMOOTH", "true").lower() not in {"0", "false", "no"}

TEXTURES_DIR: Path | None = None
OUTPUT_DIR: Path | None = None
UNITY_DIR: Path | None = None
SESSION_LABEL: str | None = None


def ensure_object_mode() -> None:
    obj = bpy.context.view_layer.objects.active
    if obj and obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def clear_scene() -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except RuntimeError:
        pass


def ensure_stl_importer_enabled() -> None:
    try:
        if not bpy.context.preferences.addons.get("io_mesh_stl"):
            addon_utils.enable("io_mesh_stl")
    except ModuleNotFoundError:
        print("STL add-on module not available; relying on built-in importer")


def _call_stl_import_operator(filepath: Path) -> None:
    if hasattr(bpy.ops.import_mesh, "stl"):
        try:
            bpy.ops.import_mesh.stl(filepath=str(filepath))
            return
        except AttributeError:
            print("bpy.ops.import_mesh.stl missing; trying wm.stl_import")
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(filepath))
        return
    raise RuntimeError("No STL import operator available in this Blender build")


def ensure_bake_lighting() -> None:
    if any(obj.type == 'LIGHT' for obj in bpy.data.objects):
        return
    ensure_object_mode()
    bpy.ops.object.light_add(type='SUN')
    light = bpy.context.object
    light.name = "BakeLight"
    light.data.energy = 5.0
    light.rotation_euler = (0.7, 0.2, 1.0)
    print("Added temporary sun light for baking")


def resolve_surfaces_dir() -> Path:
    env_dir = os.environ.get("BRAIN_SURFACES_DIR")
    candidate = Path(env_dir).expanduser() if env_dir else DEFAULT_SURFACES_DIR
    candidate = candidate.resolve()
    if (candidate / "surfaces").is_dir():
        candidate = candidate / "surfaces"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Directory with STL surfaces not found: {candidate}")
    return candidate


def configure_output_dirs(surfaces_dir: Path) -> None:
    global OUTPUT_DIR, TEXTURES_DIR, UNITY_DIR, SESSION_LABEL
    container = surfaces_dir.parent if surfaces_dir.name == "surfaces" else surfaces_dir
    SESSION_LABEL = container.name
    OUTPUT_DIR = OUTPUT_BASE_DIR / f"Brain_{SESSION_LABEL}"
    TEXTURES_DIR = OUTPUT_DIR / "Textures"
    UNITY_DIR = OUTPUT_DIR / f"{PROJECT_NAME}_{SESSION_LABEL}_unity"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    UNITY_DIR.mkdir(parents=True, exist_ok=True)


def discover_tumor_segments(surfaces_dir: Path) -> list[Path]:
    stl_files = sorted(surfaces_dir.glob("*.stl"))
    tumor_paths = [f for f in stl_files if "cancer" in f.stem.lower() and "no_cancer" not in f.stem.lower()]
    if not tumor_paths:
        raise FileNotFoundError(f"No tumor STL files found inside {surfaces_dir}")
    print("Tumor STLs:", ", ".join(path.name for path in tumor_paths))
    return tumor_paths


def import_and_join_tumor(stl_paths: list[Path]) -> bpy.types.Object:
    ensure_object_mode()
    imported = []
    for path in stl_paths:
        _call_stl_import_operator(path)
        obj = bpy.context.selected_objects[0]
        imported.append(obj)
        print(f"Imported tumor mesh: {path.name}")

    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    if len(imported) > 1:
        bpy.ops.object.join()
        print(f"Joined {len(imported)} tumor meshes into one object")
    joined = bpy.context.view_layer.objects.active
    joined.name = "Tumor"
    return joined


def apply_voxel_remesh(obj: bpy.types.Object) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    remesh = obj.modifiers.new(name="TumorRemesh", type='REMESH')
    remesh.mode = 'VOXEL'
    remesh.voxel_size = max(0.1, TUMOR_REMESH_VOXEL_SIZE)
    remesh.use_smooth_shade = TUMOR_REMESH_SMOOTH
    remesh.use_remove_disconnected = True
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    if TUMOR_REMESH_SMOOTH:
        bpy.ops.object.shade_smooth()
    obj.select_set(False)
    print(f"Applied voxel remesh to {obj.name}")


def append_tumor_material(material_name: str = "TumorShader") -> bpy.types.Material:
    if material_name in bpy.data.materials:
        return bpy.data.materials[material_name]
    if not DEFAULT_TUMOR_SHADER.is_file():
        raise FileNotFoundError(f"Tumor shader .blend not found: {DEFAULT_TUMOR_SHADER}")

    with bpy.data.libraries.load(str(DEFAULT_TUMOR_SHADER), link=False) as (data_from, data_to):
        if not data_from.materials:
            raise RuntimeError(f"No materials inside {DEFAULT_TUMOR_SHADER}")
        if DEFAULT_TUMOR_SHADER_MAT not in data_from.materials:
            raise RuntimeError(
                f"Material '{DEFAULT_TUMOR_SHADER_MAT}' not found in {DEFAULT_TUMOR_SHADER}."
                f" Available: {data_from.materials}"
            )
        data_to.materials = [DEFAULT_TUMOR_SHADER_MAT]

    mat = bpy.data.materials.get(DEFAULT_TUMOR_SHADER_MAT)
    if not mat:
        raise RuntimeError("Failed to append tumor material from blend file")
    mat.name = material_name
    return mat


def apply_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)
    print(f"Applied material {material.name} to {obj.name}")


def uv_map_object(obj: bpy.types.Object) -> None:
    ensure_object_mode()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"UV map created for {obj.name}")


def bake_setup() -> None:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        device["use"] = 1
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'


def _ensure_textures_dir() -> Path:
    if TEXTURES_DIR is None:
        raise RuntimeError("TEXTURES_DIR not configured")
    return TEXTURES_DIR


def _bake_image(obj: bpy.types.Object, name: str, colorspace: str, bake_type: str, **kwargs) -> Path:
    textures_dir = _ensure_textures_dir()
    image = bpy.data.images.new(name=name, width=TEXTURE_SIZE, height=TEXTURE_SIZE)
    image.colorspace_settings.name = colorspace
    if not obj.data.materials:
        raise RuntimeError(f"Object {obj.name} has no material assigned for baking")
    mat = obj.data.materials[0]
    if mat.node_tree is None:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    tex_node.name = name
    nodes.active = tex_node
    tex_node.select = True
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if bake_type == 'DIFFUSE':
        bpy.ops.object.bake('EXEC_DEFAULT', type=bake_type, save_mode='EXTERNAL', pass_filter={'COLOR'})
    else:
        bpy.ops.object.bake('EXEC_DEFAULT', type=bake_type, save_mode='EXTERNAL', **kwargs)

    output_path = textures_dir / f"{obj.name}_{name.split('_')[-1]}.png"
    image.filepath_raw = str(output_path)
    image.file_format = 'PNG'
    image.save()
    obj.select_set(False)
    tex_node.select = False
    nodes.remove(tex_node)
    return output_path


def bake_color(obj: bpy.types.Object) -> None:
    # ensure_bake_lighting()
    _bake_image(obj, f"{obj.name}_color", 'sRGB', 'DIFFUSE')


def bake_normal(obj: bpy.types.Object) -> None:
    _bake_image(obj, f"{obj.name}_normal", 'Non-Color', 'NORMAL')


def bake_roughness(obj: bpy.types.Object) -> None:
    _bake_image(obj, f"{obj.name}_roughness", 'Non-Color', 'ROUGHNESS')


def create_metallic_from_roughness(obj_name: str) -> None:
    textures_dir = _ensure_textures_dir()
    roughness_path = textures_dir / f"{obj_name}_roughness.png"
    if not roughness_path.exists():
        return
    metallic_name = f"{obj_name}_metallic"
    if metallic_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[metallic_name])
    roughness_img = bpy.data.images.load(str(roughness_path))
    metallic_img = bpy.data.images.new(name=metallic_name, width=roughness_img.size[0], height=roughness_img.size[1], alpha=True)
    metallic_img.colorspace_settings.name = 'Non-Color'
    pixels = np.array(roughness_img.pixels[:]).reshape(-1, 4)
    metallic_pixels = np.zeros((len(pixels), 4), dtype=np.float32)
    metallic_pixels[:, 3] = 1.0 - pixels[:, 1]
    metallic_img.pixels = metallic_pixels.ravel()
    metallic_path = textures_dir / f"{obj_name}_metallic.png"
    metallic_img.filepath_raw = str(metallic_path)
    metallic_img.file_format = 'PNG'
    metallic_img.save()
    bpy.data.images.remove(roughness_img)
    print(f"Created metallic texture for {obj_name}")


def rebuild_material_with_bakes(obj: bpy.types.Object) -> None:
    textures_dir = _ensure_textures_dir()
    mat = obj.data.materials[0]
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output_node = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    output_node.location = (200, 0)
    principled.location = (0, 0)
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])

    def add_texture(name: str, colorspace: str, target):
        path = textures_dir / f"{obj.name}_{name}.png"
        if not path.exists():
            return None
        image = bpy.data.images.load(str(path))
        image.colorspace_settings.name = colorspace
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = image
        tex_node.location = (-400, target)
        return tex_node

    color_tex = add_texture('color', 'sRGB', 200)
    normal_tex = add_texture('normal', 'Non-Color', -100)
    rough_tex = add_texture('roughness', 'Non-Color', -300)
    metallic_tex = add_texture('metallic', 'Non-Color', -500)

    if color_tex:
        links.new(color_tex.outputs['Color'], principled.inputs['Base Color'])
    if normal_tex:
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.location = (-200, -100)
        links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    if rough_tex:
        links.new(rough_tex.outputs['Color'], principled.inputs['Roughness'])
    if metallic_tex:
        links.new(metallic_tex.outputs['Color'], principled.inputs['Metallic'])


def export_glb(obj: bpy.types.Object) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    glb_path = OUTPUT_DIR / f"{PROJECT_NAME}_{SESSION_LABEL}.glb"
    bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format='GLB', use_selection=True)
    print(f"Exported GLB to {glb_path}")


def export_unity_assets(obj: bpy.types.Object) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    required_textures = ['color', 'normal', 'metallic']
    for tex in required_textures:
        src = TEXTURES_DIR / f"{obj.name}_{tex}.png"
        if src.exists():
            shutil.copyfile(src, UNITY_DIR / src.name)
    fbx_path = UNITY_DIR / f"{PROJECT_NAME}_{SESSION_LABEL}.fbx"
    bpy.ops.export_scene.fbx(
        filepath=str(fbx_path),
        use_selection=True,
        mesh_smooth_type='FACE',
        add_leaf_bones=False,
        use_armature_deform_only=True,
        bake_anim=False,
    )
    print(f"Exported FBX to {fbx_path}")


def main() -> None:
    ensure_stl_importer_enabled()
    surfaces_dir = resolve_surfaces_dir()
    configure_output_dirs(surfaces_dir)
    clear_scene()

    tumor_paths = discover_tumor_segments(surfaces_dir)
    tumor_obj = import_and_join_tumor(tumor_paths)
    apply_voxel_remesh(tumor_obj)

    tumor_material = append_tumor_material()
    bpy.ops.wm.save_mainfile(filepath="post_append.blend")
    apply_material(tumor_obj, tumor_material)
    uv_map_object(tumor_obj)

    bake_setup()

    bake_color(tumor_obj)

    bake_normal(tumor_obj)
    bake_roughness(tumor_obj)
    create_metallic_from_roughness(tumor_obj.name)
    rebuild_material_with_bakes(tumor_obj)

    export_glb(tumor_obj)
    export_unity_assets(tumor_obj)


if __name__ == "__main__":
    main()
