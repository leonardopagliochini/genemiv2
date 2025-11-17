import os
import sys
import bpy
import numpy as np 
import shutil

## numpy import
#conda_site_packages = "/home/techuvant/miniconda3/envs/3dslicer_env/lib/python3.12/site-packages"
#if conda_site_packages not in sys.path:
#    sys.path.append(conda_site_packages)

#if "io_mesh_stl" not in bpy.context.preferences.addons:
#    bpy.ops.preferences.addon_enable(module="io_mesh_stl")
    
PROJECT_NAME = "Caso12"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/Users/1tech/Documents/blender_bake/" #bake_test
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) #bake_test
CASO111_DIR = os.path.join(PROJECT_ROOT, "Caso12")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", PROJECT_NAME)
TEXTURES_DIR = os.path.join(OUTPUT_DIR, "Textures")
SHADERS_DIR = os.path.join(PROJECT_ROOT, "Shaders")
UNITY_DIR = os.path.join(OUTPUT_DIR, PROJECT_NAME + "_uy")


def export_fbx(filepath):
    bpy.ops.export_scene.fbx(filepath=filepath)
    print(f"Exported FBX: {filepath}")
    
def export_unity_assets(material_data):
    """Export FBX and required textures to Unity-ready folder structure"""
    for obj_name in material_data.keys():
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            print(f"Object {obj_name} not found for Unity export")
            continue
        
    
        os.makedirs(UNITY_DIR, exist_ok=True)
        
        # Copy required textures
        required_textures = ['color', 'normal', 'metallic']
        for tex_type in required_textures:
            src = os.path.join(TEXTURES_DIR, f"{obj_name}_{tex_type}.png")
            dst = os.path.join(UNITY_DIR, f"{obj_name}_{tex_type}.png")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                print(f"Copied {tex_type} texture to Unity folder")
            else:
                print(f"Missing texture: {src}")

    # Export FBX
    fbx_path = os.path.join(UNITY_DIR, f"{PROJECT_NAME}.fbx")
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        use_selection=False,
        mesh_smooth_type='FACE',
        add_leaf_bones=False,
        use_armature_deform_only=True,
        bake_anim=False
    )
    print(f"Exported Unity-ready FBX to {fbx_path}")


def create_metallic_from_roughness(material_data):
    """Create metallic texture from roughness by inverting green channel in alpha"""
    for obj_name in material_data.keys():
        # Get roughness texture
        roughness_path = os.path.join(TEXTURES_DIR, f"{obj_name}_roughness.png")
        if not os.path.exists(roughness_path):
            print(f"Roughness texture missing for {obj_name}")
            continue

        # Delete existing metallic image if present
        metallic_name = f"{obj_name}_metallic"
        if metallic_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[metallic_name])

        # Create new metallic image
        roughness_img = bpy.data.images.load(roughness_path)
        metallic_img = bpy.data.images.new(
            name=metallic_name,
            width=roughness_img.size[0],
            height=roughness_img.size[1],
            alpha=True
        )
        metallic_img.colorspace_settings.name = 'Non-Color'

        # Process pixels: RGB=0, Alpha=1-GreenChannel
        print(f"Processing metallic texture for {obj_name}...")
        pixels = np.array(roughness_img.pixels[:]).reshape(-1, 4)
        metallic_pixels = np.zeros((len(pixels), 4), dtype=np.float32)
        metallic_pixels[:,3] = 1.0 - pixels[:,1]  # Invert green channel for alpha
        metallic_img.pixels = metallic_pixels.ravel()
        
        # Save and clean up
        metallic_img.filepath_raw = os.path.join(TEXTURES_DIR, f"{obj_name}_metallic.png")
        metallic_img.file_format = 'PNG'
        metallic_img.save()
        
        bpy.data.images.remove(roughness_img)
        print(f"Created metallic texture for {obj_name}")

def update_shader_nodes(material_data):
    """Update material nodes to use metallic texture instead of roughness"""
    for obj_name in material_data.keys():
        obj = bpy.data.objects.get(obj_name)
        if not obj or not obj.data.materials:
            continue
            
        mat = obj.data.materials[0]
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Remove roughness node
        if f"{obj_name}_roughness" in nodes:
            nodes.remove(nodes[f"{obj_name}_roughness"])

        # Create metallic node
        metallic_node = nodes.new('ShaderNodeTexImage')
        metallic_node.name = f"{obj_name}_metallic"
        metallic_node.image = bpy.data.images.get(f"{obj_name}_metallic")
        metallic_node.location = (-400, 200)  # Position in node tree

        # Connect to shader
        principled = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
        links.new(metallic_node.outputs['Color'], principled.inputs['Metallic'])



def import_stl_files(folder_path, stl_files, rename_mapping):

    imported_objects = []
    
    for filename in stl_files:
        filepath = os.path.join(folder_path, filename)
        
        if os.path.exists(filepath):
            bpy.ops.import_mesh.stl(filepath=filepath)
            obj = bpy.context.selected_objects[0]  # Get the imported object
            
            # Rename object if a new name is provided, otherwise use the original filename
            new_name = rename_mapping.get(filename, filename.replace(".stl", ""))
            obj.name = new_name  
            
            imported_objects.append(obj)
            print(f"Imported: {filename} as '{new_name}'")
        else:
            print(f"File not found: {filename}")

def export_glb(filepath):

    bpy.ops.export_scene.gltf(filepath=filepath, export_format='GLB')
    print(f"Exported GLB: {filepath}")

def append_materials(material_data):
    
    for key, material in material_data.items():
        material_path = material['scene']
        material_name = material['material']

        if material_name in bpy.data.materials:
            print(f"Material {material_name} already appended.")
            return bpy.data.materials[material_name]

        try:
            bpy.ops.wm.append(filepath=material_path, directory= material_path + "/Material/", filename=material_name)
            material = bpy.data.materials.get(material_name)

            if material:
                print(f"Appended material: {material_name}")
            else:
                print(f"Failed to append material: {material_name}")

        except RuntimeError as e:
            print(f"Error appending material {material_name}: {e}")

def apply_material(material_data):

    for object_name, material_info in material_data.items():
        material = bpy.data.materials.get(material_info['material'])
        obj = bpy.data.objects.get(object_name)

        if obj and material:
            obj.data.materials.clear()  # Remove existing materials
            obj.data.materials.append(material)
            print(f"Applied {material.name} to {obj.name}")
        else:
            print(f"Failed to apply material to {object_name}: Object or material not found.")

        obj = bpy.data.objects.get(object_name)
        print(f"{object_name} has material: {obj.data.materials[0].name if obj and obj.data.materials else 'No material'}")

def uv_map(material_data):

    for obj_name in material_data.keys():
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            print(f"Error: Object '{obj_name}' not found.")
            continue

        # Ensure object is in edit mode for unwrapping
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Select all faces and apply Smart UV Project
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.00)
        
        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        print(f"UV map created for {obj_name} using Smart UV Project.")

def remove_texture_nodes(material_data):

    for obj_name in material_data.keys():
        obj = bpy.data.objects.get(obj_name)
        if not obj or not obj.data.materials:
            print(f"Object '{obj_name}' not found or has no materials.")
            continue
        
        for mat in obj.data.materials:
            if mat.use_nodes:
                nodes = mat.node_tree.nodes
                for node in nodes:
                    if isinstance(node, bpy.types.ShaderNodeTexImage):
                        name = node.name
                        nodes.remove(node)
                        print(f"Removed texture node: {name}")

def bake_setup():

        # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    # Set bake settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

def bake_color(material_data):

    for obj_name in material_data.keys():
        print(obj_name)
        obj = bpy.data.objects.get(obj_name)
        if not obj or not obj.data.materials:
            print(f"Object '{obj_name}' not found or has no materials.")
            continue
        
        #create a new image for texturing
        image = bpy.data.images.new(name= f"{obj_name}_color", width=1024, height=1024)
        image.colorspace_settings.name = 'sRGB'

        #create a texture node for baking
        mat = obj.data.materials[0]
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = image
        tex_node.name = image.name
        nodes.active = tex_node

        tex_node.select = True
        obj.select_set(True)

        #bake the texture
        bpy.ops.object.bake('EXEC_DEFAULT', type='DIFFUSE', pass_filter={'COLOR'}, save_mode='EXTERNAL')
    
        image_path = os.path.join(TEXTURES_DIR, f"{obj_name}_color.png")
        image.filepath_raw = image_path
        image.file_format = 'PNG'
        image.save()
    
        obj.select_set(False)
        tex_node.select = False

def bake_normal(material_data):

    for obj_name in material_data.keys():
        print(obj_name)
        obj = bpy.data.objects.get(obj_name)
        if not obj or not obj.data.materials:
            print(f"Object '{obj_name}' not found or has no materials.")
            continue
        
        #create a new image for texturing
        image = bpy.data.images.new(name= f"{obj_name}_normal", width=1024, height=1024)
        image.colorspace_settings.name = 'Non-Color'

        #create a texture node for baking
        mat = obj.data.materials[0]
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = image
        tex_node.name = image.name
        nodes.active = tex_node

        tex_node.select = True
        obj.select_set(True)

        #bake the texture
        bpy.ops.object.bake('EXEC_DEFAULT', type='NORMAL', save_mode='EXTERNAL')
        
        image_path = os.path.join(TEXTURES_DIR, f"{obj_name}_normal.png")
        image.filepath_raw = image_path
        image.file_format = 'PNG'
        image.save()
        
        obj.select_set(False)
        tex_node.select = False

def bake_roughness(material_data):

    for obj_name in material_data.keys():
        print(obj_name)
        obj = bpy.data.objects.get(obj_name)
        if not obj or not obj.data.materials:
            print(f"Object '{obj_name}' not found or has no materials.")
            continue
        
        #create a new image for texturing
        image = bpy.data.images.new(name= f"{obj_name}_roughness", width=1024, height=1024)
        image.colorspace_settings.name = 'Non-Color'

        #create a texture node for baking
        mat = obj.data.materials[0]
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = image
        tex_node.name = image.name  # Give the new node the same name as the image
        nodes.active = tex_node

        tex_node.select = True
        obj.select_set(True)

        #bake the texture
        bpy.ops.object.bake('EXEC_DEFAULT', type='ROUGHNESS', save_mode='EXTERNAL')
        
        image_path = os.path.join(TEXTURES_DIR, f"{obj_name}_roughness.png")
        image.filepath_raw = image_path
        image.file_format = 'PNG'
        image.save()
        
        obj.select_set(False)
        tex_node.select = False

def link_nodes(material_data):

        for obj_name, material_info in material_data.items():
            obj = bpy.data.objects.get(obj_name)
            if not obj or not obj.data.materials:
                print(f"Object '{obj_name}' not found or has no materials.")
                continue
            
            mat = obj.data.materials[0]
            if not mat.use_nodes:
                print(f"Material {mat.name} does not use nodes.")
                continue
            
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Find Principled BSDF node
            principled_node = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
            if not principled_node:
                print(f"Error: No Principled BSDF node found in {mat.name}.")
                continue

            # Connect baked textures to the correct nodes
            for texture_type in ['color', 'normal', 'roughness']:
                texture_node = next((node for node in nodes if node.name == f"{obj_name}_{texture_type}"), None)

                if not texture_node:
                    print(f"Error: No texture node found for {texture_type} in {mat.name}.")
                    continue

                if texture_type == 'color':
                    links.new(texture_node.outputs[0], principled_node.inputs['Base Color'])
                elif texture_type == 'normal':
                    normal_map_node = nodes.new('ShaderNodeNormalMap')
                    links.new(texture_node.outputs[0], normal_map_node.inputs['Color'])
                    links.new(normal_map_node.outputs[0], principled_node.inputs['Normal'])
                elif texture_type == 'roughness':
                    links.new(texture_node.outputs[0], principled_node.inputs['Roughness'])

            #remove every unsed node
            for node in nodes:
                if node.type != 'BSDF_PRINCIPLED' and node.type != 'TEX_IMAGE' and node.type != 'NORMAL_MAP' and node.type != 'OUTPUT_MATERIAL':
                    nodes.remove(node)

            print(f"Linked texture nodes for {obj_name}.")
            

def assign_custom_colors():
    color_assignments = {
        "atrium_center": "#24B3B1",
        "vein_entry": "#484798",
        "entry": "#EE7119",
        "exit": "#E2047F"
    }
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,)
    
    for obj_name, hex_color in color_assignments.items():
        obj = bpy.data.objects.get(obj_name)
        if obj:
            # Create new material
            mat_name = f"{obj_name}_color"
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            
            # Get principled BSDF node
            nodes = mat.node_tree.nodes
            principled = nodes.get('Principled BSDF')
            
            # Set base color
            principled.inputs['Base Color'].default_value = hex_to_rgb(hex_color)
            
            # Assign material to object
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            
            print(f"Assigned color {hex_color} to {obj_name}")
        else:
            print(f"Object {obj_name} not found for color assignment")


if __name__ == "__main__":

    case_number = 12

    stl_folder = CASO111_DIR# Change this to your STL folder
    output_glb = OUTPUT_DIR  # Change this to your desired output file
    #output_fbx = OUTPUT_DIR + f"Heart/Heart.fbx"

#    # Clear existing objects (optional)
#    bpy.ops.object.select_all(action='SELECT')
#    bpy.ops.object.delete()

    # List of specific STL files to import
    stl_files = [f"Segmentation_Inside{case_number}.stl",
#                f"Segmentation_Fat{case_number}.stl",
                f"Segmentation_Bones{case_number}.stl", 
                f"Segmentation_Skin{case_number}.stl",
                f"catheter{case_number}.stl",
                f"point0.stl",
                f"point1.stl",
                f"point2.stl",
                f"point3.stl"
                ]

    # Dictionary for renaming (optional)
    rename_mapping = {
        f"Segmentation_Inside{case_number}.stl": "Heart",
#        f"Segmentation_Fat{case_number}.stl": "Fat",
        f"Segmentation_Bones{case_number}.stl": "Bones",
        f"Segmentation_Skin{case_number}.stl": "Skin",
        f"catheter{case_number}.stl": "Catheter",
        "point0.stl": "atrium_center",
        "point1.stl": "vein_entry",
        "point2.stl": "entry",
        "point3.stl": "exit"
        }
    
    material_mapping = {
        "Heart": {"scene": SHADERS_DIR + "/veiny_organ.blend", "material": "veiny_organ_mat"},
#        "Fat": {"scene": SHADERS_DIR + "/fat_tissue.blend", "material": "fat_tissue_mat"},
        "Bones": {"scene": SHADERS_DIR + "/bone.blend", "material": "bone_mat"}, 
        "Skin" : {"scene" : SHADERS_DIR + "/skin.blend", "material": "skin_mat"}
    }

    import_stl_files(stl_folder, stl_files, rename_mapping)
    
    append_materials(material_mapping)
    apply_material(material_mapping)
    
    assign_custom_colors()

    uv_map(material_mapping)
    remove_texture_nodes(material_mapping)

    bake_setup()
    bake_color(material_mapping)
    bake_normal(material_mapping)
    bake_roughness(material_mapping)

    link_nodes(material_mapping)
    
    export_glb(output_glb)
    
    ##########################
    
    create_metallic_from_roughness(material_mapping)
    update_shader_nodes(material_mapping)
    
    export_unity_assets(material_mapping)


