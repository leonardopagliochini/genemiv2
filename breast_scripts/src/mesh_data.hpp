#include "FisherKolmogorov.hpp"

template<unsigned int dim>
MeshData<dim> get_mesh_data(const std::string &mesh_preset, unsigned int mpiRank);

inline std::string
mesh_basename(const std::string &path)
{
    const std::string separators = "/\\";
    const auto pos = path.find_last_of(separators);
    if (pos == std::string::npos)
        return path;
    return path.substr(pos + 1);
}

inline bool
contains_path_separator(const std::string &path)
{
    return path.find('/') != std::string::npos || path.find('\\') != std::string::npos;
}

template <typename MeshDataType>
inline bool
matches_mesh_identifier(const std::string &user_value, const MeshDataType &candidate)
{
    return user_value == candidate.mesh_file_name ||
           mesh_basename(user_value) == mesh_basename(candidate.mesh_file_name);
}

inline bool
is_known_3d_mesh_preset(const std::string &mesh_preset)
{
    return mesh_preset == "MNI" ||
           mesh_preset == "Ernie" ||
           mesh_preset == "BrainCoarse" ||
           mesh_preset == "Breast" ||
           mesh_preset == "Cube40";
}

inline bool
is_known_2d_mesh_preset(const std::string &mesh_preset)
{
    return mesh_preset == "Sagittal" ||
           mesh_preset == "Sagittal_whiteGrayDiff";
}

MeshData<3> Breast{
    "../breast_128_augmented.msh", // mesh_file_name
    {    // material_names (id -> name)
        {1, "skin"},
        {2, "fat"},
        {3, "ductal"},
        {4, "lobulus"},
        {5, "stroma"}
    },
    {    // isotropic diffusion (name -> value)
        {"skin", 0},
        {"fat", 0.1826},
        {"ductal", 3.6525},
        {"lobulus", 0.3653},
        {"stroma", 1.8263}
    },
    {    // axonal diffusion (name -> value)
        {"skin", 0},
        {"fat", 0},
        {"ductal", 0},
        {"lobulus", 0},
        {"stroma", 0}
    },
    {    // alpha coefficients (name -> value)
        {"skin", 0},
        {"fat", 0.904},
        {"ductal", 1.292},
        {"lobulus", 1.421},
        {"stroma", 1.55}
    },
    0, 0, 0, // x0, y0, z0 (initial condition center coords)
    15, // radius
    10, // center_threshold
    {0.0, 0.0, 40.0}, // axonal_center
    40, // a (X axis)
    60, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> MNI{
    "../mesh/MNI_with_phys.msh", // mesh_file_name
    {    // material_names (id -> name)
        {0, "gray matter"},
        {1, "CSF"},
        {2, "white matter"},
        {3, "ventricule"}
    },
    {    // isotropic diffusion (name -> value)
        {"gray matter", 3},
        {"CSF", 0},
        {"white matter", 3},
        {"ventricule", 0}
    },
    {    // axonal diffusion (name -> value)
        {"gray matter", 50},
        {"CSF", 0},
        {"white matter", 50},
        {"ventricule", 0}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 0.6},
        {"CSF", 0},
        {"white matter", 1.8},
        {"ventricule", 0}
    },
    -40, -18.0, -10.0, // x0, y0, z0 (initial condition center coords)
    15, // radius
    10, // center_threshold
    {0.0, 0.0, 40.0}, // axonal_center
    40, // a (X axis)
    60, // b (Y axis)
    30  // c (Z axis)
};


MeshData<3> Ernie{
    "../mesh/ernie_brain_dealii.msh", // mesh_file_name
    {    // material_names (id -> name)
        {1, "white matter"},
        {2, "gray matter"}
    },
    {    // isotropic diffusion (name -> value)
        {"white matter", 3},
        {"gray matter", 3}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 50},
        {"gray matter", 50}
    },
    {    // alpha coefficients (name -> value)
        {"white matter", 1.2},
        {"gray matter", 0.6}
    },
    0, 0, 25.0, // x0, y0, z0 (initial condition center coords)
    15, // radius
    10, // center_threshold
    {0.0, 0.0, 0.0}, // axonal_center
    60, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> BrainCoarse{
    "../mesh/brain_coarse.msh", // mesh_file_name
    {    // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    },
    {    // isotropic diffusion (name -> value)
        {"white matter", 3},
        {"gray matter", 3}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 40},
        {"gray matter", 40}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 0.6},
        {"white matter", 1.2}
    },
    0.0, 0.0, 25.0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    10, // center_threshold
    {0.0, 0.0, 0.0}, // axonal_center
    40, // a (X axis)
    60, // b (Y axis)
    30  // c (Z axis)
};

MeshData<3> Cube40{
    "../mesh/mesh-cube-40.msh", // mesh_file_name
    {   // material_names (id -> name)
        {0, "matter"},
        {10, "matter"}
    },
    {    // isotropic diffusion (name -> value)
        {"matter", 0.003}
    },
    {    // axonal diffusion (name -> value)
        {"matter", 0.01}
    },
    {    // alpha coefficients (name -> value)
        {"matter", 5}
    },
    0.5, 0.0, 0.1, // x0, y0, z0 (initial condition center coords)
    0.05, // radius
    0.05,  // center_threshold
    {0.5, 0.5, 0.5}, // axonal_center
    0.25, // a (X axis)
    0.25, // b (Y axis)
    0.25  // c (Z axis)
};

MeshData<2> Sagittal{
    "../mesh/sagittal.msh", // mesh_file_name
    {   // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    }, 
    {    // isotropic diffusion (name -> value)
        {"white matter", 5},
        {"gray matter", 5}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 50},
        {"gray matter", 50}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 1},
        {"white matter", 2}
    },
    230, 100, 0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    5,  // center_threshold
    {190.0, 140.0}, // axonal_center
    70, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

MeshData<2> Sagittal_whiteGrayDiff{
    "../mesh/sagittal.msh", // mesh_file_name
    {   // material_names (id -> name)
        {0, "gray matter"},
        {1, "white matter"}
    }, 
    {    // isotropic diffusion (name -> value)
        {"white matter", 20},
        {"gray matter", 0.05}
    },
    {    // axonal diffusion (name -> value)
        {"white matter", 0.05},
        {"gray matter", 30}
    },
    {    // alpha coefficients (name -> value)
        {"gray matter", 1},
        {"white matter", 2}
    },
    230, 100, 0, // x0, y0, z0 (initial condition center coords)
    20, // radius
    5,  // center_threshold
    {190.0, 140.0}, // axonal_center
    70, // a (X axis)
    40, // b (Y axis)
    30  // c (Z axis)
};

void printErrValidOptions(const std::string& mesh_preset, unsigned int mpiRank){
    if (mpiRank != 0) return;
    std::cerr << "Unknown mesh preset: " << mesh_preset << std::endl;
    std::cerr << "Valid options are: " << std::endl;

    std::cerr << "2D: " << std::endl;
    std::cerr << " - Sagittal" << std::endl;
    std::cerr << " - Sagittal_whiteGrayDiff" << std::endl;

    std::cerr << "3D: " << std::endl;
    std::cerr << " - MNI" << std::endl;
    std::cerr << " - Ernie" << std::endl;
    std::cerr << " - BrainCoarse" << std::endl;
    std::cerr << " - Breast" << std::endl;
    std::cerr << " - Cube40" << std::endl;
}

inline void printRankZero(const std::string& msg, unsigned int mpiRank){
    if (mpiRank == 0) std::cout << msg << std::endl;
}

inline bool try_match_mesh_path_3d(const std::string &mesh_value, MeshData<3> &matched_mesh){
    if(matches_mesh_identifier(mesh_value, MNI)){
        matched_mesh = MNI;
        if (contains_path_separator(mesh_value))
            matched_mesh.mesh_file_name = mesh_value;
        return true;
    }
    if(matches_mesh_identifier(mesh_value, Ernie)){
        matched_mesh = Ernie;
        if (contains_path_separator(mesh_value))
            matched_mesh.mesh_file_name = mesh_value;
        return true;
    }
    if(matches_mesh_identifier(mesh_value, BrainCoarse)){
        matched_mesh = BrainCoarse;
        if (contains_path_separator(mesh_value))
            matched_mesh.mesh_file_name = mesh_value;
        return true;
    }
    if(matches_mesh_identifier(mesh_value, Breast)){
        matched_mesh = Breast;
        if (contains_path_separator(mesh_value))
            matched_mesh.mesh_file_name = mesh_value;
        return true;
    }
    if(matches_mesh_identifier(mesh_value, Cube40)){
        matched_mesh = Cube40;
        if (contains_path_separator(mesh_value))
            matched_mesh.mesh_file_name = mesh_value;
        return true;
    }
    return false;
}

inline bool try_match_mesh_path_2d(const std::string &mesh_value, MeshData<2> &matched_mesh){
    if(matches_mesh_identifier(mesh_value, Sagittal)){
        matched_mesh = Sagittal;
        if (contains_path_separator(mesh_value))
            matched_mesh.mesh_file_name = mesh_value;
        return true;
    }
    if(matches_mesh_identifier(mesh_value, Sagittal_whiteGrayDiff)){
        matched_mesh = Sagittal_whiteGrayDiff;
        if (contains_path_separator(mesh_value))
            matched_mesh.mesh_file_name = mesh_value;
        return true;
    }
    return false;
}

unsigned int get_mesh_dimension(const std::string& mesh_value, unsigned int mpiRank){
    if(is_known_3d_mesh_preset(mesh_value))
        return 3;
    if(is_known_2d_mesh_preset(mesh_value))
        return 2;

    MeshData<3> mesh3;
    if (try_match_mesh_path_3d(mesh_value, mesh3))
        return 3;

    MeshData<2> mesh2;
    if (try_match_mesh_path_2d(mesh_value, mesh2))
        return 2;

    printErrValidOptions(mesh_value, mpiRank);
    exit(-1);
}

template<>
MeshData<3> get_mesh_data(const std::string& mesh_preset, unsigned int mpiRank){
    if(mesh_preset == "MNI"){
        printRankZero("Using MNI mesh.", mpiRank);
        return MNI;
    } else if(mesh_preset == "Ernie"){
        printRankZero("Using Ernie mesh.", mpiRank);
        return Ernie;
    } else if(mesh_preset == "BrainCoarse"){
        printRankZero("Using BrainCoarse mesh.", mpiRank);
        return BrainCoarse;
    } else if(mesh_preset == "Breast"){
        printRankZero("Using Breast mesh.", mpiRank);
        return Breast;
    } else if(mesh_preset == "Cube40"){
        printRankZero("Using Cube40 mesh.", mpiRank);
        return Cube40;
    } else {
        printErrValidOptions(mesh_preset, mpiRank);
        exit(-1);
    }
}

inline MeshData<3> get_mesh_data_from_path_3d(const std::string &mesh_value, unsigned int mpiRank){
    MeshData<3> mesh;
    if (try_match_mesh_path_3d(mesh_value, mesh)){
        std::string description = "Using " + mesh_basename(mesh.mesh_file_name) + " mesh.";
        if (contains_path_separator(mesh_value) && mesh.mesh_file_name == mesh_value)
            description = "Using 3D mesh file '" + mesh.mesh_file_name + "'.";
        printRankZero(description, mpiRank);
        return mesh;
    }
    printErrValidOptions(mesh_value, mpiRank);
    exit(-1);
}

inline MeshData<2> get_mesh_data_from_path_2d(const std::string &mesh_value, unsigned int mpiRank){
    MeshData<2> mesh;
    if (try_match_mesh_path_2d(mesh_value, mesh)){
        std::string description = "Using " + mesh_basename(mesh.mesh_file_name) + " mesh.";
        if (contains_path_separator(mesh_value) && mesh.mesh_file_name == mesh_value)
            description = "Using 2D mesh file '" + mesh.mesh_file_name + "'.";
        printRankZero(description, mpiRank);
        return mesh;
    }
    printErrValidOptions(mesh_value, mpiRank);
    exit(-1);
}
template<>
MeshData<2> get_mesh_data(const std::string& mesh_preset, unsigned int mpiRank){
    if(mesh_preset == "Sagittal"){
        printRankZero("Using Sagittal mesh.", mpiRank);
        return Sagittal;
    } else if(mesh_preset == "Sagittal_whiteGrayDiff"){
        printRankZero("Using Sagittal mesh with more pronounced differences between white and gray matter.", mpiRank);
        return Sagittal_whiteGrayDiff;
    } else {
        printErrValidOptions(mesh_preset, mpiRank);
        exit(-1);
    }
}
