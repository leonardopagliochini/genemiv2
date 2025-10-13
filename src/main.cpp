#ifndef __attr_dealloc
#  define __attr_dealloc(dealloc, argno)
#endif
#ifndef __attr_dealloc_free
#  define __attr_dealloc_free
#endif
#ifndef __attribute_alloc_align__
#  define __attribute_alloc_align__(...)
#endif
#ifndef __fortified_attr_access
#  define __fortified_attr_access(a, b, c)
#endif
#ifndef __COLD
#  define __COLD
#endif

#include "FisherKolmogorov.hpp"
#include "mesh_data.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace
{
  void print_usage(const char *program_name, unsigned int mpi_rank)
  {
    if (mpi_rank != 0)
      return;

    std::cerr << "Usage: " << program_name
              << " --mesh <preset_or_path> [options]\n"
              << "Options:\n"
              << "  --T <value>               Final simulation time (default: 9)\n"
              << "  --deltat <value>          Time step size (default: 1/12)\n"
              << "  --output <directory>      Output folder for VTU results (default: ./output)\n"
              << "  --output-period <value>   Number of time steps between outputs (default: 6)\n"
              << "  -h, --help                Show this message\n"
              << "\n"
              << "Mesh identifiers can be one of the predefined presets "
              << "(MNI, Ernie, BrainCoarse, Cube40, Sagittal, Sagittal_whiteGrayDiff)\n"
              << "or a path to a supported mesh file.\n";
  }

  bool is_option(const std::string &arg, const std::string &long_opt, const std::string &short_opt)
  {
    return arg == long_opt || (!short_opt.empty() && arg == short_opt);
  }
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int r = 1;
  double             T = 9.0;
  double             deltat = 1.0 / 12.0;
  unsigned int       output_period = 6;
  std::string        output_directory = "./output";
  std::string        mesh_argument;
  bool               mesh_argument_set = false;

  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];

      if (is_option(arg, "--help", "-h"))
        {
          print_usage(argv[0], mpi_rank);
          return 0;
        }
      else if (is_option(arg, "--mesh", "-m"))
        {
          if (i + 1 >= argc)
            {
              if (mpi_rank == 0)
                std::cerr << "Option " << arg << " requires a value." << std::endl;
              print_usage(argv[0], mpi_rank);
              return 1;
            }
          mesh_argument     = argv[++i];
          mesh_argument_set = true;
        }
      else if (arg == "--T")
        {
          if (i + 1 >= argc)
            {
              if (mpi_rank == 0)
                std::cerr << "Option --T requires a value." << std::endl;
              print_usage(argv[0], mpi_rank);
              return 1;
            }
          try
            {
              T = std::stod(argv[++i]);
            }
          catch (const std::exception &)
            {
              if (mpi_rank == 0)
                std::cerr << "Unable to parse value for --T." << std::endl;
              return 1;
            }
        }
      else if (arg == "--deltat")
        {
          if (i + 1 >= argc)
            {
              if (mpi_rank == 0)
                std::cerr << "Option --deltat requires a value." << std::endl;
              print_usage(argv[0], mpi_rank);
              return 1;
            }
          try
            {
              deltat = std::stod(argv[++i]);
            }
          catch (const std::exception &)
            {
              if (mpi_rank == 0)
                std::cerr << "Unable to parse value for --deltat." << std::endl;
              return 1;
            }
        }
      else if (is_option(arg, "--output", "-o"))
        {
          if (i + 1 >= argc)
            {
              if (mpi_rank == 0)
                std::cerr << "Option " << arg << " requires a value." << std::endl;
              print_usage(argv[0], mpi_rank);
              return 1;
            }
          output_directory = argv[++i];
        }
      else if (arg == "--output-period")
        {
          if (i + 1 >= argc)
            {
              if (mpi_rank == 0)
                std::cerr << "Option --output-period requires a value." << std::endl;
              print_usage(argv[0], mpi_rank);
              return 1;
            }
          try
            {
              output_period = static_cast<unsigned int>(std::stoi(argv[++i]));
            }
          catch (const std::exception &)
            {
              if (mpi_rank == 0)
                std::cerr << "Unable to parse value for --output-period." << std::endl;
              return 1;
            }
        }
      else if (!arg.empty() && arg[0] != '-')
        {
          if (!mesh_argument_set)
            {
              mesh_argument     = arg;
              mesh_argument_set = true;
            }
          else
            {
              if (mpi_rank == 0)
                std::cerr << "Unexpected positional argument '" << arg << "'." << std::endl;
              print_usage(argv[0], mpi_rank);
              return 1;
            }
        }
      else
        {
          if (mpi_rank == 0)
            std::cerr << "Unknown option '" << arg << "'." << std::endl;
          print_usage(argv[0], mpi_rank);
          return 1;
        }
    }

  if (!mesh_argument_set)
    {
      if (mpi_rank == 0)
        std::cerr << "Missing required argument --mesh." << std::endl;
      print_usage(argv[0], mpi_rank);
      return 1;
    }

  const unsigned int dim = get_mesh_dimension(mesh_argument, mpi_rank);

  MeshData<2> mesh2;
  MeshData<3> mesh3;

  if (dim == 2)
    {
      if (is_known_2d_mesh_preset(mesh_argument))
        mesh2 = get_mesh_data<2>(mesh_argument, mpi_rank);
      else
        mesh2 = get_mesh_data_from_path_2d(mesh_argument, mpi_rank);
    }
  else
    {
      if (is_known_3d_mesh_preset(mesh_argument))
        mesh3 = get_mesh_data<3>(mesh_argument, mpi_rank);
      else
        mesh3 = get_mesh_data_from_path_3d(mesh_argument, mpi_rank);
    }

  std::string normalized_output_directory =
    output_directory.empty() ? std::string("./output") : output_directory;

  if (mpi_rank == 0)
    {
      const std::string command = "mkdir -p \"" + normalized_output_directory + "\"";
      const int         mkdir_result = std::system(command.c_str());
      if (mkdir_result != 0)
        {
          std::cerr << "Failed to create output directory '"
                    << normalized_output_directory
                    << "' (command exit code " << mkdir_result << ")." << std::endl;
          return 1;
        }
    }

  MPI_Barrier(MPI_COMM_WORLD);
  output_directory = normalized_output_directory;

  printRankZero("Note: axonal_vector field and material_id will be written only at the first time step "
                "(time_step=0), to save disk space.",
                mpi_rank);

  if (dim == 2)
    {
      FisherKolmogorov<2> problem(mesh2, r, T, deltat, output_period, output_directory);
      problem.setup();
      problem.solve();
    }
  else
    {
      FisherKolmogorov<3> problem(mesh3, r, T, deltat, output_period, output_directory);
      problem.setup();
      problem.solve();
    }

  return 0;
}
