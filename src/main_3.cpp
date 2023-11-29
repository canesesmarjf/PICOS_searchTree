#include <iostream>
#include <armadillo>
// #include <chrono>
#include <string>
// #include <sstream>
// #include <cstdlib>
// #include <ctime>
#include <filesystem>
#include <vector>
#include "particle_tree.h"

using namespace std::chrono;
using namespace std;
using namespace arma;
namespace fs = filesystem;

/* OBJECTIVE:

*/

/* HOW TO USE THIS SCRIPT:

*/

void create_dir(string dir_path)
{
  if (fs::is_directory(dir_path) == true)
    fs::remove_all(dir_path);

  if (fs::is_directory(dir_path) == false)
    fs::create_directories(dir_path);
}

void load_input_data(string root_input, string picos_case, string species_index, string time_index, vec * x_p, mat * v_p, vec * a_p)
{
  vector<string> input_file_name;
  input_file_name.resize(3);

  input_file_name[0] = root_input + picos_case + "/" + "x_p" + "_ss_" + species_index + "_tt_" + time_index + ".csv";
  input_file_name[1] = root_input + picos_case + "/" + "v_p" + "_ss_" + species_index + "_tt_" + time_index + ".csv";
  input_file_name[2] = root_input + picos_case + "/" + "a_p" + "_ss_" + species_index + "_tt_" + time_index + ".csv";

  x_p->load(input_file_name[0],csv_ascii);
  v_p->load(input_file_name[1],csv_ascii);
  a_p->load(input_file_name[2],csv_ascii);
}

int main()
{
  // Input from user:
  // =====================================================================================
  string picos_case    = "PICOS_case_2";
  string species_index = "2"; // ss
  string time_index    = "50"; // tt
  string root_input  = "./Step_1_output/";
  string root_output = "./Step_2_output/";

  // Derived strings from user's input:
  string scenario = "ss_" + species_index + "_tt_" + time_index;
  string output_dir = root_output + picos_case + "/" + scenario;

  // Create directory where output data is to be stored:
  // =====================================================================================
  create_dir(output_dir);

  // STEP 1:
  // =====================================================================================
  // Load input data:
  // =====================================================================================
  // We have three data sets: x_p, v_p and a_p
  // We will use a binary tree to index the x_p data and produce a leaf_x vector
  // For each element in leaf_x vector which has a surplus, we will use a quadtree to index the v_p data and produce a leaf_v 2D vector

  vec x_p;
  mat v_p;
  vec a_p;
  load_input_data(root_input,picos_case,species_index,time_index,&x_p, &v_p, &a_p);

  // Normalize data:
  // -------------------------------------------------------------------------------------
  // Velocity:
  double y_norm;
  if (0)
  {
      y_norm = max(max(v_p))*1;
      if (species_index == "1"){y_norm = 2e6;}
  }
  else
  {
    y_norm = 300e6;
  }

  // Position:
  double x_norm = 0;
  if (0)
  {
    x_norm = ceil(max(x_p))*2;
  }
  else
  {
    x_norm = 0.1;
  }

  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // STEP 2:
  // =====================================================================================
  // Define tree parameters:
  // =====================================================================================

  // 1D binary tree parameters:
  // -------------------------------------------------------------------------------------
  // The min and max of the bt must exactly correspond to the min and max of the data to be  indexed. Also, it must not have empty cells since this cannot be dealt with at present:

  bt_params_TYP bt_params;

  bt_params.dimensionality = 1;
  bt_params.min       = {min(x_p)};
  bt_params.max       = {max(x_p)};
  bt_params.max_depth = {+6};

  // 2D Quadtree parameters:
  // -------------------------------------------------------------------------------------
  qt_params_TYP qt_params;

  qt_params.min = {-max(max(v_p)),-max(max(v_p))};
  qt_params.max = {+max(max(v_p)),+max(max(v_p))};
  if (species_index == "1")
  {
    qt_params.min_depth = +5;
    qt_params.max_depth = +6;
    qt_params.min_count = 7;
  }
  else if (species_index == "2")
  {
    qt_params.min_depth = +5;
    qt_params.max_depth = +6;
    qt_params.min_count = 7;
  }

  // STEP 3:
  // =====================================================================================
  // Create and populate tree:
  // =====================================================================================
  // Create composite particle tree:
  // -------------------------------------------------------------------------------------
  particle_tree_TYP particle_tree(&bt_params,&qt_params,&x_p,&v_p,&a_p);

  // Populate particle tree with data:
  // -------------------------------------------------------------------------------------
  // This steps populates the trees (binary and quad) and as the final output it provides the leaf nodes for both physical and velocity space:
  particle_tree.populate_tree("binary and quad");

  // Binary tree diagnostics:
  {
    // ----------------------------------------------------------------------------------
    int k = particle_tree.bt.count_leaf_nodes();
    cout << "total number of leaf nodes populated: " << k << endl;
    k = particle_tree.bt.count_leaf_points();
    cout << "total number of leaf points inserted: " << k << endl;

    // Save leaf_x ip_count profile:
    // ----------------------------------------------------------------------------------
    particle_tree.xq.save(output_dir + "/"  + "x_q" + ".csv", csv_ascii);
    particle_tree.ip_count.save(output_dir + "/"  + "leaf_x_ip_count" + ".csv", csv_ascii);
  }

  // Quad tree diagnostics:
  {
    int Nx = bt_params.num_nodes;
    vec qt_count = zeros<vec>(Nx);
    vec bt_count = zeros<vec>(Nx);
    for (int xx = 0; xx < Nx; xx++)
    {
      qt_count[xx] = particle_tree.qt[xx].count_leaf_points();
      if (particle_tree.leaf_v[xx][0] != NULL)
        bt_count[xx] = particle_tree.ip_count[xx];
    }

    cout << "Total number of particles indexed in quad tree (binary tree) " << sum(bt_count) << endl;

    cout << "Total number of particles indexed in quad tree (quad tree) " << sum(qt_count) << endl;
  }

  // STEP 4:
  // =====================================================================================
  // Assess conservation:PRIOR TO RESAMPLING
  // =====================================================================================
  particle_tree.assess_conservation(output_dir,"0");

  // STEP 7:
  // =====================================================================================
  // Save tree data PRIOR TO RESAMPLING to visualise tree structure in MATLAB:
  // =====================================================================================
  particle_tree.save_leaf_v_structure(output_dir);

  // STEP 5:
  // =====================================================================================
  // Resample distribution:
  // =====================================================================================
  // In this process, we assemble both the binary and quad trees with corresponding leaf_x and leaf_v objects. In addition, the contents of the particle tree removed as the resampling process changes the distribution.
  particle_tree.resample_distribution();

  // STEP 6:
  // =====================================================================================
  // Save RESAMPLED distribution to file:
  // =====================================================================================
  // Rescale:
  x_p*= x_norm;
  v_p*= y_norm;

  // Save resampled distribution for post-processing
  file_type format = csv_ascii;
  x_p.save(output_dir + "/"  + "x_p_new.csv",csv_ascii);
  v_p.save(output_dir + "/"  + "v_p_new.csv",csv_ascii);
  a_p.save(output_dir + "/"  + "a_p_new.csv",csv_ascii);

  // STEP 8:
  // =====================================================================================
  // Assess conservation: AFTER RESAMPLING
  // =====================================================================================
  // Immediately after resampling, the tree is cleared. we have resampled the data, in order to assess conservation, we need clear contents of trees and re-populate them

  if (false)
  {
    // Load a new data set:
    load_input_data(root_input, picos_case, species_index, time_index,&x_p, &v_p, &a_p);
  }

  // Normalize data:
  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // Repopulate trees with resampled data:
  particle_tree.populate_tree("binary and quad");

  // Save output:
  particle_tree.ip_count.save(output_dir + "/"  + "leaf_x_ip_count_new" + ".csv", csv_ascii);

  // STEP 9:
  // =====================================================================================
  // Assess conservation: AFTER RESAMPLING
  // =====================================================================================
  particle_tree.assess_conservation(output_dir,"1");

  // At this point, we have succesfully demonstrated how to apply the trees and resample the distribution and then update the tree.

  // We now need to demonstrate that we can clear the tree data, release the tree memory with a delete function and reuse the tree without incurring into memory leaks

  // STEP 10:
  // =====================================================================================
  // Check for memory leaks when reusing particle_tree after a data clear:
  // =====================================================================================
  // Substeps are the following:
  // - Clear contents of particle_tree:
  // - Load in new data to x_p, v_p, a_p
  // - Normalize data and populate tree
  // - Resample distribution
  // - Rescale data and save results

  // Clear contents of tree:
  // particle_tree.clear_all_contents();

  // Load a new data set:
  // Choose PICOS++ case:
  picos_case = "PICOS_case_2";
  species_index = "1"; // ss
  time_index    = "50"; // tt
  load_input_data(root_input, picos_case, species_index, time_index,&x_p, &v_p, &a_p);

  // Normalize data:
  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // Repopulate trees with new data:
  particle_tree.populate_tree("binary and quad");

  // STEP XX:
  // =====================================================================================
  // Resample distribution:
  // =====================================================================================
  particle_tree.resample_distribution();

  // STEP XX:
  // =====================================================================================
  // Save resampled distribution to file:
  // =====================================================================================
  // Rescale:
  x_p*= x_norm;
  v_p*= y_norm;

  // Save resampled distribution for post-processing
  format = csv_ascii;
  x_p.save(output_dir + "/"  + "x_p_new2.csv",csv_ascii);
  v_p.save(output_dir + "/"  + "v_p_new2.csv",csv_ascii);
  a_p.save(output_dir + "/"  + "a_p_new2.csv",csv_ascii);

  // STEP 11:
  // =====================================================================================
  // Check for memory leaks when using delete:
  // =====================================================================================

  return 0;
}
