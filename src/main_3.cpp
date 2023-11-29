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
  // string input_dir  = root_output + picos_case + "/" + scenario;

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
  particle_tree.populate_tree();

  // Binary tree diagnostics:
  // -------------------------------------------------------------------------------------
  int k = particle_tree.bt.count_leaf_nodes();
  cout << "total number of leaf nodes populated: " << k << endl;
  k = particle_tree.bt.count_leaf_points();
  cout << "total number of leaf points inserted: " << k << endl;

  // Save leaf_x ip_count profile:
  // -------------------------------------------------------------------------------------
  particle_tree.xq.save(output_dir + "/"  + "x_q" + ".csv", csv_ascii);
  particle_tree.ip_count.save(output_dir + "/"  + "leaf_x_ip_count" + ".csv", csv_ascii);

  // Quad tree diagnostics:
  // -------------------------------------------------------------------------------------
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
  {
    int Nx = bt_params.num_nodes;
    vec m_t(Nx, fill::zeros);
    vec p_x(Nx, fill::zeros);
    vec p_r(Nx, fill::zeros);
    vec KE(Nx, fill::zeros);

    for (int xx = 0; xx < Nx ; xx++)
    {
      if (particle_tree.leaf_x[xx] != NULL)
      {
        uvec ip = conv_to<uvec>::from(particle_tree.leaf_x[xx]->ip);
        m_t[xx] = sum(a_p.elem(ip));
        mat v_p_subset = v_p.rows(ip);
        p_x[xx] = dot(a_p.elem(ip),v_p_subset.col(0));
        p_r[xx] = dot(a_p.elem(ip),v_p_subset.col(1));

        KE[xx] = (dot(a_p.elem(ip),pow(v_p_subset.col(0),2)) + dot(a_p.elem(ip),pow(v_p_subset.col(1),2)));
      }
    }

    cout << "Total KE before resampling = " + to_string(sum(KE)) << endl;

    m_t.save(output_dir + "/" + "m_profile" + ".csv",csv_ascii);
    p_x.save(output_dir + "/" + "p_x_profile" + ".csv",csv_ascii);
    p_r.save(output_dir + "/" + "p_r_profile" + ".csv",csv_ascii);
    KE.save(output_dir + "/" + "KE_profile" + ".csv",csv_ascii);
  }

  // STEP 5:
  // =====================================================================================
  // Resample distribution:
  // =====================================================================================
  particle_tree.resample_distribution();

  // STEP 6:
  // =====================================================================================
  // Save resampled distribution to file:
  // =====================================================================================
  // Rescale:
  x_p*= x_norm;
  v_p*= y_norm;

  // Save resampled distribution for post-processing
  file_type format = csv_ascii;
  x_p.save(output_dir + "/"  + "x_p_new.csv",csv_ascii);
  v_p.save(output_dir + "/"  + "v_p_new.csv",csv_ascii);
  a_p.save(output_dir + "/"  + "a_p_new.csv",csv_ascii);

  // STEP 7:
  // =====================================================================================
  // Save data from tree to be post-processed in MATLAB:
  // =====================================================================================
  // For every leaf_x node, there is a corresponding quadtree in velocity space.
  // From each of those quadtrees, we have extracted leaf_v which contains a list of all the leaf nodes in velocity space for a given "x" location.
  // What we are doing in this section is to save the particle count, coordinates and dimensions of the leaf_v nodes on each "x" location,
  // This data can then be plotted in MATLAB as a group of nested bisected quadrants in velocity space and then we can superimpose the particle data to confirm whether of not the quadtree algorithm is working.
  // We can visually inspect of the particle count, coordinate and dimensions of the node match the observed number of particles in that region based on the particle data.

  // Save data to postprocess tree:
  int Nx = bt_params.num_nodes;
  for (int xx = 0; xx < Nx ; xx++)
  {
    if (particle_tree.leaf_v[xx][0] != NULL)
    {
      // Get the number of leaf nodes on current leaf_v:
      int num_v_nodes = particle_tree.leaf_v[xx].size();

      // Create variables to contain data:
      ivec node_ip_count(num_v_nodes);
      mat  node_center(num_v_nodes,2);
      mat  node_dim(num_v_nodes,2);

      // Assemble data:
      for (int vv = 0; vv < num_v_nodes; vv++)
      {
        // cout << "ip_count = " << leaf_v[xx][vv]->ip_count << endl;
        node_ip_count(vv)   = particle_tree.leaf_v[xx][vv]->ip_count;
        node_center.row(vv) = particle_tree.leaf_v[xx][vv]->center.t();
        node_dim.row(vv)    = particle_tree.leaf_v[xx][vv]->max.t() - particle_tree.leaf_v[xx][vv]->min.t();
      }

      // Save data:
      string file_name;
      file_name = output_dir + "/"  + "leaf_v_" + "ip_count" + "_xx_" + to_string(xx) + ".csv";
      node_ip_count.save(file_name, csv_ascii);

      file_name = output_dir + "/"  + "leaf_v_" + "node_center" + "_xx_" + to_string(xx) + ".csv";
      node_center.save(file_name, csv_ascii);

      file_name = output_dir + "/"  + "leaf_v_" + "node_dim" + "_xx_" + to_string(xx) + ".csv";
      node_dim.save(file_name, csv_ascii);
    }
  }

  // STEP 8:
  // =====================================================================================
  // Assess conservation: AFTER RESAMPLING
  // =====================================================================================
  // Since we have resampled the data, in order to assess conservation, we need clear contents of trees and re-populate them

  // Clear contents of particle tree:
  particle_tree.clear_all_contents();

  if (false)
  {
    // Load a new data set:
    load_input_data(root_input, picos_case, species_index, time_index,&x_p, &v_p, &a_p);
  }

  // Normalize data:
  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // Repopulate trees with resampled data:
  particle_tree.populate_tree();

  // Save output:
  particle_tree.ip_count.save(output_dir + "/"  + "leaf_x_ip_count_new" + ".csv", csv_ascii);

  // STEP 9:
  // =====================================================================================
  // Assess conservation: AFTER RESAMPLING
  // =====================================================================================
  {
    int Nx = bt_params.num_nodes;
    vec m_t(Nx, fill::zeros);
    vec p_x(Nx, fill::zeros);
    vec p_r(Nx, fill::zeros);
    vec KE(Nx, fill::zeros);

    for (int xx = 0; xx < Nx ; xx++)
    {
      if (particle_tree.leaf_x[xx] != NULL)
      {
        uvec ip = conv_to<uvec>::from(particle_tree.leaf_x[xx]->ip);
        m_t[xx] = sum(a_p.elem(ip));
        mat v_p_subset = v_p.rows(ip);
        p_x[xx] = dot(a_p.elem(ip),v_p_subset.col(0));
        p_r[xx] = dot(a_p.elem(ip),v_p_subset.col(1));

        KE[xx] = (dot(a_p.elem(ip),pow(v_p_subset.col(0),2)) + dot(a_p.elem(ip),pow(v_p_subset.col(1),2)));
      }
    }

    cout << "Total KE after resampling = " + to_string(sum(KE)) << endl;

    m_t.save(output_dir + "/" + "m_new_profile" + ".csv",csv_ascii);
    p_x.save(output_dir + "/" + "p_x_new_profile" + ".csv",csv_ascii);
    p_r.save(output_dir + "/" + "p_r_new_profile" + ".csv",csv_ascii);
    KE.save(output_dir + "/" + "KE_new_profile" + ".csv",csv_ascii);
  }

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
  particle_tree.clear_all_contents();

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
  particle_tree.populate_tree();

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
