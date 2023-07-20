#include <iostream>
#include <armadillo>
// #include <chrono>
#include <string>
// #include <sstream>
// #include <cstdlib>
// #include <ctime>
#include <filesystem>
#include <vector>

// #include "BinaryTree.h"
// #include "QuadTree.h"
#include "particle_tree.h"

using namespace std::chrono;
using namespace std;
using namespace arma;
namespace fs = filesystem;

/* OBJECTIVE:

*/

/* HOW TO USE THIS SCRIPT:

*/

int main()
{
  // Choose PICOS++ case:
  // =====================================================================================
  string picos_case = "PICOS_case_2";
  string species_index = "2"; // ss
  string time_index    = "50"; // tt
  string scenario      = "ss_" + species_index + "_tt_" + time_index;

  // Choose the input data:
  // =====================================================================================
  string root_input = "./Step_1_output/";

  // Folder where output data is to be stored:
  // =====================================================================================
  string root_output = "./Step_2_output/" + picos_case + "/" + scenario;
  if (fs::is_directory(root_output) == true)
  {
    fs::remove_all(root_output);
  }

  if (fs::is_directory(root_output) == false)
  {
    fs::create_directories(root_output);
  }

  // STEP 1:
  // =====================================================================================
  // Load input data:
  // =====================================================================================
  // We have three data sets: x_p, v_p and a_p
  // We will use a binary tree to index the x_p data and produce a leaf_x vector
  // For each element in leaf_x vector which has a surplus, we will use a quadtree to index the v_p data and produce a leaf_v 2D vector

  arma::vec x_p;
  arma::mat v_p;
  arma::vec a_p;

  string input_file_name_1 = root_input + picos_case + "/" + "x_p" + "_ss_" + species_index + "_tt_" + time_index + ".csv";
  string input_file_name_2 = root_input + picos_case + "/" + "v_p" + "_ss_" + species_index + "_tt_" + time_index + ".csv";
  string input_file_name_3 = root_input + picos_case + "/" + "a_p" + "_ss_" + species_index + "_tt_" + time_index + ".csv";

  x_p.load(input_file_name_1,csv_ascii);
  v_p.load(input_file_name_2,csv_ascii);
  a_p.load(input_file_name_3,csv_ascii);

  // Normalize data:
  // -------------------------------------------------------------------------------------
  double y_norm = max(max(v_p));
  if (species_index == "1"){y_norm = 2e6;}
  double x_norm = ceil(max(x_p));

  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // STEP 2:
  // =====================================================================================
  // Define tree parameters:
  // =====================================================================================

  // 1D binary tree parameters:
  // -------------------------------------------------------------------------------------
  tree_params_TYP tree_params;

  tree_params.dimensionality = 1;
  tree_params.min       = {-1};
  tree_params.max       = {+1};
  tree_params.max_depth = {+6};

  // 2D Quadtree parameters:
  // -------------------------------------------------------------------------------------
  quadTree_params_TYP quadTree_params;

  quadTree_params.min = {-1,-1};
  quadTree_params.max = {+1,+1};
  if (species_index == "1")
  {
    quadTree_params.min_depth = +5;
    quadTree_params.max_depth = +6;
    quadTree_params.min_count = 7;
  }
  else if (species_index == "2")
  {
    quadTree_params.min_depth = +5;
    quadTree_params.max_depth = +6;
    quadTree_params.min_count = 7;
  }

  // STEP 3:
  // =====================================================================================
  // Create and populate tree:
  // =====================================================================================
  // Create composite particle tree:
  // -------------------------------------------------------------------------------------
  particle_tree_TYP particle_tree(&tree_params,&quadTree_params,&x_p,&v_p,&a_p);

  // Populate particle tree with data:
  // -------------------------------------------------------------------------------------
  // particle_tree.compute_bt();
  // particle_tree.compute_qt();
  particle_tree.populate_tree();

  // Binary tree diagnostics:
  // -------------------------------------------------------------------------------------
  int k = particle_tree.binary_tree.count_leaf_nodes();
  cout << "total number of leaf nodes populated: " << k << endl;
  k = particle_tree.binary_tree.count_leaf_points();
  cout << "total number of leaf points inserted: " << k << endl;

  // Save leaf_x p_count profile:
  // -------------------------------------------------------------------------------------
  particle_tree.xq.save(root_output + "/"  + "x_q" + ".csv", arma::csv_ascii);
  particle_tree.p_count.save(root_output + "/"  + "leaf_x_p_count" + ".csv", arma::csv_ascii);

  // Quad tree diagnostics:
  // -------------------------------------------------------------------------------------
  {
    vec qt_count = zeros<vec>(particle_tree.Nx);
    vec bt_count = zeros<vec>(particle_tree.Nx);
    for (int xx = 0; xx < particle_tree.Nx; xx++)
    {
      qt_count[xx] = particle_tree.quad_tree[xx].count_leaf_points();
      if (particle_tree.quad_tree[xx].root != NULL)
        bt_count[xx] = particle_tree.p_count[xx];
    }

    cout << "Total number of particles indexed in quad tree (binary tree) " << sum(bt_count) << endl;

    cout << "Total number of particles indexed in quad tree (quad tree) " << sum(qt_count) << endl;
  }

  // STEP 4:
  // =====================================================================================
  // Assess conservation:PRIOR TO RESAMPLING
  // =====================================================================================
  {
    int Nx = particle_tree.Nx;
    arma::vec m_t(Nx);
    arma::vec p_x(Nx);
    arma::vec p_r(Nx);
    arma::vec KE(Nx);

    for (int xx = 0; xx < Nx ; xx++)
    {
      uvec ip = conv_to<uvec>::from(particle_tree.leaf_x[xx]->ip);
      m_t[xx] = sum(a_p.elem(ip));
      mat v_p_subset = v_p.rows(ip);
      p_x[xx] = dot(a_p.elem(ip),v_p_subset.col(0));
      p_r[xx] = dot(a_p.elem(ip),v_p_subset.col(1));

      KE[xx] = (dot(a_p.elem(ip),pow(v_p_subset.col(0),2)) + dot(a_p.elem(ip),pow(v_p_subset.col(1),2)));
    }

    cout << "Total KE before resampling = " + to_string(sum(KE)) << endl;

    m_t.save(root_output + "/" + "m_profile" + ".csv",csv_ascii);
    p_x.save(root_output + "/" + "p_x_profile" + ".csv",csv_ascii);
    p_r.save(root_output + "/" + "p_r_profile" + ".csv",csv_ascii);
    KE.save(root_output + "/" + "KE_profile" + ".csv",csv_ascii);
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
  arma::file_type format = arma::csv_ascii;
  x_p.save(root_output + "/"  + "x_p_new.csv",csv_ascii);
  v_p.save(root_output + "/"  + "v_p_new.csv",csv_ascii);
  a_p.save(root_output + "/"  + "a_p_new.csv",csv_ascii);

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
  for (int xx = 0; xx < particle_tree.leaf_x.size() ; xx++)
  {
    if (particle_tree.leaf_v[xx][0] != NULL)
    {
      // Create variables to contain data:
      ivec node_p_count(particle_tree.leaf_v[xx].size());
      mat  node_center(particle_tree.leaf_v[xx].size(),2);
      mat  node_dim(particle_tree.leaf_v[xx].size(),2);

      // Assemble data:
      for (int vv = 0; vv < particle_tree.leaf_v[xx].size(); vv++)
      {
        // cout << "p_count = " << leaf_v[xx][vv]->p_count << endl;
        node_p_count(vv)  = particle_tree.leaf_v[xx][vv]->p_count;
        node_center.row(vv) = particle_tree.leaf_v[xx][vv]->center.t();
        node_dim.row(vv)    = particle_tree.leaf_v[xx][vv]->max.t() - particle_tree.leaf_v[xx][vv]->min.t();
      }

      // Save data:
      string file_name;
      file_name = root_output + "/"  + "leaf_v_" + "p_count" + "_xx_" + to_string(xx) + ".csv";
      node_p_count.save(file_name, arma::csv_ascii);

      file_name = root_output + "/"  + "leaf_v_" + "node_center" + "_xx_" + to_string(xx) + ".csv";
      node_center.save(file_name, arma::csv_ascii);

      file_name = root_output + "/"  + "leaf_v_" + "node_dim" + "_xx_" + to_string(xx) + ".csv";
      node_dim.save(file_name, arma::csv_ascii);
    }
  }

  // STEP 8:
  // =====================================================================================
  // Assess conservation: AFTER RESAMPLING
  // =====================================================================================
  // Since we have resampled the data, in order to assess conservation, we need clear contents of trees and re-populate them

  // Clear contents of binary tree:
  particle_tree.binary_tree.clear_all();
  for (int xx = 0; xx < particle_tree.Nx; xx++)
  {
    particle_tree.quad_tree[xx].clear_tree();
  }

  if (false)
  {
    // Load a new data set:
    x_p.load(input_file_name_1,csv_ascii);
    v_p.load(input_file_name_2,csv_ascii);
    a_p.load(input_file_name_3,csv_ascii);
  }

  // Normalize data:
  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // Repopulate trees with resampled data:
  particle_tree.populate_tree();

  // Save output:
  particle_tree.p_count.save(root_output + "/"  + "leaf_x_p_count_new" + ".csv", arma::csv_ascii);

  // STEP 9:
  // =====================================================================================
  // Assess conservation: AFTER RESAMPLING
  // =====================================================================================
  {
    int Nx = particle_tree.Nx;
    arma::vec m_t(Nx);
    arma::vec p_x(Nx);
    arma::vec p_r(Nx);
    arma::vec KE(Nx);
    for (int xx = 0; xx < particle_tree.Nx ; xx++)
    {
      uvec ip = conv_to<uvec>::from(particle_tree.leaf_x[xx]->ip);
      m_t[xx] = sum(a_p.elem(ip));
      mat v_p_subset = v_p.rows(ip);
      p_x[xx] = dot(a_p.elem(ip),v_p_subset.col(0));
      p_r[xx] = dot(a_p.elem(ip),v_p_subset.col(1));

      KE[xx] = (dot(a_p.elem(ip),pow(v_p_subset.col(0),2)) + dot(a_p.elem(ip),pow(v_p_subset.col(1),2)));
    }

    cout << "Total KE after resampling = " + to_string(sum(KE)) << endl;

    m_t.save(root_output + "/" + "m_new_profile" + ".csv",csv_ascii);
    p_x.save(root_output + "/" + "p_x_new_profile" + ".csv",csv_ascii);
    p_r.save(root_output + "/" + "p_r_new_profile" + ".csv",csv_ascii);
    KE.save(root_output + "/" + "KE_new_profile" + ".csv",csv_ascii);
  }

  // At this point, we have succesfully demonstrated how to apply the trees and resample the distribution and then update the tree.

  // We now need to demonstrate that we can clear the tree data, release the tree memory with a delete function and reuse the tree without incurring into memory leaks

  // STEP XX:
  // =====================================================================================
  // Test clearing data:
  // =====================================================================================

  // Clear contents of binary tree:
  particle_tree.binary_tree.clear_all();

  // Clear contents of quad tree:
  for (int xx = 0; xx < particle_tree.Nx; xx++)
  {
    particle_tree.quad_tree[xx].clear_tree();
  }

  // The above effectively only clears the value of p_count. This is becuase everytime we create a new subnode, the values of ip are passed onto the new subnode and ip on the parent node is cleared.
  // It is quite possible that we do not need to use this function. Instead we could just clear p_count at the same time we clear ip upon creating a new subnode.

  /*
  // STEP XX:
  // =====================================================================================
  // Test deleting tree and releasing memory:
  // =====================================================================================
  // Deleting entire binary tree:
  // particle_tree.binary_tree.delete_nodes();

  // Deleting the xxth quadtree:
  // particle_tree.quad_tree[xx].delete_tree();

  // Re-analize the data to test reusing tree infrastructure:
  // ======================================================================
  // Since we have resampled the data, in order to assess conservation, we need clear contents of trees and re-populate them

  // Clear contents of binary tree:
  particle_tree.binary_tree.clear_all();
  for (int xx = 0; xx < particle_tree.Nx; xx++)
  {
    particle_tree.quad_tree[xx].clear_tree();
  }

  // Load a new data set:
  x_p.load(input_file_name_1,csv_ascii);
  v_p.load(input_file_name_2,csv_ascii);
  a_p.load(input_file_name_3,csv_ascii);

  // Normalize data:
  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // Repopulate trees with resampled data:
  particle_tree.populate_tree();

  // Save output:
  particle_tree.p_count.save(root_output + "/"  + "leaf_x_p_count_new" + ".csv", arma::csv_ascii);

  */


  return 0;
}
