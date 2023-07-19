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
  // In this example, a_p data set is not used as we are not focusing on re-sampling. This will be dealt with in another script.

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
  // Assemble and populate tree:
  // =====================================================================================

  // 1D binary tree:
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

  // Create composite particle tree:
  // -------------------------------------------------------------------------------------
  particle_tree_TYP particle_tree(&tree_params,&quadTree_params,&x_p,&v_p,&a_p);

  // Populate particle tree with data:
  // -------------------------------------------------------------------------------------
  particle_tree.populate_tree();

  // Diagnostics:
  // -------------------------------------------------------------------------------------
  int k = particle_tree.binary_tree.count_leaf_nodes();
  cout << "total number of leaf nodes populated: " << k << endl;
  k = particle_tree.binary_tree.count_leaf_points();
  cout << "total number of leaf points inserted: " << k << endl;

  // Save leaf_x p_count profile:
  // -------------------------------------------------------------------------------------
  particle_tree.xq.save(root_output + "/"  + "x_q" + ".csv", arma::csv_ascii);
  particle_tree.p_count.save(root_output + "/"  + "leaf_x_p_count" + ".csv", arma::csv_ascii);

  // STEP 3:
  // =====================================================================================
  // Assess conservation:
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

  // STEP 4:
  // =====================================================================================
  // Resample distribution:
  // =====================================================================================
  particle_tree.resample_distribution();

  // Rescale:
  x_p*= x_norm;
  v_p*= y_norm;

  // Save resampled distribution for post-processing
  arma::file_type format = arma::csv_ascii;
  x_p.save(root_output + "/"  + "x_p_new.csv",csv_ascii);
  v_p.save(root_output + "/"  + "v_p_new.csv",csv_ascii);
  a_p.save(root_output + "/"  + "a_p_new.csv",csv_ascii);

 /*
  vranic_TYP vranic;
  std::vector<uint> ip_free;
  int N_min = 7;
  int N_max = 300;
  int N;
  int M = 6;
  int particle_surplus;
  arma::file_type format = arma::csv_ascii;
  vector<int> leaf_v_p_count;
  int particle_deficit;
  int exit_flag_v = 0;

  // Down-sample surplus nodes:
  // -------------------------------------------------------------------------------------
  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Clear flag:
    exit_flag_v = 0;

    // Calculate particle surplus
    particle_surplus = leaf_x[xx]->p_count - mean_p_count;

    if (leaf_v[xx][0] != NULL)
    {
      // Assemble depth vector:
      int Nv = leaf_v[xx].size();
      vec depth(Nv);
      vec p_count(Nv);
      for(int vv = 0; vv < Nv; vv++){depth(vv) = leaf_v[xx][vv]->depth;}
      for(int vv = 0; vv < Nv; vv++){p_count(vv) = leaf_v[xx][vv]->p_count;}

      // Create sorted list starting from highest depth to lowest:
      uvec sorted_index_list = arma::sort_index(depth,"descend");
      // uvec sorted_index_list = arma::sort_index(p_count,"descend");

      // Loop over sorted leaf_v and apply vranic method:
      for (int vv = 0; vv < sorted_index_list.n_elem; vv++)
      {
        if (exit_flag_v == 1)
          break;

        // Sorted index:
        int tt = sorted_index_list(vv);

        // Diagnostics:
        // {
        //   cout << "density = " << leaf_v[xx][tt]->p_count << endl;
        //   cout << "depth = " << leaf_v[xx][tt]->depth << endl;
        //   leaf_v[xx][tt]->center.print("center = ");
        // }

        // Total number of particles in leaf_v cube:
        N = leaf_v[xx][tt]->p_count;

        // Test if current node is suitable for resampling:
        if (N < N_min){continue;}

        // Upper limit to the number of particles to down-sample in present cell:
        if (N > N_max){N = N_max;}

        // Exit condition:
        if (particle_surplus - (N-M) < 0)
        {
          N = particle_surplus + M;
        }

        // Test if current node is suitable for resampling:
        if (N < N_min){continue;}

        // Create objects for down-sampling:
        merge_cell_TYP set_N(N);
        merge_cell_TYP set_M(M);

        // Define set N particles:
        arma::uvec ip = conv_to<uvec>::from(leaf_v[xx][tt]->ip);
        arma::mat v_p_subset = v_p.rows(ip.head(N));
        set_N.xi = x_p.elem(ip.head(N));
        set_N.yi = v_p_subset.col(0);
        set_N.zi = v_p_subset.col(1);
        set_N.wi = a_p.elem(ip.head(N));

        // Calculate set M based on set N:
        // vranic.down_sample_3D(&set_N, &set_M);
        vranic.down_sample_2D(&set_N, &set_M);

        // Diagnostics:
        if (false)
        {
          // Print statistics:
          cout << "Set N: " << endl;
          vranic.print_stats(&set_N);

          // Print statistics:
          cout << "Set M: " << endl;
          vranic.print_stats(&set_M);
        }

        // Diagnostics:
        double ratio;
        if (true)
        {
          // Print statistics:
          double sigma_N = vranic.get_sigma(&set_N);
          // cout << "Set N: sigma_r = " << sigma_N << endl;

          // Print statistics:
          double sigma_M = vranic.get_sigma(&set_M);
          // cout << "Set M: sigma_r = " << sigma_M << endl;

          ratio = sigma_M/sigma_N;
          // cout << "sqrt(ratio of M/N) = " << sqrt(ratio) << endl;
          // cout << " " << endl;
        }

        // Apply changes to distribution function:
        for (int ii = 0; ii < N; ii++)
        {
          // Get global index:
          int jj = ip(ii);

          // Apply changes to x_p, v_p, a_p OR create new mem locations:
          if (ii < set_M.n_elem)
          {
            // Down-sample global distribution:
            // Use set_N for xi in order to remove oscillations in density:
            // Use set_M for all other quantities (v and a):

            // Set N:
            x_p(jj) = set_N.xi(ii);

            // Set M:
            // x_p(jj)   = set_M.xi(ii);
            v_p(jj,0) = set_M.yi(ii);
            v_p(jj,1) = set_M.zi(ii);
            a_p(jj)   = set_M.wi(ii);

          }
          else
          {
            // Record global indices that correspond to memory locations that are to be repurposed:
            ip_free.push_back(jj);

            // Set values to -1 to flag them as undefined values memory locations:
            x_p(jj)   = -1;
            v_p(jj,0) = -1;
            v_p(jj,1) = -1;
            a_p(jj)   = -1;

            // Decrement particle surplus only after ip_free.push_back operation:
            particle_surplus--;
          }
        } // for ip loop
      } // sorted index Loop
    } // if not NULL
  } // xx loop

  // Populate deficit regions:
  // -------------------------------------------------------------------------------------
  // Initialize replication flag:
  int ip_free_flag = 0;

  // Vector to keep track of particle counts in nodes:
  ivec node_counts(Nx);
  for (int xx = 0; xx < Nx; xx++)
    node_counts(xx) = leaf_x[xx]->p_count;

  // Vector to keep track how many times we need to replicate particles per node:
  ivec rep_num_vec(Nx);
  for (int xx = 0; xx < Nx; xx++)
    rep_num_vec(xx) = ceil((double)mean_p_count/node_counts(xx));

  // Layers:
  vec layer_fraction = {1/2, 1/4, 1/8, 1/16, 1/16};
  ivec layer(5);
  layer(0) = (int)round((double)mean_p_count/2);
  layer(1) = (int)round((double)mean_p_count/4);
  layer(2) = (int)round((double)mean_p_count/8);
  layer(3) = (int)round((double)mean_p_count/16);
  layer(4) = mean_p_count - sum(layer.subvec(0,layer.n_elem - 2));

  for (int ll = 0; ll < layer.n_elem; ll++)
  {
    cout << "ll = " << ll << endl;
    for (int xx = 0; xx < Nx ; xx++)
    {
      // If ip_free is empty, then stop replication:
      if (ip_free_flag == 1)
        break;

      // Calculate particle deficit
      int target_counts = sum(layer.subvec(0,ll));
      particle_deficit = node_counts(xx) - target_counts;

      if (particle_deficit < 0)
      {
        // Number of particles to replicate (original particles in node):
        int num_0 = leaf_x[xx]->p_count;

        // Replication number: represents how many times a particle needs to be replicated
        // rep_num - 1 gives you the number of new particles per parent particle
        // int rep_num = ceil((double)layer(ll)/num_0);
        int rep_num = rep_num_vec(xx);

        // Loop over all particles in node:
        for (int ii = num_0 - 1; ii >= 0; ii--)
        {
          // Get global index of particle to replicate:
          uint jj = leaf_x[xx]->ip[ii];

          // On the last layer, remove the indexes since they have been fully used:
          if (ll == layer.n_elem)
          {
            leaf_x[xx]->ip.pop_back();
            leaf_x[xx]->p_count--;
          }

          // Diagnostics:
          if (leaf_x[xx]->p_count < 0)
            cout << "error:" << endl;

          // Parent particle attributes:
          double xi = x_p(jj);
          double yi = v_p(jj,0);
          double zi = v_p(jj,1);
          double wi = a_p(jj);

          // Calculate number of new daughter particles to create:
          int num_free_left = ip_free.size();
          int num_deficit_left = -particle_deficit;
          int num_requested = rep_num - 1;
          ivec num_vec = {num_requested, num_free_left, num_deficit_left};
          int num_new = num_vec(num_vec.index_min());

          // Create num_new daughter particles:
          for (int rr = 0; rr < num_new; rr++)
          {
            uint jj_free = ip_free.back();
            ip_free.pop_back();
            x_p(jj_free)   = xi; // - sign(xi)*0.01;
            v_p(jj_free,0) = yi;
            v_p(jj_free,1) = zi;
            a_p(jj_free)   = wi/((double)num_new + 1.0);

            // Modify deficit:
            particle_deficit++;
            node_counts(xx)++;
          }

          // Adjust weight of parent particle to conserve mass:
          a_p(jj) = wi/((double)num_new + 1.0);

          // if size of ip_free vanishes, then stop all replication:
          int num_free = ip_free.size();
          if (num_free == 0)
          {
            ip_free_flag = 1;
            break;
          }

          // If deficit becomes +ve, then stop:
          if (particle_deficit >= 0)
          {
            break;
          }
        } // particle Loop

        if (particle_deficit != 0)
        {
          //abort();
        }

      } // deficit if
    } // xx loop
  } // ll Loop

  // Rescale:
  x_p*= x_norm;
  v_p*= y_norm;

  // Save resampled distribution for post-processing
  x_p.save(root_output + "/"  + "x_p_new.csv",csv_ascii);
  v_p.save(root_output + "/"  + "v_p_new.csv",csv_ascii);
  a_p.save(root_output + "/"  + "a_p_new.csv",csv_ascii);
  */

  // STEP 6:
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

  // return 0;

  // STEP 7:
  // =====================================================================================
  // Test counting points in quadtree:
  // =====================================================================================
  {
    vec qt_count = zeros<vec>(particle_tree.Nx);
    vec bt_count = zeros<vec>(particle_tree.Nx);
    for (int xx = 0; xx < particle_tree.Nx; xx++)
    {
      qt_count[xx] = particle_tree.quad_tree[xx].count_leaf_points();
      if (particle_tree.quad_tree[xx].root != NULL)
        bt_count[xx] = particle_tree.p_count[xx];
    }

    cout << "Total number of particles in sim (binary tree) " << sum(bt_count) << endl;

    cout << "Total number of particles in sim (quad tree) " << sum(qt_count) << endl;
  }

  // STEP 8:
  // =====================================================================================
  // Test clearing data:
  // =====================================================================================
  int xx = 24;
  particle_tree.quad_tree[xx].clear_tree();

  // The above effectively only clears the value of p_count. This is becuase everytime we create a new subnode, the values of ip are passed onto the new subnode and ip on the parent node is cleared.
  // It is quite possible that we do not need to use this function. Instead we could just clear p_count at the same time we clear ip upon creating a new subnode.

  // STEP 9:
  // =====================================================================================
  // Test deleting tree and releasing memory:
  // =====================================================================================
  // Deleting entire binary tree:
  // particle_tree.binary_tree.delete_nodes();

  // Deleting the xxth quadtree:
  // particle_tree.quad_tree[xx].delete_tree();

  // Re-analize the modified data:
  // ======================================================================
  // Since we have resampled the data, in order to assess conservation, we need clear contents of trees and re-populate them

  // Normalized data:
  x_p/= x_norm;
  v_p/= y_norm;

  // Clear contents of binary tree:
  particle_tree.binary_tree.clear_all();
  for (int xx = 0; xx < particle_tree.Nx; xx++)
  {
    particle_tree.quad_tree[xx].clear_tree();
  }

  // Repopulate trees with resampled data:
  particle_tree.populate_tree();

  // Save output:
  particle_tree.p_count.save(root_output + "/"  + "leaf_x_p_count_new" + ".csv", arma::csv_ascii);

  // STEP 10:
  // =====================================================================================
  // Assess conservation:
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

  return 0;
}
