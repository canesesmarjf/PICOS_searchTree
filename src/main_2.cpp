#include <iostream>
#include <armadillo>
#include <chrono>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <vector>

#include "BinaryTree.h"
#include "QuadTree.h"
#include "Vranic.h"

using namespace std::chrono;
using namespace std;
using namespace arma;
namespace fs = filesystem;

/* OBJECTIVE:
We have created this script based on main_1.cpp and modify it to save the data in the directories required for another project directory structure:

describe procudure tp use thise script and needed directires and worfkflow

*/

/* HOW TO USE THIS SCRIPT:
This script belongs to the PICOS_searchTree repo.
In its current use, this script is ran in the folowing directory:

/home/jfcm/Documents/compX/ARPAE/Effort_2_Vranic/Step_3_Vranic_improved_resampling

This script is part of a workflow:
0 - pwd needs to be the following directory:
  + /home/jfcm/Documents/compX/ARPAE/Effort_2_Vranic/Step_3_Vranic_improved_resampling

1 - Step_1_PreviewPICOSdata.m: Extracts data from HDF5 files and produces .csv files
  + There are at least three source of data:
    > PICOS_case_0
    > PICOS_case_1
    > PICOS_case_2
  + Each case has several time steps (tt) and species (ss) to select From

2 - Step_2 is to run this script and read the .csv files produced by step 1 and carries out the binary and quadtrees and produces a resampled distribution in .csv files and a set of files which describe the indexing performed by the search trees

3 - Step_3_PostProcess_Search_Tree.m, reads the output produced by Step_2 and checks results of tree indexing with actual particle data, it also performs a conservation checks

4 - Step_4_PostProcess_moments.m, calculates particle moments for original and resampled distirbution
*/

int main()
{
  // Choose PICOS++ case:
  // =====================================================================================
  string picos_case = "PICOS_case_2";
  string species_index = "1"; // ss
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

  vec x_p;
  mat v_p;
  vec a_p;

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
  // Assemble binary tree for x data and produce leaf_x vector
  // =====================================================================================

  // Start with a 1D binary tree:
  // -------------------------------------------------------------------------------------
  bt_params_TYP bt_params;

  bt_params.dimensionality = 1;
  bt_params.min       = {-1};
  bt_params.max       = {+1};
  bt_params.max_depth = {+5+1};

  // Create a vector with pointers to the data:
  // -------------------------------------------------------------------------------------
  vector<vec *> x_data = {&x_p};

  // Create binary tree based on parameters:
  // -------------------------------------------------------------------------------------
  bt_TYP tree(&bt_params);

  // Insert 1D points into tree:
  // -------------------------------------------------------------------------------------
  tree.insert_all(x_data);

  // Diagnostics:
  // -------------------------------------------------------------------------------------
  int k = tree.count_leaf_nodes();
  cout << "total number of leaf nodes populated: " << k << endl;
  k = tree.count_leaf_points();
  cout << "total number of leaf points inserted: " << k << endl;

  // Create grid for x dimension in binary tree:
  // -------------------------------------------------------------------------------------
  double Lx = bt_params.max[0] - bt_params.min[0];
  int Nx    = pow(2,bt_params.max_depth[0]);
  double dx = Lx/Nx;
  vec xq = bt_params.min[0] + dx/2 + regspace(0,Nx-1)*dx;

  // Calculate leaf_x list:
  // -------------------------------------------------------------------------------------
  std::vector<node_TYP *> leaf_x(Nx);
  ivec ip_count(Nx);
  for (int xx = 0; xx < Nx ; xx++)
  {
   leaf_x[xx] = tree.find(xq(xx));
   ip_count[xx] = leaf_x[xx]->ip_count;
  }

  // Calculating the mean number of particles per node:
  // -------------------------------------------------------------------------------------
  // This operation requires the division between two integers; hence, care needs to be taken since this leads to loss of precision:
  // To account for this, we need to round up to the nearest LARGEST integer so that we over-estimate total number of particles.
  // Underestimation leads to not being able to reuse all free memory locations later in the replication process.
  int p_count_sum = sum(ip_count);
  int mean_ip_count = ceil((double)p_count_sum/Nx);

  // Save leaf_x ip_count profile:
  // -------------------------------------------------------------------------------------
  xq.save(root_output + "/"  + "x_q" + ".csv", csv_ascii);
  ip_count.save(root_output + "/"  + "leaf_x_ip_count" + ".csv", csv_ascii);

  // STEP 3:
  // =====================================================================================
  // Assess conservation:
  // =====================================================================================

  vec m_t(Nx);
  vec p_x(Nx);
  vec p_r(Nx);
  vec KE(Nx);

  for (int xx = 0; xx < Nx ; xx++)
  {
    uvec ip = conv_to<uvec>::from(leaf_x[xx]->ip);
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

  // STEP 4:
  // =====================================================================================
  // Assemble quad-tree for v-space data and produce leaf_v 2D vector
  // =====================================================================================

  // In what follows, we are using a quadtree that differs somewhat from the binary tree used for the x data. The two main differences are the following:
  // - In binary tree, the root node is encapsulated inside a bt_TYP object; while, for the quadtree, the root node is an object of its own
  // - In binary tree, each particle present in x_p is inserted into the tree until it reaches is associted leaf and then it moves to the next particle. In the quadtree, all particles are inserted into the next depth and distributed amongst their respective subnodes. This allows one to track the particle number in each node as be used as stopping metric rather than the depth as in the binary tree for the xdata.

  // Quadtree parameters:
  // -------------------------------------------------------------------------------------
  qt_params_TYP qt_params;
  qt_params.min       = {-1,-1};
  qt_params.max       = {+1,+1};
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

  // Create quadtree vector for every leaf_x dataset:
  // -------------------------------------------------------------------------------------
  // For each leaf_x element, there will be a corresponding v-space quadtree:
  vector<qt_TYP> quadTree;
  quadTree.reserve(Nx);

  // Create 2D vector to store leaf_v:
  // -------------------------------------------------------------------------------------
  // leaf_v represents a 2D vector for the leaf nodes in the quadtree. The first dimension corresponds to each element of leaf_x vector or each quadtree. The second dimension corresponds to all the leaf_v nodes present in a given leaf_x_vector:
  vector<vector<q_node_TYP *>> leaf_v(Nx); // leaf_v[xx][rr]

  // Assemble quadtrees for each leaf_x element:
  // -------------------------------------------------------------------------------------
  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Create quadtree[xx] only if leaf_x[xx] is surplus:
    int delta_ip_count = ip_count[xx] - mean_ip_count;
    if (delta_ip_count > 0)
    {
      // Data for quadtree:
      vector<uint> ip = leaf_x[xx]->ip;

      // Create quadtree:
      quadTree.emplace_back(&qt_params,ip,&v_p);

      // Populate quadtree:
      quadTree[xx].populate_tree();
    }
    else
    {
      quadTree.emplace_back();
    }
  }

  // Assembling leaf_v 2D vector:
  // -------------------------------------------------------------------------------------
  // Constructing leaf_v should not be part of the quadtree class. The reason is that leaf_v is a collection of quadtrees. Such a collection of quadtrees needs to belong to the object which carries out the resampling.

  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Create quadtree[xx] only if leaf_x[xx] is surplus:
    int delta_ip_count = ip_count[xx] - mean_ip_count;
    if (delta_ip_count > 0)
    {
      // Extract all leaf nodes:
      leaf_v[xx] = quadTree[xx].get_leaf_nodes();

      // Diagnostics:
      {
      int sum = 0;
      for(int ll = 0; ll < leaf_v[xx].size(); ll++)
        sum = sum + leaf_v[xx][ll]->ip_count;
      int ip_count = quadTree[xx].root->count_leaf_points(0);

      // cout << "ip_count[xx] = " << ip_count[xx] << endl;
      // cout << "ip_count = " << ip_count << endl;
      // cout << "sum = " << sum << endl;
      }

    }
    else
    {
      leaf_v[xx] = {NULL};
    }
  }

  // STEP 5:
  // =====================================================================================
  // Apply vranic method and prioritize smaller nodes first:
  // =====================================================================================
  // This type of script does NOT belong to either the quadtree NOR the vranic method. This is because this process below relies on a collection of quadtrees. Each of those quadtrees is a collection of nodes in velocity space which become the inputs to the vranic downsampling method:

  // NOTE:
  // I have observed that the KE is not fully conserved using the vranic method. it is conserved to 1e-4, however, on closer inspection, it seems that I need to make a N to 4 vranic method so that we can exactly conserve KE.

  vranic_TYP vranic;
  std::vector<uint> ip_free;
  int N_min = 7;
  int N_max = 300;
  int N;
  int M = 6;
  int particle_surplus;
  file_type format = csv_ascii;
  vector<int> leaf_v_ip_count;
  int particle_deficit;
  int exit_flag_v = 0;

  // Down-sample surplus nodes:
  // -------------------------------------------------------------------------------------
  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Clear flag:
    exit_flag_v = 0;

    // Calculate particle surplus
    particle_surplus = leaf_x[xx]->ip_count - mean_ip_count;

    if (leaf_v[xx][0] != NULL)
    {
      // Assemble depth vector:
      int Nv = leaf_v[xx].size();
      vec depth(Nv);
      vec ip_count(Nv);
      for(int vv = 0; vv < Nv; vv++){depth(vv) = leaf_v[xx][vv]->depth;}
      for(int vv = 0; vv < Nv; vv++){ip_count(vv) = leaf_v[xx][vv]->ip_count;}

      // Create sorted list starting from highest depth to lowest:
      uvec sorted_index_list = sort_index(depth,"descend");
      // uvec sorted_index_list = sort_index(ip_count,"descend");

      // Loop over sorted leaf_v and apply vranic method:
      for (int vv = 0; vv < sorted_index_list.n_elem; vv++)
      {
        if (exit_flag_v == 1)
          break;

        // Sorted index:
        int tt = sorted_index_list(vv);

        // Diagnostics:
        // {
        //   cout << "density = " << leaf_v[xx][tt]->ip_count << endl;
        //   cout << "depth = " << leaf_v[xx][tt]->depth << endl;
        //   leaf_v[xx][tt]->center.print("center = ");
        // }

        // Total number of particles in leaf_v cube:
        N = leaf_v[xx][tt]->ip_count;

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
        uvec ip = conv_to<uvec>::from(leaf_v[xx][tt]->ip);
        mat v_p_subset = v_p.rows(ip.head(N));
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
    node_counts(xx) = leaf_x[xx]->ip_count;

  // Vector to keep track how many times we need to replicate particles per node:
  ivec rep_num_vec(Nx);
  for (int xx = 0; xx < Nx; xx++)
    rep_num_vec(xx) = ceil((double)mean_ip_count/node_counts(xx));

  // Layers:
  vec layer_fraction = {1/2, 1/4, 1/8, 1/16, 1/16};
  ivec layer(5);
  layer(0) = (int)round((double)mean_ip_count/2);
  layer(1) = (int)round((double)mean_ip_count/4);
  layer(2) = (int)round((double)mean_ip_count/8);
  layer(3) = (int)round((double)mean_ip_count/16);
  layer(4) = mean_ip_count - sum(layer.subvec(0,layer.n_elem - 2));

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
        int num_0 = leaf_x[xx]->ip_count;

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
            leaf_x[xx]->ip_count--;
          }

          // Diagnostics:
          if (leaf_x[xx]->ip_count < 0)
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
  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    if (leaf_v[xx][0] != NULL)
    {
      // Create variables to contain data:
      ivec particle_count(leaf_v[xx].size());
      mat node_center(leaf_v[xx].size(),2);
      mat node_dim(leaf_v[xx].size(),2);

      // Assemble data:
      for (int vv = 0; vv < leaf_v[xx].size(); vv++)
      {
        // cout << "ip_count = " << leaf_v[xx][vv]->ip_count << endl;
        particle_count(vv)  = leaf_v[xx][vv]->ip_count;
        node_center.row(vv) = leaf_v[xx][vv]->center.t();
        node_dim.row(vv)    = leaf_v[xx][vv]->max.t() - leaf_v[xx][vv]->min.t();
      }

      // Save data:
      string file_name;
      file_name = root_output + "/"  + "leaf_v_" + "ip_count" + "_xx_" + to_string(xx) + ".csv";
      particle_count.save(file_name, csv_ascii);

      file_name = root_output + "/"  + "leaf_v_" + "node_center" + "_xx_" + to_string(xx) + ".csv";
      node_center.save(file_name, csv_ascii);

      file_name = root_output + "/"  + "leaf_v_" + "node_dim" + "_xx_" + to_string(xx) + ".csv";
      node_dim.save(file_name, csv_ascii);
    }
  }

  // STEP 7:
  // =====================================================================================
  // Test clearing data:
  // =====================================================================================
  // int xx = 12;
  // quadTree[xx].clear_tree();

  // STEP 8:
  // =====================================================================================
  // Test deleting tree and releasing memory:
  // =====================================================================================
  // quadTree[xx].delete_tree();

  // Re-analize the modified data:
  // ======================================================================
  // Rescale:
  x_p/= x_norm;
  v_p/= y_norm;
  tree.clear_all();
  tree.insert_all(x_data);

  // Calculate leaf_x list:
  // ======================================================================
  ip_count.zeros();
  leaf_x.clear();
  for (int xx = 0; xx < Nx ; xx++)
  {
   leaf_x[xx] = tree.find(xq(xx));
   ip_count[xx] = leaf_x[xx]->ip_count;
  }

  ip_count.save(root_output + "/"  + "leaf_x_ip_count_new" + ".csv", csv_ascii);

  // STEP 9:
  // =====================================================================================
  // Assess conservation:
  // =====================================================================================
  for (int xx = 0; xx < Nx ; xx++)
  {
    uvec ip = conv_to<uvec>::from(leaf_x[xx]->ip);
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

  return 0;
}
