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
We explore how to insert ALL velocity space data one level at a time and stop process via density on velocity space square rather than on level
*/

int main()
{
  // Folder where output data is to be stored:
  // =====================================================================================
  string file_root = "./output_files/main_1/";
  if (fs::is_directory(file_root) == false)
  {
    fs::create_directory(file_root);
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

  x_p.load("input_files/Step_1_x_p.csv",csv_ascii);
  v_p.load("input_files/Step_1_v_p.csv",csv_ascii);
  a_p.load("input_files/Step_1_a_p.csv",csv_ascii);

  // Normalize data:
  // -------------------------------------------------------------------------------------
  double y_norm = max(max(v_p));
  double x_norm = 1;

  x_p = x_p/x_norm;
  v_p = v_p/y_norm;

  // STEP 2:
  // =====================================================================================
  // Assemble binary tree for x data and produce leaf_x vector
  // =====================================================================================

  // Start with a 1D binary tree:
  // -------------------------------------------------------------------------------------
  tree_params_TYP tree_params;

  tree_params.dimensionality = 1;
  tree_params.min       = {-1};
  tree_params.max       = {+1};
  tree_params.max_depth = {+5};

  // Create a vector with pointers to the data:
  // -------------------------------------------------------------------------------------
  vector<arma::vec *> x_data = {&x_p};

  // Create binary tree based on parameters:
  // -------------------------------------------------------------------------------------
  binaryTree_TYP tree(&tree_params);

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
  double Lx = tree_params.max[0] - tree_params.min[0];
  int Nx    = pow(2,tree_params.max_depth[0]);
  double dx = Lx/Nx;
  arma::vec xq = tree_params.min[0] + dx/2 + regspace(0,Nx-1)*dx;

  // Calculate leaf_x list:
  // -------------------------------------------------------------------------------------
  std::vector<node_TYP *> leaf_x(Nx);
  arma::ivec p_count(Nx);
  for (int xx = 0; xx < Nx ; xx++)
  {
   leaf_x[xx] = tree.find(xq(xx));
   p_count[xx] = leaf_x[xx]->p_count;
  }
  int mean_p_count = mean(p_count);

  // Save leaf_x p_count profile:
  // -------------------------------------------------------------------------------------
  xq.save(file_root + "x_q" + ".csv", arma::csv_ascii);
  p_count.save(file_root + "leaf_x_p_count" + ".csv", arma::csv_ascii);

  // STEP 2:
  // =====================================================================================
  // Assemble quad-tree for v-space data and produce leaf_v 2D vector
  // =====================================================================================

  // In what follows, we are using a quadtree that differs somewhat from the binary tree used for the x data. The two main differences are the following:
  // - In binary tree, the root node is encapsulated inside a binaryTree_TYP object; while, for the quadtree, the root node is an object of its own
  // - In binary tree, each particle present in x_p is inserted into the tree until it reaches is associted leaf and then it moves to the next particle. In the quadtree, all particles are inserted into the next depth and distributed amongst their respective subnodes. This allows one to track the particle number in each node as be used as stopping metric rather than the depth as in the binary tree for the xdata.

  // Quadtree parameters:
  // -------------------------------------------------------------------------------------
  quadTree_params_TYP quadTree_params;
  quadTree_params.min       = {-1,-1};
  quadTree_params.max       = {+1,+1};
  quadTree_params.max_depth = {+6};
  quadTree_params.min_count = {+36};

  // Create quadtree vector for every leaf_x dataset:
  // -------------------------------------------------------------------------------------
  // For each leaf_x element, there will be a corresponding v-space quadtree:
  vector<quadTree_TYP> quadTree;
  quadTree.reserve(Nx);

  // Create 2D vector to store leaf_v:
  // -------------------------------------------------------------------------------------
  // leaf_v represents a 2D vector for the leaf nodes in the quadtree. The first dimension corresponds to each element of leaf_x vector or each quadtree. The second dimension corresponds to all the leaf_v nodes present in a given leaf_x_vector:
  vector<vector<quadNode_TYP *>> leaf_v(Nx); // leaf_v[xx][rr]

  // Assemble quadtrees for each leaf_x element:
  // -------------------------------------------------------------------------------------
  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Create quadtree[xx] only if leaf_x[xx] is surplus:
    int delta_p_count = p_count[xx] - mean_p_count;
    if (delta_p_count > 0)
    {
      // Data for quadtree:
      vector<uint> ip = leaf_x[xx]->ip;

      // Create quadtree:
      quadTree.emplace_back(&quadTree_params,ip,&v_p);

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
    int delta_p_count = p_count[xx] - mean_p_count;
    if (delta_p_count > 0)
    {
      // Extract all leaf nodes:
      leaf_v[xx] = quadTree[xx].get_leaf_nodes();

      // Diagnostics:
      {
      int sum = 0;
      for(int ll = 0; ll < leaf_v[xx].size(); ll++)
        sum = sum + leaf_v[xx][ll]->p_count;
      int ip_count = quadTree[xx].root->count_leaf_points(0);

      cout << "p_count[xx] = " << p_count[xx] << endl;
      cout << "ip_count = " << ip_count << endl;
      cout << "sum = " << sum << endl;
      }

    }
    else
    {
      leaf_v[xx] = {NULL};
    }
  }

  // STEP X:
  // =====================================================================================
  // Mock up a practice script to apply vranic method and prioritize smaller nodes first:
  // =====================================================================================
  // This type of script does NOT belong to either the quadtree NOR the vranic method. This is because this process below relies on a collection of quadtrees. Each of those quadtrees is a collection of nodes in velocity space which become the inputs to the vranic downsampling method:

  vranic_TYP vranic;
  std::vector<uint> ip_free;
  int N_min = 9;
  int N_max = 24;
  int N = 0;
  int M = 6;
  int particle_surplus = 0;
  arma::file_type format = arma::csv_ascii;
  vector<int> leaf_v_p_count;
  int particle_deficit = 0;
  int exit_flag_v = 0;

  // Down-sample deficit nodes:
  // -------------------------------------------------------------------------------------
  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Clear flag:
    exit_flag_v = 0;

    // Calculate particle surplus
    particle_surplus = leaf_x[xx]->p_count - round(mean_p_count);

    if (leaf_v[xx][0] != NULL)
    {
      // Assemble depth vector:
      int Nv = leaf_v[xx].size();
      vec depth(Nv);
      for(int vv = 0; vv < Nv; vv++){depth(vv) = leaf_v[xx][vv]->depth;}

      // Create sorted list starting from highest depth to lowest:
      uvec sorted_index_list = arma::sort_index(depth,"descend");

      // Loop over sorted leaf_v and apply vranic method:
      for (int vv = 0; vv < sorted_index_list.n_elem; vv++)
      {
        if (exit_flag_v == 1)
          break;

        // Sorted index:
        int tt = sorted_index_list(vv);

        // Diagnostics:
        {
          cout << "density = " << leaf_v[xx][tt]->p_count << endl;
          cout << "depth = " << leaf_v[xx][tt]->depth << endl;
          leaf_v[xx][tt]->center.print("center = ");
        }

        // Total number of particles in leaf_v cube:
        N = leaf_v[xx][tt]->p_count;

        // Test if current node is suitable for resampling:
        if (N < N_min){continue;}

        // Upper limit to the number of particles to down-sample in present cell:
        if (N > N_max){N = N_max;}

        // Exit condition:
        if (particle_surplus - (N-M) < 0)
        {
          // We have observed that this exit condition ensures that elements in x_p and v_p are not left to -1. I am not entirely sure why this happens
          break;
        }

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

        // Diagnostics: (proven to be correct)
        {
          ip.print("ip = ");
          set_N.xi.print("xi = ");
          set_N.yi.print("yi = ");
          set_N.zi.print("zi = ");
          set_N.wi.print("wi = ");
        }

        // Calculate set M based on set N:
        vranic.down_sample_3D(&set_N, &set_M);

        // Print statistics:
        cout << "Set N: " << endl;
        vranic.print_stats(&set_N);

        // Print statistics:
        cout << "Set M: " << endl;
        vranic.print_stats(&set_M); //(I AM HERE) continue with line 269 from main_7.cpp

        // Apply changes to distribution function:
        for (int ii = 0; ii < ip.n_elem; ii++)
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

  // Loop over leaf_x:
  arma::uvec reg_space = regspace<uvec>(0,1,Nx-1);
  arma::arma_rng::set_seed_random();
  arma::uvec shuffled_index = arma::shuffle(reg_space);

  for (int xx = 0; xx < Nx ; xx++)
  {
    // If ip_free is empty, then stop replication:
    if (ip_free_flag == 1)
      break;

    // Shuffled index:
    int xxs = shuffled_index(xx);

    // Calculate particle deficit
    particle_deficit = leaf_x[xxs]->p_count - mean_p_count;

    if (particle_deficit < 0)
    {
      // Number of particles to replicate (original particles in node):
      int num_0 = leaf_x[xxs]->p_count;

      // Replication number: represents how many times a particle needs to be replicated
      int rep_num = ceil(mean_p_count/num_0);

      // Initialize flag that stops replication:
      int rep_stop = 0;

      // Loop over replication number (represents the number of replication events)
      for (int rr = 0; rr < rep_num; rr++)
      {
        if (rep_stop == 1)
          break;

        // Loop over num_0 and replicate each one of them once
        for (int ii = 0; ii < num_0; ii++)
        {
          // Copy the data from the particle to replicate:
          uint jj = leaf_x[xxs]->ip[ii];
          double xi = x_p(jj);
          double yi = v_p(jj,0);
          double zi = v_p(jj,1);
          double wi = a_p(jj);

          // Replicate particle using the free memory location produced by the down-sampling:
          uint jj_free = ip_free.back();
          ip_free.pop_back();
          x_p(jj_free)   = xi; // - sign(xi)*0.01;
          v_p(jj_free,0) = yi;
          v_p(jj_free,1) = zi;

          // Adjust weight to conserve mass:
          a_p(jj)      = wi/2;
          a_p(jj_free) = wi/2;

          // Decrement deficit:
          particle_deficit++;
          if (particle_deficit >= 0)
          {
            rep_stop = 1;
            break;
          }

          // if size of ip_free vanishes, then stop all replication:
          int num_free = ip_free.size();
          if (num_free == 0)
          {
            rep_stop = 1;
            ip_free_flag = 1;
            break;
          }

        } // num_0 loop
      } // rep_num loop
    } // deficit if
  } // xx loop

  // Rescale:
  x_p*= x_norm;
  v_p*= y_norm;

  // Save resampled distribution for post-processing
  x_p.save(file_root + "x_p_new.csv",csv_ascii);
  v_p.save(file_root + "v_p_new.csv",csv_ascii);
  a_p.save(file_root + "a_p_new.csv",csv_ascii);

  // STEP X:
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
        cout << "p_count = " << leaf_v[xx][vv]->p_count << endl;
        particle_count(vv)  = leaf_v[xx][vv]->p_count;
        node_center.row(vv) = leaf_v[xx][vv]->center.t();
        node_dim.row(vv)    = leaf_v[xx][vv]->max.t() - leaf_v[xx][vv]->min.t();
      }

      // Save data:
      string file_name;
      file_name = file_root + "leaf_v_" + "p_count" + "_xx_" + to_string(xx) + ".csv";
      particle_count.save(file_name, arma::csv_ascii);

      file_name = file_root + "leaf_v_" + "node_center" + "_xx_" + to_string(xx) + ".csv";
      node_center.save(file_name, arma::csv_ascii);

      file_name = file_root + "leaf_v_" + "node_dim" + "_xx_" + to_string(xx) + ".csv";
      node_dim.save(file_name, arma::csv_ascii);
    }
  }

  // STEP X:
  // =====================================================================================
  // Test clearing data:
  // =====================================================================================
  quadTree[9].clear_tree();

  // STEP X:
  // =====================================================================================
  // Test deleting tree and releasing memory:
  // =====================================================================================
  quadTree[9].delete_tree();

  // Re-analize the modified data:
  // ======================================================================
  // Rescale:
  // x_p/= x_max;
  // v_p/= v_max;
  tree.clear_all();
  tree.insert_all(x_data);

  // Calculate leaf_x list:
  // ======================================================================
  p_count.zeros();
  leaf_x.clear();
  for (int xx = 0; xx < Nx ; xx++)
  {
   leaf_x[xx] = tree.find(xq(xx));
   p_count[xx] = leaf_x[xx]->p_count;
  }

  p_count.save(file_root + "leaf_x_p_count_new" + ".csv", arma::csv_ascii);

  return 0;
}
