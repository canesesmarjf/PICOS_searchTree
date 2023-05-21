#include <iostream>
#include <armadillo>
#include "BinaryTree.h"
#include "Vranic.h"
#include <chrono>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <filesystem>

using namespace std::chrono;
using namespace std;
using namespace arma;
namespace fs = filesystem;

/* OBJECTIVE:

*/

int main()
{
  // Load input data:
  // ===================================================================
  arma::vec x_p;
  arma::mat v_p;
  arma::vec a_p;

  string input_step    = "Step_1";
  string output_step   = "Step_2";
  string input_folder  = "./" + input_step + "_Output/";
  string output_folder = "./" + output_step + "_Output/";
  string input_prefix  = input_step + "_";
  string output_prefix = output_step + "_";

  string picos_case  = "PICOS_case_2_";
  string species     = "ss_2_";
  string time        = "tt_50_";
  string scenario    = picos_case + species + time;

  string file_name = input_folder + input_prefix + scenario;
  x_p.load(file_name + "x_p.csv",csv_ascii);
  v_p.load(file_name + "v_p.csv",csv_ascii);
  a_p.load(file_name + "a_p.csv",csv_ascii);

  // Normalize data:
  double v_max = max(max(v_p));
  v_max *= 1.2;
  double x_max = 2.0;

  double x_norm = x_max;
  double y_norm = v_max;
  double z_norm = v_max;

  x_p = x_p/x_max;
  v_p = v_p/v_max;

  // Start with a 3D binary tree:
  // ======================================================================
  tree_params_TYP tree_params;

  tree_params.dimensionality = 3;
  tree_params.min       = {-1 , -1, 0 };
  tree_params.max       = {+1 , +1, +1};
  tree_params.max_depth = {5+1, 6-0 , 5-0 };

  // Assemble data into format needed for tree:
  // ===================================================================
  // Copy the data to arma::vec types:
  auto start = high_resolution_clock::now();
  arma::vec v_par = v_p.col(0);
  arma::vec v_per = v_p.col(1);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "time required for copy = " << duration.count() << endl;

  // Create a vector with pointers to the data:
  vector<arma::vec *> data = {&x_p, &v_par, &v_per, &a_p};

  // Create tree based on parameters:
  // ======================================================================
  binaryTree_TYP tree(&tree_params);

  // Insert 2D points into tree:
  // ======================================================================
  tree.insert_all(data);

  // Count how many nodes where populated:
  // ======================================================================
  int k = tree.count_leaf_nodes();
  cout << "total number of leaf nodes populated: " << k << endl;

  // Count how many points where inserted:
  // ======================================================================
  k = tree.count_leaf_points();
  cout << "total number of leaf points inserted: " << k << endl;

  // Folder where data is to be stored:
  // ======================================================================
  string file_root = "./Step_2_Output/";
  if (fs::is_directory(file_root) == false)
  {
    fs::create_directory(file_root);
  }

  // Create grid for each dimension in binary tree:
  // ======================================================================
  // This section probably needs to be part of the tree_params_TYP class:
  // tree_params.calculate_grids();

  double Lx = tree_params.max[0] - tree_params.min[0];
  double Ly = tree_params.max[1] - tree_params.min[1];
  double Lz = tree_params.max[2] - tree_params.min[2];

  int Nx = pow(2,tree_params.max_depth[0]);
  int Ny = pow(2,tree_params.max_depth[1]);
  int Nz = pow(2,tree_params.max_depth[2]);

  double dx = Lx/Nx;
  double dy = Ly/Ny;
  double dz = Lz/Nz;

  arma::vec xq = tree_params.min[0] + dx/2 + regspace(0,Nx-1)*dx;
  arma::vec yq = tree_params.min[1] + dy/2 + regspace(0,Ny-1)*dy;
  arma::vec zq = tree_params.min[2] + dz/2 + regspace(0,Nz-1)*dz;

  // Create pointers for leaf nodes:
  // ======================================================================
  vector<node_TYP *> leaf_x(Nx);
  node_TYP * leaf_y = NULL;
  node_TYP * leaf_z = NULL;

  // Calculate leaf_x list:
  // ======================================================================
  arma::ivec p_count(Nx);
  cout << "Nx = " << Nx << endl;
  for (int xx = 0; xx < Nx ; xx++)
  {
   cout << "xq(xx) = " << xq(xx) << endl;
   leaf_x[xx] = tree.find(xq(xx));
   p_count[xx] = leaf_x[xx]->p_count;
  }

  int mean_pcount = mean(p_count);

  // Save data:
  file_name = output_folder + output_prefix + scenario;
  p_count.save(file_name + "p_count" + ".csv", arma::csv_ascii);

  // Apply improved method that enables to loop over cells with largest density first:
  // ======================================================================
  // Input:
  // N_min, N, M
  // ip_free
  // data
  // leaf_x from tree: leaf_x[xx]->p_count
  // xq, yq, zq

  // Output:
  // data
  // ip_free

  vranic_TYP vranic;
  std::vector<uint> ip_free;
  int N_min = 10;
  int N = 0;
  int M = 6;
  int particle_surplus = 0;
  arma::file_type format = arma::csv_ascii;
  vector<node_TYP *> leaf_v;
  vector<int> leaf_v_p_count;
  ip_free.clear();
  int particle_deficit = 0;
  int exit_flag_v = 0;

  // Loop over leaf_x:
  for (int xx = 0; xx < Nx ; xx++)
  {
    // Clear flag:
    exit_flag_v = 0;

    // Calculate particle surplus
    particle_surplus = leaf_x[xx]->p_count - mean_pcount;

    // Check if current leaf_x is in surplus:
    if (particle_surplus >= 0)
    {
      // Loop over y:
      for (int yy = 0; yy < Ny; yy++)
      {
        // Get current leaf_y:
        leaf_y = leaf_x[xx]->find(yq[yy],1);

        if (leaf_y != NULL)
        {
          // Loop over z:
          for (int zz = 0; zz < Nz; zz++)
          {
           leaf_z = leaf_y->find(zq[zz],2);
           if (leaf_z != NULL)
           {
             // Total number of particles in leaf_z cube:
             leaf_v_p_count.push_back(leaf_z->p_count);

             // Record pointer to leaf_z:
             leaf_v.push_back(leaf_z);

           } // leaf_z NULL check
          } // leaf_z loop
        } // leaf_y NULL check
      } // leaf_y loop

      // Sort leaf_v_p_count:
      uvec uv = conv_to<uvec>::from(leaf_v_p_count);
      uvec sorted_uv = sort_index(uv,"descend");

      // Loop over sorted leaf_v and apply vranic method:
      for (int vv = 0; vv < sorted_uv.n_elem; vv++)
      {

        if (exit_flag_v == 1)
          break;

        int tt = sorted_uv(vv);
        cout << "density = " << leaf_v[tt]->p_count << endl;

        // Total number of particles in leaf_z cube:
        N = leaf_v[tt]->p_count;

        if (N >= N_min)
        {
          // Adjust N so that surplus count does not become negative:
          // Negative means that leaf_x will be in deficit.
          if (particle_surplus - N < 0)
          {
           N = particle_surplus;

           // Check that new set N is not smaller than M:
           if (N < (M+1))
           {
             exit_flag_v = 1;
             break;
           }
          }
        }

        // Create objects for down-sampling:
        merge_cell_TYP set_N(N);
        merge_cell_TYP set_M(M);

        // Create set_N:
        arma::uvec ip = conv_to<uvec>::from(leaf_v[tt]->ip);
        set_N.xi = data[0]->elem(ip.head(N));
        set_N.yi = data[1]->elem(ip.head(N));
        set_N.zi = data[2]->elem(ip.head(N));
        set_N.wi = data[3]->elem(ip.head(N));

        // Calculate set M:
        vranic.down_sample(&set_N, &set_M);

        // Print statistics:
        cout << "Set N: " << endl;
        vranic.print_stats(&set_N);

        // Print statistics:
        cout << "Set M: " << endl;
        vranic.print_stats(&set_M);

        for (int ii = 0; ii < N; ii++)
        {
          // Get global index:
          int jj = ip(ii);

          if (ii < set_M.n_elem)
          {
            // Down-sample global distribution:
            // Use set_N for xi in order to remove oscillations in density:
            // Use set_M for all other quantities (v and a):

            // Set N:
            data[0]->at(jj) = set_N.xi(ii);

            // Set M:
            data[1]->at(jj) = set_M.yi(ii);
            data[2]->at(jj) = set_M.zi(ii);
            data[3]->at(jj) = set_M.wi(ii);
          }
          else
          {
            // Record global indices that correspond to memory locations that are to be repurposed:
            ip_free.push_back(jj);

            // Clear values:
            data[0]->at(jj) = -1;
            data[1]->at(jj) = -1;
            data[2]->at(jj) = -1;
            data[3]->at(jj) = -1;

          }

          // Decrement particle surplus:
          particle_surplus--;
        }

      } // leaf_v loop

      // Clear contents of leaf_v;
      leaf_v.clear();
      leaf_v_p_count.clear();

    } // surplus if
  } // leaf_x loop

  // Populate deficit regions:
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
    particle_deficit = leaf_x[xxs]->p_count - mean_pcount;

    if (particle_deficit < 0)
    {
      // Number of particles to replicate:
      int num_rep = leaf_x[xxs]->p_count;

      // Replication number:
      int rep = ceil(mean_pcount/num_rep);

      // Initialize flag that stops replication:
      int rep_flag = 0;

      // Loop over num_rep;
      for (int rr = 0; rr < rep; rr++)
      {
        if (rep_flag == 1)
          break;

        for (int ii = 0; ii < num_rep; ii++)
        {
          // Get data from the particle to replicate:
          uint jj = leaf_x[xxs]->ip[ii];
          double xi = data[0]->at(jj);
          double yi = data[1]->at(jj);
          double zi = data[2]->at(jj);
          double wi = data[3]->at(jj);

          // Replicate particle:
          uint jj_free = ip_free.back();
          ip_free.pop_back();
          data[0]->at(jj_free) = xi; // - sign(xi)*0.01;
          data[1]->at(jj_free) = yi;
          data[2]->at(jj_free) = zi;

          // Adjust weight:
          data[3]->at(jj) = wi/2;
          data[3]->at(jj_free) = wi/2;

          // Decrement deficit:
          particle_deficit++;
          if (particle_deficit > 0)
          {
            rep_flag = 1;
            break;
          }

          // if size of ip_free vanishes, then stop all replication:
          int num_free = ip_free.size();
          if (num_free == 0)
          {
            rep_flag = 1;
            ip_free_flag = 1;
            break;
          }

        } // num_rep loop
      } // rep loop
    } // deficit if

  } // xx loop

  // Update v_p:
  v_p.col(0) = *(data[1]);
  v_p.col(1) = *(data[2]);

  // Save data:
  // data_folder = "./Step_2_Output/";
  // picos_case  = "Step_2_PICOS_case_1_";
  file_name = output_folder + output_prefix + scenario;
  // file_name = data_folder + picos_case + species + time;

  // Rescale:
  x_p*= x_max;
  v_p*= v_max;

  x_p.save(file_name + "x_p_new.csv",csv_ascii);
  v_p.save(file_name + "v_p_new.csv",csv_ascii);
  a_p.save(file_name + "a_p_new.csv",csv_ascii);

  // Re-analize the modified data:
  // ======================================================================
  // Rescale:
  x_p/= x_max;
  v_p/= v_max;
  tree.clear_all();
  tree.insert_all(data);

  // Calculate leaf_x list:
  // ======================================================================
  p_count.zeros();
  leaf_x.clear();
  for (int xx = 0; xx < Nx ; xx++)
  {
   leaf_x[xx] = tree.find(xq(xx));
   p_count[xx] = leaf_x[xx]->p_count;
  }

  // Save data:
  // data_folder = "./Step_2_Output/";
  // picos_case  = "Step_2_PICOS_case_1_";
  // file_name = data_folder + picos_case + species + time;
  file_name = output_folder + output_prefix + scenario;
  p_count.save(file_name + "p_count_new" + ".csv", arma::csv_ascii);

  // Count how many nodes where populated:
  // ======================================================================
  k = tree.count_leaf_nodes();
  cout << "total number of leaf nodes populated: " << k << endl;

  // Count how many points where inserted:
  // ======================================================================
  k = tree.count_leaf_points();
  cout << "total number of leaf points inserted: " << k << endl;

  return 0;
}
