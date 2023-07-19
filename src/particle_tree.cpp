#include "particle_tree.h"

using namespace arma;

// ======================================================================================
particle_tree_TYP::particle_tree_TYP(tree_params_TYP * binary_tree_params, quadTree_params_TYP * quad_tree_params, arma::vec * x_p, arma::mat * v_p, arma::vec * a_p)
{
  // Store pointer to tree parameters:
  this->binary_tree_params = binary_tree_params;
  this->quad_tree_params = quad_tree_params;

  // Store pointers to data:
  this->x_p = x_p;
  this->v_p = v_p;
  this->a_p = a_p;

  // Create instance of binary tree for x dimension:
  binary_tree = binaryTree_TYP(binary_tree_params);

  // Create x dimension query grid for binary_tree:
  this->create_x_query_grid();

  // Set capacity of quad trees for v dimension:
  // quad_tree.reserve(this->Nx);

}

// vector<particle_tree_TYP> IONS_tree;
// IONS_tree.leaf_x[xx]->ip;

// ======================================================================================
void particle_tree_TYP::create_x_query_grid()
{
  // Number of x grid nodes:
  this->Nx = pow(2,binary_tree_params->max_depth[0]);

  // Calculate total length and node length in x-space:
  double Lx = binary_tree_params->max[0] - binary_tree_params->min[0];
  double dx = Lx/Nx;

  // create query grid:
  this->xq = binary_tree_params->min[0] + dx/2 + regspace(0,Nx-1)*dx;
}

// ======================================================================================
void particle_tree_TYP::calculate_leaf_x()
{
  // Allocate memory and resize leaf_x:
  leaf_x.resize(Nx);

  // Allocate memory and resize p_count:
  p_count = arma::zeros<arma::ivec>(Nx);

  // Assemble leaf_x and p_count:
  for (int xx = 0; xx < Nx ; xx++)
  {
   leaf_x[xx] = binary_tree.find(xq(xx));
   p_count[xx] = leaf_x[xx]->p_count;
  }
}

// ======================================================================================
void particle_tree_TYP::get_mean_p_count()
{
  // This operation requires the division between two integers; hence, care needs to be taken since this leads to loss of precision. Method used the the following:
  // 1- Cast numerator as a double so that division retains fractional part
  // 2- Round UP to the nearest LARGEST integer so as to over-estimate total number of particles. We later test for total particle usage to avoid "out of bounds" access.

  int p_count_sum = sum(p_count);
  mean_p_count = ceil((double)p_count_sum/Nx);
}

// ======================================================================================
void particle_tree_TYP::assemble_quad_tree_vector()
{
  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Create quadtree[xx] only if leaf_x[xx] is surplus:
    int delta_p_count = p_count[xx] - mean_p_count;
    if (delta_p_count > 0)
    {
      // Data for quadtree:
      vector<uint> ip = leaf_x[xx]->ip;

      // Create quadtree:
      quad_tree.emplace_back(quad_tree_params,ip,v_p);

      // Populate quadtree:
      quad_tree[xx].populate_tree();
    }
    else
    {
      quad_tree.emplace_back();
    }
  }
}

// ======================================================================================
void particle_tree_TYP::calculate_leaf_v()
{
  // Allocate memory and resize leaf_v:
  leaf_v.resize(Nx);

  for (int xx = 0; xx < leaf_x.size() ; xx++)
  {
    // Create quadtree[xx] only if leaf_x[xx] is surplus:
    int delta_p_count = p_count[xx] - mean_p_count;
    if (delta_p_count > 0)
    {
      cout << "hello 2" << endl;

      // Extract all leaf nodes:
      leaf_v[xx] = quad_tree[xx].get_leaf_nodes();

      // Diagnostics:
      if (false)
      {
      int sum = 0;
      for(int ll = 0; ll < leaf_v[xx].size(); ll++)
        sum = sum + leaf_v[xx][ll]->p_count;
      int ip_count = quad_tree[xx].root->count_leaf_points(0);

      cout << "p_count[xx] = " << p_count[xx] << endl;
      cout << "ip_count = " << ip_count << endl;
      cout << "sum = " << sum << endl;
      }

    }
    else
    {
      leaf_v[xx] = {NULL};
      cout << "hello:" << endl;
    }
  }
}

// ======================================================================================
void particle_tree_TYP::populate_tree()
{
  // Create a vector with pointers to the data:
  vector<arma::vec *> x_data = {x_p};

  // Insert 1D points into tree:
  binary_tree.insert_all(x_data);

  // Calculate leaf_x list:
  this->calculate_leaf_x();

  // Calculating the mean number of particles per node:
  this->get_mean_p_count();

  // Assemble quadtrees for each leaf_x element:
  this->assemble_quad_tree_vector();

  // Assembling leaf_v 2D vector:
  this->calculate_leaf_v();

}

// ======================================================================================
void particle_tree_TYP::resample_distribution()
{

  // Notes:
  // Consider updating leaf_x[xx]->ip and p_count everytime you remove or add elements. this allows to reuse the binary tree data leaf_x for other operations.

  // Consider having the following two methods in this block:
  // - downsample_process(&ip_free);
  // - replication_process(&ip_free);


  // Down-sample surplus nodes:
  // -------------------------------------------------------------------------------------
  std::vector<uint> ip_free;
  // this->down_sample(&ip_free);

  // Down sampling process:
  {
    // Create vranic down-sampling object:
    vranic_TYP vranic;

    // Set the min and maximum number of computational particles per node:
    int N_min = 7;
    int N_max = 300;

    // Set the particle set sizes for the input and output sets:
    int N; // Input set
    int M = 6; // Output set
    int particle_surplus;

    // Down-sample distribution and populate ip_free vector:
    for (int xx = 0; xx < leaf_x.size() ; xx++)
    {
      // Calculate particle surplus
      particle_surplus = leaf_x[xx]->p_count - mean_p_count;

      // Apply Vranic method if x-node is NOT NULL:
      if (leaf_v[xx][0] != NULL)
      {
        // Total number of v-nodes for this xx position:
        int Nv = leaf_v[xx].size();

        // Assemble depth and p_count vectors:
        vec depth(Nv);
        vec p_count(Nv);
        vec node_metric(Nv);
        for(int vv = 0; vv < Nv; vv++){depth(vv) = leaf_v[xx][vv]->depth;}
        for(int vv = 0; vv < Nv; vv++){p_count(vv) = leaf_v[xx][vv]->p_count;}

        // Calculate metric for every node based on depth*p_count (elementwise multiplication). The higher the depth AND number of particles, the higher the metric:
        node_metric = depth%p_count;

        // Create sorted list starting from highest depth to lowest:
        // By giving priority to nodes with highest metric, we are operating on the regions that minimize the changes to the distribution function since they are more localized in velocity space:
        uvec sorted_index_list = arma::sort_index(node_metric,"descend");

        // Loop over sorted leaf_v and apply vranic method:
        for (int vv = 0; vv < sorted_index_list.n_elem; vv++)
        {
          // Sorted index:
          int tt = sorted_index_list(vv);

          // Total number of particles in leaf_v cube:
          N = leaf_v[xx][tt]->p_count;

          // Test if current node is suitable for resampling:
          if (N < N_min){continue;}

          // Upper limit to the number of particles to down-sample in present cell:
          if (N > N_max){N = N_max;}

          // Check that we do not go into deficit:
          if (particle_surplus - (N-M) < 0){N = particle_surplus + M;}

          // Create objects for down-sampling:
          merge_cell_TYP set_N(N);
          merge_cell_TYP set_M(M);

          // Define particle indices for set N:
          arma::uvec ip = conv_to<uvec>::from(leaf_v[xx][tt]->ip);
          arma::uvec ip_subset = ip.head(N);

          // Assign particle attributes for set N:
          arma::mat v_p_subset = v_p->rows(ip_subset);
          set_N.xi = x_p->elem(ip_subset);
          set_N.yi = v_p_subset.col(0);
          set_N.zi = v_p_subset.col(1);
          set_N.wi = a_p->elem(ip_subset);

          // Calculate set M based on set N:
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
              (*x_p)(jj) = set_N.xi(ii);

              // Set M:
              (*v_p)(jj,0) = set_M.yi(ii);
              (*v_p)(jj,1) = set_M.zi(ii);
              (*a_p)(jj)   = set_M.wi(ii);

            }
            else
            {
              // Record global indices that correspond to memory locations that are to be repurposed:
              ip_free.push_back(jj);

              // Set values to -1 to flag them as undefined values memory locations:
              (*x_p)(jj)   = -1;
              (*v_p)(jj,0) = -1;
              (*v_p)(jj,1) = -1;
              (*a_p)(jj)   = -1;

              // Decrement particle surplus only after ip_free.push_back operation:
              particle_surplus--;
            }
          } // for ip loop
        } // sorted index Loop
      } // if not NULL
    } // xx loop

  } // Down sampling process

  // Populate deficit regions:
  // -------------------------------------------------------------------------------------
  // INPUTS: ip_free;
  // class members used:
  // leaf_x
  // Nx
  // x_p, v_p, a_p

  // Replication process:
  {
    // Initialize replication flag:
    int ip_free_flag = 0;
    int particle_deficit;

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
            double xi = (*x_p)(jj);
            double yi = (*v_p)(jj,0);
            double zi = (*v_p)(jj,1);
            double wi = (*a_p)(jj);

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
              (*x_p)(jj_free)   = xi; // - sign(xi)*0.01;
              (*v_p)(jj_free,0) = yi;
              (*v_p)(jj_free,1) = zi;
              (*a_p)(jj_free)   = wi/((double)num_new + 1.0);

              // Modify deficit:
              particle_deficit++;
              node_counts(xx)++;
            }

            // Adjust weight of parent particle to conserve mass:
            (*a_p)(jj) = wi/((double)num_new + 1.0);

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

  } // Replication process
}
