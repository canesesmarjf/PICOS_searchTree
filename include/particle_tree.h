#ifndef PARTICLETREE_H
#define PARTICLETREE_H

#include <vector>
#include <armadillo>

#include "BinaryTree.h"
#include "QuadTree.h"
#include "Vranic.h"

using namespace std;
using namespace arma;

class particle_tree_TYP
{
// private:


public:
  // Variables:
  bt_TYP bt; // Binary tree object that hold x-space ROOT nodes:
  vector<qt_TYP> qt; // vector of quadtrees to hold v-space ROOT nodes
  bt_params_TYP * bt_params;  // Pointer to tree parameters
  qt_params_TYP * qt_params; // Pointer to tree parameters
  vec * x_p; // Pointer to particle x data
  mat * v_p; // Pointer to particle v data
  mat * a_p; // Pointer to particle weight data
  // int Nx; // Number of x-nodes on binary tree
  vec xq; // x query grid

  // Variables:
  vector<node_TYP *> leaf_x; // Vector of x-space LEAF nodes
  vector<vector<q_node_TYP *>> leaf_v; // Vector of v-space LEAF nodes
  ivec ip_count; // Hold the particle count for each leaf_x node
  int mean_ip_count;

  // Constructor:
  particle_tree_TYP(){};
  particle_tree_TYP(bt_params_TYP * bt_params, qt_params_TYP * qt_params, vec * x_p, mat * v_p, vec * a_p);

  // Methods:
  void create_x_query_grid();
  void calculate_leaf_x();
  void get_mean_ip_count();
  void populate_tree();
  void assemble_qt_vector();
  void calculate_leaf_v();
  void resample_distribution();
  void clear_all_contents();

private:
  void downsample_surplus_nodes(vector<uint> * ip_free);
  void upsample_deficit_nodes(vector<uint> * ip_free);

};

#endif
