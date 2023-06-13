#ifndef PARTICLETREE_H
#define PARTICLETREE_H

#include <vector>
#include <armadillo>

#include "BinaryTree.h"
#include "QuadTree.h"
#include "Vranic.h"

using namespace std;

class particle_tree_TYP
{
// private:


public:
  // Variables:
  binaryTree_TYP binary_tree; // Binary tree object that hold x-space ROOT nodes:
  std::vector<quadTree_TYP> quad_tree; // vector of quadtrees to hold v-space ROOT nodes
  tree_params_TYP * binary_tree_params;  // Pointer to tree parameters
  quadTree_params_TYP * quad_tree_params; // Pointer to tree parameters
  arma::vec * x_p; // Pointer to particle x data
  arma::mat * v_p; // Pointer to particle v data
  arma::mat * a_p; // Pointer to particle weight data
  int Nx; // Number of x-nodes on binary tree
  arma::vec xq; // x query grid

  // Variables:
  vector<node_TYP *> leaf_x; // Vector of x-space LEAF nodes
  vector<vector<quadNode_TYP *>> leaf_v; // Vector of v-space LEAF nodes
  arma::ivec p_count; // Hold the particle count for each leaf_x node
  int mean_p_count;

  // Constructor:
  particle_tree_TYP(){};
  particle_tree_TYP(tree_params_TYP * binary_tree_params, quadTree_params_TYP * quad_tree_params, arma::vec * x_p, arma::mat * v_p, arma::vec * a_p);

  // Methods:
  void create_x_query_grid();
  void calculate_leaf_x();
  void get_mean_p_count();
  void populate_tree();
  void assemble_quad_tree_vector();
  void calculate_leaf_v();
  void resample_distribution();

};

#endif
