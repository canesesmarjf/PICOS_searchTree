#ifndef BINARYTREE_H
#define BINARYTREE_H

#include <vector>
#include <armadillo>
#include <cassert>
using namespace std;

// This header file contains three main classes:
// - tree_params_TYP
// - binaryTree_TYP
// - node_TYP

// Is important to node that a tree is composed of nodes; thus a tree as a whole will have parameters such as:
// - Maximum depth
// - Left and right boundaries
// - Number of nodes in the final layer
// Moreover, each node will also have its own parameters:
// - current depth
// - Left and right boundaries
// - Data stored in node

// =====================================================================================
class tree_params_TYP
{
public:
  uint dimensionality;
  arma::vec min;
  arma::vec max;
  arma::uvec max_depth;
  arma::uvec dim_levels; // Determines when the tree needs to change to the next dimension
  uint num_nodes; // Number of nodes based on 2^depth_max

  // Derived variables:
  // double length; // Length of entire domain
  // double dx;
  // int num_elem; // Total number of elements to hold in tree

  // int elem_per_node; // number of elements per node based on a uniform distribution
  // arma::vec node_centers; // Vector containing the center locations of the nodes
};

// =====================================================================================
class node_TYP
{
public:
  // Data variables:
  int p_count;            // Number of points indexed in the current node
  std::vector<uint> ip;   // Indices of data appended to this node

  // Node parameters:
  uint depth;
  arma::vec min;
  arma::vec max;
  arma::vec center;

  // Constructor:
  node_TYP(){};
  node_TYP(arma::vec min, arma::vec max, uint depth, tree_params_TYP * tree_params);

  // Methods:
  void insert(uint i, vector<arma::vec *> data, bool write_data); // Insert the ith element of data
  void insert_all(vector<arma::vec *> data); // Insert all elements of the data
  node_TYP * find(uint i, vector<arma::vec *> data, int search_dimensionality);
  void clear_node();
  node_TYP * find(double xq); // Find and return pointer of node corresponding to position xq
  node_TYP * find(double xq, int dim); // Find and return pointer of node corresponding to position xq searched along dimension "dim"
  int count_leaf_nodes(int k);
  int count_leaf_points(int k);
  void delete_nodes();

private:
  // Variables:
  tree_params_TYP * tree_params; // Pointer to tree parameters

  // Subnodes within this node:
  // subnode[0] : right_node
  // subnode[1] : left_node
  //   +------------------+------------------+
  //   |  left_node = 1   |  right_node = 0  |
  std::vector<node_TYP *> subnode;

  // Methods:
  bool IsPointInsideBoundary(double p, int dim);
  bool HasNodeReachMaxDepth(int dim);
  int WhichSubNodeDoesItBelongTo(double p, int dim);
  bool DoesSubNodeExist(int subNode);
  void CreateSubNode(int subNode, int dim);
};

// =====================================================================================
class binaryTree_TYP
{
public:
  // Constructor:
  binaryTree_TYP();
  binaryTree_TYP(tree_params_TYP * tree_params);

  // Variables:
  node_TYP * root; // Root node of tree
  tree_params_TYP * tree_params;  // Pointer to tree attributes
  std::vector<node_TYP *> x_nodes; // List of pointers to leaf nodes on 1st dimension

  // Methods:
  void insert_all(vector<arma::vec *> data);
  node_TYP * find(int i,vector<arma::vec *> data,int search_dimensionality);
  node_TYP * find(double xq);
  void clear_all();
  int count_leaf_nodes();
  int count_leaf_points();
  void delete_nodes();

  // int get_num_nodes();
  // arma::vec get_node_centers();
  // int get_max_depth();
  // void print_info(int ii);
  // void save_data_all(string prefix);

private:
  // Variables:
  // node_TYP * root; // Root node of tree

  // Methods:
  // void assemble_node_list();
  // void save_data(int ii, string prefix);
};

#endif
