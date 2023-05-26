#ifndef QUADTREE_H
#define QUADTREE_H

#include <vector>
#include <armadillo>
using namespace std;

// =====================================================================================
class quadTree_params_TYP
{
public:
  arma::vec min; // Minimum lengths that bound domain covered by tree
  arma::vec max; // Maximum lengths that bound domain covered by tree
  uint max_depth; // Maximum depth to be reached of min_count is not reached first
  uint min_count; // Minimum number of particles in cell allowed.

  // Note:
  // We need to add am additional variable that enables us to determine the exit condition to form a leaf node based on either max_depth or min_count
};

// =====================================================================================
class quadNode_TYP
{
public:
  // Data variables:
  int p_count;            // Number of points indexed in the current node
  std::vector<uint> ip;   // Indices of data appended to this node
  arma::mat *v;           // Pointer to data to be indexed

  // Node parameters:
  uint depth;
  arma::vec min;
  arma::vec max;
  arma::vec center;

  // Constructor:
  quadNode_TYP(){};
  quadNode_TYP(arma::vec min, arma::vec max, uint depth, quadTree_params_TYP * quadTree_params, std::vector<uint> ip,arma::mat * v);

  // Methods:
  void populate_node();
  void organize_points(uint jj);
  void clear_node();
  void delete_nodes();
  int count_leaf_points(int k);
  void get_all_leafs(vector<quadNode_TYP *> *);
  void get_subnode_bounds(int node_index, arma::vec * min_l, arma::vec * max_l);
  void get_leaf_nodes(vector<quadNode_TYP *> * leafs);

private:
  // Variables:
  quadTree_params_TYP * quadTree_params; // Pointer to tree parameters

  // Subnodes within this node:
  //   +------------------+------------------+
  //   |  left_node = 1   |  right_node = 0  |
  //   +------------------+------------------+
  //   |  left_node = 2   |  right_node = 3  |
  //   +------------------+------------------+

  std::vector<quadNode_TYP *> subnode;
  std::vector<vector<uint>> subnode_ip;

  // Methods:
  bool IsPointInsideBoundary(arma::vec r);
  int WhichSubNodeDoesItBelongTo(arma::vec r);
  bool DoesSubNodeExist(int subNode);
  void CreateSubNode(int subNode);
};

// =====================================================================================
class quadTree_TYP
{
public:
  // Constructor:
  quadTree_TYP();
  quadTree_TYP(quadTree_params_TYP * quadTree_params, vector<uint> ip, arma::mat * v);

  // Variables:
  quadNode_TYP * root; // Root node of tree
  quadTree_params_TYP * quadTree_params;  // Pointer to tree attributes
  std::vector<quadNode_TYP *> leaf_v; // List of pointers to leaf nodes

  // Methods:
  void populate_tree();
  void clear_tree();
  void delete_tree();
  vector<quadNode_TYP *> get_leaf_nodes();

private:
  // Variables:
  // node_TYP * root; // Root node of tree

  // Methods:
  // void assemble_node_list();
  // void save_data(int ii, string prefix);
};


#endif
