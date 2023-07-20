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
  uint min_depth;
  uint max_depth; // Maximum depth to be reached of min_count is not reached first
  uint min_count; // Minimum number of particles in cell allowed.

  // Note:
  // We need to add am additional variable that enables us to determine the exit condition to form a leaf node based on either max_depth or min_count
};

// =====================================================================================
class quadNode_TYP
{
public:
  // Node "data" attributes:
  int p_count;            // Number of points indexed in the current node
  std::vector<uint> ip;   // Indices of data appended to this node
  arma::mat *v;           // Pointer to data to be indexed

  // Node "natural" attributes:
  uint depth;
  arma::vec min;
  arma::vec max;
  arma::vec center;

  // Constructor:
  quadNode_TYP(){};
  quadNode_TYP(arma::vec min, arma::vec max, uint depth, quadTree_params_TYP * quadTree_params, std::vector<uint> ip,arma::mat * v);

  // Methods:
  void populate_node();
  void calculate_ip_subnode();
  void clear_node();
  void delete_nodes();
  int count_leaf_points(int k);
  void get_all_leafs(vector<quadNode_TYP *> *);
  void get_subnode_bounds(int node_index, arma::vec * min_l, arma::vec * max_l);
  void get_leaf_nodes(vector<quadNode_TYP *> * leafs);

private:
  // Variables:
  quadTree_params_TYP * quadTree_params; // Pointer to tree parameters
  std::vector<quadNode_TYP *> subnode;
  std::vector<vector<uint>> ip_subnode;
  bool is_leaf;

  // subnodes within this node:
  //   +------------------+------------------+
  //   |    subnode[1]    |    subnode[0]    |
  //   +------------------+------------------+
  //   |    subnode[2]    |    subnode[3]    |
  //   +------------------+------------------+

  // Methods:
  bool IsPointInsideBoundary(arma::vec r);
  int WhichSubNodeDoesItBelongTo(arma::vec r);
  bool DoesSubNodeExist(int subNode);
  void CreateSubNode(int subNode);
  bool is_node_leaf(int method);
  int apply_conditionals_ip_subnode();

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

  // This might need to be removed as it is unused:
  std::vector<quadNode_TYP *> leaf_v; // List of pointers to leaf nodes

  // Methods:
  void populate_tree();
  void clear_tree();
  void delete_tree();
  vector<quadNode_TYP *> get_leaf_nodes();
  int count_leaf_points();

  // QUESTION:
  // Everytime we add data to the quadtree, do we need to delete all leaf nodes? This means releasing memory and deleting dangling pointers. We need to consider this carefully as we will be using the quadtrees potetially millions of times.

private:
  // Variables:
  // node_TYP * root; // Root node of tree

  // Methods:
  // void assemble_node_list();
  // void save_data(int ii, string prefix);
};


#endif
