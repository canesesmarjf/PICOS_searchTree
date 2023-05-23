#include "QuadTree.h"
#include <iostream>

using namespace std;
using namespace arma;

// =======================================================================================
quadTree_TYP::quadTree_TYP()
{
  this->quadTree_params = NULL;
  root = NULL;
  leaf_v = {NULL};
}

// =======================================================================================
quadTree_TYP::quadTree_TYP(quadTree_params_TYP * quadTree_params, vector<uint> ip, arma::mat * v)
{
  // Store pointer to tree parameters
  this->quadTree_params = quadTree_params;

  // Create root node:
  uint depth_root = 0;
  arma::vec min = quadTree_params->min;
  arma::vec max = quadTree_params->max;
  root = new quadNode_TYP(min,max,depth_root,quadTree_params,ip,v);
}

// =======================================================================================
void quadTree_TYP::populate_tree()
{
  this->root->populate_node();
}

// =======================================================================================
void quadTree_TYP::clear_tree()
{
  if (NULL == root)
  {
    cout << "Tree has NULL root node" << endl;
    return;
  }

  root->clear_node();
}

// =======================================================================================
void quadTree_TYP::delete_tree()
{
  this->root->delete_nodes();
}

// =======================================================================================
vector<quadNode_TYP *> quadTree_TYP::get_leaf_nodes()
{
  vector<quadNode_TYP *> leafs;
  root->get_leaf_nodes(&leafs);
  return leafs;
}

// =======================================================================================
void quadNode_TYP::get_leaf_nodes(vector<quadNode_TYP *> * leafs)
{
  for (int nn = 0; nn < 4; nn++)
  {
    if (this->subnode[nn] != NULL)
      this->subnode[nn]->get_leaf_nodes(leafs);
  }

  // Determine of node is a leaf;
  bool is_leaf = true;
  for(int nn = 0; nn < 4; nn++)
  {
    is_leaf = this->subnode[nn] == NULL && is_leaf;
  }

  // If leaf node, accumulate p_count:
  if (is_leaf == true)
  {
    leafs->push_back(this);
  }
  else
  {
   // Do nothing
  }
}

// =======================================================================================
void quadNode_TYP::delete_nodes()
{
  for (int nn = 0; nn < 4; nn++)
  {
    if (this->subnode[nn] != NULL)
    {
      this->subnode[nn]->delete_nodes();

      // Release memory pointed by subnode[nn]:
      delete this->subnode[nn];

      // Prevent dangling pointer:
      this->subnode[nn] = NULL;
    }
    else
    {
      // Do nothing
    }
  }
}

// =======================================================================================
quadNode_TYP::quadNode_TYP(arma::vec min, arma::vec max, uint depth, quadTree_params_TYP * quadTree_params,std::vector<uint> ip, arma::mat * v)
{
  // Node attributes:
  this->center  = (min + max)/2;
  this->min     = min;
  this->max     = max;
  this->depth   = depth;
  this->quadTree_params = quadTree_params;
  this->ip = ip;
  this->v  = v;

  // Allocate memory for subnodes:
  this->subnode.reserve(4);
  this->subnode[0] = NULL;
  this->subnode[1] = NULL;
  this->subnode[2] = NULL;
  this->subnode[3] = NULL;
  this->subnode_ip.resize(4);
  this->p_count = ip.size();
}

// =======================================================================================
void quadNode_TYP::populate_node()
{
  // Insert data into subnode_ip vectors:
  for (int ii = 0; ii < p_count; ii++)
  {
    // Insert the current point a single level down to the correspoding subnode:
    uint jj = ip[ii];
    insert(jj);
  }

  // Clear ip since they are now distributed in vector subnode_ip:
  // Retain p_count, as this tells you how many particles went through this node
  ip.clear();

  // Calculate new depth:
  uint depth = this->depth + 1;

  // Local variables to store bounds of subnodes:
  vec min_local(2);
  vec max_local(2);

  for (int ni = 0; ni < 4; ni++)
  {
    if (subnode_ip[ni].size() > 0)
    {
      // Bounds of new subnode:
      get_subnode_bounds(ni,&min_local,&max_local);

      // Create subnode:
      vector<uint> ip = subnode_ip[ni];
      subnode[ni] = new quadNode_TYP(min_local,max_local,depth,quadTree_params,ip,v);

      // Diagnostic:
      cout << subnode[ni]->ip.size() << endl;

      // Populate current subnode if it contains enough particles:
      bool condition_1 = subnode_ip[ni].size() > quadTree_params->min_count;
      bool condition_2 = depth <= quadTree_params->max_depth;
      if (condition_1 || condition_2)
      {
        subnode[ni]->populate_node();
      }
    }
  }

  return;
}

// =======================================================================================
void quadNode_TYP::get_subnode_bounds(int node_index, arma::vec * min_local, arma::vec * max_local)
{
  switch (node_index)
  {
  case 0: // subnode 0:
      {
        // Attributes for new subnode:
        *min_local = this->center;
        *max_local = this->max;

        // Exit:
        break;
      }
  case 1: // subnode 1:
      {
        // Attributes for new subnode:
        (*min_local)[0] = this->min[0];
        (*min_local)[1] = this->center[1];
        (*max_local)[0] = this->center[0];
        (*max_local)[1] = this->max[1];

        // Exit:
        break;
      }
  case 2: // subnode 2:
      {
        // Attributes for new subnode:
        *min_local = this->min;
        *max_local = this->center;

        // Exit:
        break;
      }
  case 3: // subnode 3:
      {
        // Attributes for new subnode:
        (*min_local)[0] = this->center[0];
        (*min_local)[1] = this->min[1];
        (*max_local)[0] = this->max[0];
        (*max_local)[1] = this->center[1];

        // Exit:
        break;
      }
  }
}

// =======================================================================================
void quadNode_TYP::insert(uint jj)
{
  // Objective:
  // insert point jj into a subnode of the current node. Insert a single level only

  // Current data point:
  double y = (*v)(jj,0);
  double z = (*v)(jj,1);
  arma::vec r = {y,z};

  // Check if data is within node's boundaries:
  // ========================================
  if (!IsPointInsideBoundary(r))
  {
    cout << "point (" << r[0] << "," << r[1] << ") is out of bounds" << endl;
    return;
  }

  // Determine which subnode to insert point:
  // ========================================
  int node_index = WhichSubNodeDoesItBelongTo(r);

  // Store index jj in corresponding subnode_ip:
  // ========================================
  subnode_ip[node_index].push_back(jj);
}

// =======================================================================================
int quadNode_TYP::count_leaf_points(int k)
{
  // Recursive loop:
  for(int nn = 0; nn < 4; nn++)
  {
    // If subnode[nn] exists, drill into it
    if (this->subnode[nn] != NULL)
    {
      k = this->subnode[nn]->count_leaf_points(k);
    }
  }

  // Determine of node is a leaf;
  bool is_leaf = true;
  for(int nn = 0; nn < 4; nn++)
  {
    is_leaf = this->subnode[nn] == NULL && is_leaf;
  }

  // If leaf node, accumulate p_count:
  if (is_leaf == true)
  {
    return k + this->p_count;
  }
  else
  {
    return k;
  }
}

// =======================================================================================
void quadNode_TYP::get_all_leafs(vector<quadNode_TYP *>* leaf_v)
{

  for(int nn = 0; nn < 4; nn++)
  {
    // If subnode[nn] exists, drill into it
    if (this->subnode[nn] != NULL)
    {
      this->subnode[nn]->get_all_leafs(leaf_v);
    }
  }

  // Determine of node is a leaf;
  bool is_leaf = true;
  for(int nn = 0; nn < 4; nn++)
  {
    is_leaf = this->subnode[nn] == NULL && is_leaf;
  }

  // If leaf node, append its pointer:
  if (is_leaf == true)
  {
    leaf_v->push_back(this);
    return;
  }

}

// =======================================================================================
bool quadNode_TYP::IsPointInsideBoundary(arma::vec r)
{
    // Objective:
    // if r is inside the boundaries of the node, return true, otherwise false

    // coordinates of point:
    double y = r[0];
    double z = r[1];

    // Define boundaries of node:
    double y_min = this->min[0];
    double y_max = this->max[0];
    double z_min = this->min[1];
    double z_max = this->max[1];

    // Create boolean result:
    bool flag_y = (y >= y_min) && (y <= y_max);
    bool flag_z = (z >= z_min) && (z <= z_max);

    return flag_y && flag_z;
}

// =======================================================================================
int quadNode_TYP::WhichSubNodeDoesItBelongTo(arma::vec r)
{
    //   |-----------------+------------------+
    //   |  node_left = 1  |  node_right = 0  |
    //   +-----------------o------------------+
    //   |  node_left = 2  |  node_right = 3  |
    //   +-----------------+------------------+

    // Origin of parent node:
    arma::vec r0 = center;

    //  Vector pointing in the direction of point r relative to center of node:
    arma::vec d = r - r0;

    // Number associated with subnode:
    int node_index;

    // Select node index based on vector d
    if (d[0] >= 0) // Quadrants 0 or 3
    {
      if (d[1] >=0) // Quadrant 0
      {
        node_index = 0;
      }
      else // Quadrant 3
      {
        node_index = 3;
      }
    }
    else // Quadrants 1 or 2
    {
      if (d[1] >=0)
      {
        node_index = 1;
      }
      else
      {
        node_index = 2;
      }
    }

    return node_index;
}

// =======================================================================================
bool quadNode_TYP::DoesSubNodeExist(int node_index)
{
    if (NULL == subnode[node_index])
    {
        // Does not exist:
        return 0;
    }
    else
    {
        // It already exists:
        return 1;
    }
}

// =======================================================================================
void quadNode_TYP::clear_node()
{
  // This method just removes the particle count and indices stored in this node. It does not remove the depth, min, max, center information as these are the node parameters and will not change if we use another dataset in this->v;

  this->p_count = 0;
  this->ip.clear();

  for (int nn = 0; nn < 4; nn ++)
  {
    if (NULL != this->subnode[nn])
    {
      this->subnode[nn]->clear_node();
    }
  }
}
