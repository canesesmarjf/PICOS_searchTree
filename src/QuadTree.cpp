#include "QuadTree.h"
#include <iostream>

using namespace std;
using namespace arma;

// =======================================================================================
quadTree_TYP::quadTree_TYP()
{
  // All pointers must be made NULL:
  this->quadTree_params = NULL;
  root = NULL;
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
int quadTree_TYP::count_leaf_points()
{
  int k = 0;
  if (this->root != NULL)
    return this->root->count_leaf_points(k);
  else
    return k;
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
  // Check if present node is a leaf:
  int method = 2;
  if (is_node_leaf(method) == true)
  {
    leafs->push_back(this);
  }

  // Traverse the tree:
  for(int nn = 0; nn < 4; nn++)
  {
    // If subnode[nn] exists, drill into it
    if (this->subnode[nn] != NULL)
    {
      this->subnode[nn]->get_leaf_nodes(leafs);
    }
  }
}

// =======================================================================================
void quadNode_TYP::delete_nodes()
{
  for (int nn = 0; nn < 4; nn++)
  {
    if (this->subnode[nn] != NULL) // if subnode[nn] exists
    {
      this->subnode[nn]->delete_nodes();

      // Release memory pointed by subnode[nn]:
      delete this->subnode[nn];

      // Prevent dangling pointer:
      this->subnode[nn] = NULL;

    } // if
  } // for
}

// =======================================================================================
// Overloaded constructor:
quadNode_TYP::quadNode_TYP(arma::vec min, arma::vec max, uint depth, quadTree_params_TYP * quadTree_params,std::vector<uint> ip, arma::mat * v)
{
  // Node "natural" attributes:
  this->min     = min;
  this->max     = max;
  this->depth   = depth;
  this->quadTree_params = quadTree_params;

  // Node "data" attributes:
  this->ip = ip;
  this->v  = v;

  // Derived node "natural" attributes:
  this->center  = (min + max)/2;

  // Derived "data" attributes:
  this->p_count = ip.size();
  this->is_leaf = false;

  // Allocate memory for subnodes:
  this->subnode.reserve(4);
  this->subnode[0] = NULL;
  this->subnode[1] = NULL;
  this->subnode[2] = NULL;
  this->subnode[3] = NULL;
  this->ip_subnode.resize(4);
}

// =======================================================================================
int quadNode_TYP::apply_conditionals_ip_subnode()
{
  // Conditionals:
  bool condition_1 = depth >= quadTree_params->min_depth;
  bool condition_2 = depth < quadTree_params->max_depth;

  // Reject new subnodes by default:
  int accept_new_subnodes = 0;

  if (condition_1 == false) // if depth < min_depth, always accept new subnodes:
  {
    accept_new_subnodes = 1;
  }
  else if (condition_1 && condition_2) // if depth is between [min_depth, max_depth], accept new subnodes if at least ONE of them has enough particles (> min_count)
  {
    // Check if at least one of the new subnodes has counts > min_count:
    // If so, accept new subnodes
    for (int n = 0; n < 4; n++)
    {
      if (ip_subnode[n].size() > quadTree_params->min_count)
      {
        accept_new_subnodes = 1;
        break;
      }
    }
  }
  else // if depth > max_depth, always reject subnodes:
  {
    accept_new_subnodes = 0;
  }

  return accept_new_subnodes;
}


// =======================================================================================
void quadNode_TYP::populate_node()
{
  // Organize point data into ip_subnode to determine if new subnodes need to be created:
  // -------------------------------------------------------------------------------------
  // Propose new subnodes
  calculate_ip_subnode();

  // Run tests to determine if new subnodes are to be accepted:
  // -------------------------------------------------------------------------------------
  // Do we accept new proposed subnodes?
  int accept_new_subnodes = apply_conditionals_ip_subnode();

  // If accepted, create populate subnodes or create them if they dont exist
  // If not accepted, declare parent node a leaf node
  // -------------------------------------------------------------------------------------
  if (accept_new_subnodes == 1)
  {
    // Loop over proposed subnodes.
    // if ip_subnode[n] is not empty, do the following
    // If they already exist, just update subnode[n].ip variable
    // If they dont exist, create them using the new operator

    // Loop over proposed subnodes:
    for (int n = 0; n < 4; n++)
    {
      // Check that proposed subnode has data:
      if (ip_subnode[n].size() > 0)
      {
        // Index data to push into the new subnode[n]:
        vector<uint> ip_local = ip_subnode[n];

        // Create new subnode if it doesnt exist:
        if (subnode[n] == NULL)
        {
          // Could consider creating a method called create_subnode(n,ip)

          // Bounds of new subnode:
          vec min_local(2);
          vec max_local(2);
          get_subnode_bounds(n,&min_local,&max_local);

          // Create subnode:
          uint depth = this->depth + 1;
          subnode[n] = new quadNode_TYP(min_local,max_local,depth,quadTree_params,ip_local,v);
        }
        else // subnode[n] already exists:
        {
          // consider a method called update_subnode(n,ip)

          // Update node "data" attributes of the subnode:
          subnode[n]->ip = ip_local;
          subnode[n]->p_count = ip_local.size();
          subnode[n]->is_leaf = false;
        }

        // Move in deeper:
        subnode[n]->populate_node();
      }
    } // end, for loop

    // Clear ip on parent node since they have now being distributed amongsnt new subnodes:
    this->ip.clear();
    this->p_count = 0;

    // Label parent node as NOT a leaf node:
    this->is_leaf = false;
  }
  else
  {
    // Declare parent node as leaf node
    this->is_leaf = true;
    cout << "leaf_node" << endl;
  }

  // Clear ip_subnode. Data is either not needed OR has been inserted into new subnodes:
  // -------------------------------------------------------------------------------------
  for (int n = 0; n < 4; n++)
  {
    ip_subnode[n].clear();
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
void quadNode_TYP::calculate_ip_subnode()
{
  for (int ii = 0; ii < this->p_count; ii++)
  {
    // Get global index:
    uint jj = this->ip[ii];

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

    // Store index jj in corresponding ip_subnode:
    // ========================================
    ip_subnode[node_index].push_back(jj);
  }
}

// =======================================================================================
bool quadNode_TYP::is_node_leaf(int method)
{
  // This method determines if current node is a leaf node.
  // There are at least two ways to identify leaf_nodes.
  // 1- if ALL subnodes == NULL.
  // 2- if is_leaf == true where the flag is set during the formation of the quad tree in "populate_node()".
  // The second method is favoured because the quad tree heap memory infrastructure can be reutilized many times; whereas the NULL method relies on realeasing the quad-tree memory every time it is to be refreshed because the data *v has changed.

  bool leaf_flag = true;

  switch (method)
  {
    case 1: // NULL method
    {
      // Check all subnodes:
      // If at LEAST ONE subode exist, current node is NOT a leaf:
      for (int nn = 0; nn < 4; nn++)
      {
        if (this->subnode[nn] != NULL)
        {
          leaf_flag = false;
          break;
        }
      }
      break;
    }
    case 2: // Read is_leaf variable
    {
      leaf_flag = this->is_leaf;
      break;
    }
    default:
    {
      // Error: Invalid method
      std::cerr << "Invalid method. Choose either method 1 or method 2." << std::endl;
      break;
    }
  }

  return leaf_flag;
}

// =======================================================================================
int quadNode_TYP::count_leaf_points(int k)
{
  // This method counts all the points present in a quad tree
  // First, it deterines if present node is a leaf.
  // If so, it accumulates p_count
  // If not so, it proceeds to traverse tree using recursion
  // Every time a new node is accessed, leaf status is checked.

  // If current is node is a leaf node, accumulate p_count and return to calling stack:
  int method = 2;
  if (is_node_leaf(method) == true)
  {
    return k + this->p_count;
  }

  // Traverse the tree:
  for(int nn = 0; nn < 4; nn++)
  {
    // If subnode[nn] exists, drill into it
    if (this->subnode[nn] != NULL)
    {
      k = this->subnode[nn]->count_leaf_points(k);
    }
  }

  // Return to calling stack:
  return k;
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
  this->is_leaf = false;

  for (int nn = 0; nn < 4; nn ++)
  {
    if (NULL != this->subnode[nn])
    {
      this->subnode[nn]->clear_node();
    }
  }
}
