#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <numeric> 
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <coroutine>
#include "coro_scheduler.hpp"

namespace cs {

// graph structure
class Node;
class Edge;
class Graph;

class Node {

friend class Graph;

public:

  Node(const std::string& name=""): _name(name) {}

  inline
  cs::Task* get_task_coro() { return _task_coro; }

private:

  std::string _name;
  std::list<Edge*> _fanouts;
  std::list<Edge*> _fanins;

  cs::Task* _task_coro;
};

class Edge {

friend class Graph;

private:

  Node* _from;
  Node* _to;

};

class Graph {

public:

  // constructor
  Graph() {} // default
  Graph(const std::string& filename);

  // operation
  Node* insert_node(const std::string& name = "");
  Edge* insert_edge(Node* from, Node* to);

private:

  // use list instead of vector to prevent messing up pointer addressing due to reallocation in vector
  std::list<Node> _nodes;
  std::list<Edge> _edges;

};

inline
Graph::Graph(const std::string& filename) { // construct by circuit file

  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "Error opening file." << std::endl;
    std::exit(1);
  } 

  int num_nodes;
  infile >> num_nodes; // Read the number of nodes

  std::unordered_map<std::string, Node*> nodes;

  // Read node names and add them to the graph
  std::string node_name;
  for (int i = 0; i < num_nodes; ++i) {
    infile >> node_name;
    // Remove quotes from node name
    node_name = node_name.substr(1, node_name.size() - 3);
    nodes[node_name] = insert_node(node_name);
  }

  // Read edges and add them to the graph
  std::string from, to, arrow;
  while (infile >> from >> arrow >> to) {
    from = from.substr(1, from.size() - 2);
    to = to.substr(1, to.size() - 3);
    insert_edge(nodes[from], nodes[to]);
  }
}

inline
Node* Graph::insert_node(const std::string& name) {

  // emplace new node to _nodes
  Node* node_ptr = &(_nodes.emplace_back(name));

  return node_ptr;
}

inline
Edge* Graph::insert_edge(Node* from, Node* to) {

  // emplace new edge to _edges
  Edge* edge_ptr = &(_edges.emplace_back());

  // add edge attributes
  edge_ptr->_from = from;
  edge_ptr->_to = to;

  // add node fanin/fanout
  from->_fanouts.push_back(edge_ptr);
  to->_fanins.push_back(edge_ptr);

  return edge_ptr;
}

} // end of namespace cs 

































