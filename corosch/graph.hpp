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
#include "taskflow/taskflow.hpp"
#include "work_stealingN.hpp"  // non-coroutine version

namespace cs {

// graph structure
class Node;
class Edge;
class Graph;

class Node {

friend class Graph;

public:

  Node(const std::string& name=""): _name(name) {}

  // Accessors
  const std::string& name() const { return _name; }
  const std::list<Edge*>& fanouts() const { return _fanouts; }
  const std::list<Edge*>& fanins() const { return _fanins; }

  // task setter and getter
  inline
  void set_task_coro( cs::Task* t ) { _task_coro = t; }
  inline
  cs::Task* get_task_coro() { return _task_coro; }
  inline
  void set_task_non_coro( cs::NTask* t ) { _task_non_coro = t; }
  inline
  cs::NTask* get_task_non_coro() { return _task_non_coro; }
  inline
  void set_task_tf(tf::Task t) { _task_tf = std::move(t); }
  inline
  tf::Task& get_task_tf() { return _task_tf; }   

  inline
  void set_task_tf_status( bool done ) { _task_tf_done = done; }
  bool get_task_tf_status() { return _task_tf_done; }

private:

  std::string _name;
  std::list<Edge*> _fanouts;
  std::list<Edge*> _fanins;

  cs::Task* _task_coro;
  cs::NTask* _task_non_coro;
  tf::Task _task_tf;
  bool _task_tf_done = false;
};

class Edge {

friend class Graph;

public:
  Node* from() const { return _from; }
  Node* to() const { return _to; }

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

  // read-only access
  inline
  const std::list<Node>& nodes() const { return _nodes; }
  // read/write access
  inline
  std::list<Node>& nodes() { return _nodes; }

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

































