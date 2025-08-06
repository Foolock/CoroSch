#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <numeric> 
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <coroutine>

namespace cs {

// graph structure
class Node;
class Edge;
class Graph;

class Node {

friend class Graph;

public:

  Node(const std::string& name=""): _name(name) {}

private:

  std::string _name;
  std::list<Edge*> _fanouts;
  std::list<Edge*> _fanins;

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

  // operation
  Node* insert_node(const std::string& name = "");
  Edge* insert_edge(Node* from, Node* to);

private:

  // use list instead of vector to prevent messing up pointer addressing due to reallocation in vector
  std::list<Node> _nodes;
  std::list<Edge> _edges;

};

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

































