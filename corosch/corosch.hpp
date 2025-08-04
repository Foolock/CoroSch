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
class CoroSch;

class Node {

friend class CoroSch;

public:

  Node(const std::string& name=""): _name(name) {}

  inline void set_done() { _done_coro = true; }

private:

  std::string _name;
  std::list<Edge*> _fanouts;
  std::list<Edge*> _fanins;

  bool _done_coro = false;
};

class Edge {

friend class CoroSch;

private:

  Node* _from;
  Node* _to;

};

class CoroSch {

public:

  // constructor
  CoroSch() {} // default

  // operation
  Node* insert_node(const std::string& name = "");
  Edge* insert_edge(Node* from, Node* to);

private:

  // use list instead of vector to prevent messing up pointer addressing due to reallocation in vector
  std::list<Node> _nodes;
  std::list<Edge> _edges;

};

} // end of namespace cs 

































