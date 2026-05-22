/// @file block.hpp
#pragma once
#include <graphite/utils.hpp>

namespace graphite {
class BlockCoordinates {
public:
  size_t row;
  size_t col;

  bool operator==(const BlockCoordinates &other) const {
    return (row == other.row) && (col == other.col);
  }
};

using BlockDimension = BlockCoordinates;
} // namespace graphite

namespace std {
template <> struct hash<graphite::BlockCoordinates> {
  size_t operator()(const graphite::BlockCoordinates &bc) const {
    size_t seed = 0;
    graphite::hash_combine(seed, bc.row);
    graphite::hash_combine(seed, bc.col);
    return seed;
  }
};
} // namespace std