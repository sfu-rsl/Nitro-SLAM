#pragma once
#include <stdint.h>
#include <utility>
#include <Eigen/Dense>
// Aliases
using BlockIndex = uint32_t;
using BlockPos = std::pair<BlockIndex, BlockIndex>;
using Dim = uint16_t;
using BlockDim = std::pair<Dim, Dim>;
using BlockDimPair = std::pair<BlockDim, BlockDim>;
// Matrix Aliases
template <typename DataType, enum Eigen::StorageOptions Storage = Eigen::StorageOptions::ColMajor>
using DynamicMatrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Storage>;

template <typename DataType, enum Eigen::StorageOptions Storage = Eigen::StorageOptions::ColMajor>
using DynamicMap = Eigen::Map<DynamicMatrix<DataType, Storage>>;

template <typename DataType>
using MappedPair = std::pair<DynamicMap<DataType>, DynamicMap<DataType>>;

namespace compute
{
    struct BlockPosHash
    {
        std::size_t operator()(BlockPos const &pos) const noexcept
        {
            // std::size_t h1 = std::hash<BlockIndex>{}(pos.first);
            // std::size_t h2 = std::hash<BlockIndex>{}(pos.second);
            // return h1 ^ (h2 << 1);

            std::size_t h1 = pos.first;
            return (h1 << 32) | pos.second;
        }
    };

    struct BlockDimHash
    {
        std::size_t operator()(BlockDim const &dim) const noexcept
        {
            // std::size_t h1 = std::hash<BlockIndex>{}(dim.first);
            // std::size_t h2 = std::hash<BlockIndex>{}(dim.second);
            // return h1 ^ (h2 << 1);

            std::size_t h1 = dim.first;
            return (h1 << 16) | dim.second;
        }
    };

    struct BlockDimPairHash
    {
        std::size_t operator()(std::pair<BlockDim, BlockDim> const &dims) const noexcept
        {

            const auto dim1 = dims.first;
            const auto dim2 = dims.second;

            std::size_t h1 = BlockDimHash{}(dim1);
            std::size_t h2 = BlockDimHash{}(dim2);

            // return h1 ^ (h2 << 1);
            return (h1 << 32) | h2;
        }
    };



    class MulOp
    {
    public:
        MulOp(uint32_t a, uint32_t b, uint32_t c) : a(a), b(b), c(c)
        {
        }

        uint32_t a;
        uint32_t b;
        uint32_t c;
    };

    class MulPair
    {
    public:
        MulPair(uint32_t a, uint32_t b) : a(a), b(b) {}
        uint32_t a, b;
    };

    class MulList
    {
    public:
        MulList(uint32_t start, uint32_t n, uint32_t dest) : start(start), n(n), dest(dest) {}
        uint32_t start, n, dest;
    };

    class MulBlockDiagList
    {
    public:
        MulBlockDiagList(uint32_t start, uint32_t n, uint32_t block) : start(start), n(n), block(block) {}

        uint32_t start, n, block;
    };

}
