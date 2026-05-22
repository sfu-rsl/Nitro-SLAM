// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_BLOCK_SOLVER2_H
#define G2O_BLOCK_SOLVER2_H
#include <Eigen/Core>
#include "../core/solver.h"
#include "../core/openmp_mutex.h"
#include "../../config.h"
#include "../gpu/BAStats.hpp"

#include <solver/backend.hpp>
#include <solver/sparse_block_matrix.hpp>
#include <solver/linear_solver.hpp>
#include <solver/pcg.hpp>

#include <vector>
#include <future>

namespace g2o
{
  using namespace Eigen;

  class BlockSolver2 : public BlockSolverBase
  {
  public:
    /**
     * allocate a block solver ontop of the underlying linear solver.
     * NOTE: The BlockSolver assumes exclusive access to the linear solver and will therefore free the pointer
     * in its destructor.
     */
    BlockSolver2(compute::ComputeEngine *engine, compute::LinearSolver<double> *linearSolver);
    ~BlockSolver2();

    virtual bool init(SparseOptimizer *optmizer, bool online = false);
    virtual bool buildStructure(bool zeroBlocks = false);
    virtual bool updateStructure(const std::vector<HyperGraph::Vertex *> &vset, const HyperGraph::EdgeSet &edges);
    virtual bool buildSystem();
    virtual bool solve();
    virtual bool computeMarginals(SparseBlockMatrix<MatrixXd> &spinv, const std::vector<std::pair<int, int>> &blockIndices);
    virtual bool setLambda(double lambda, bool backup = false);
    virtual void restoreDiagonal();
    virtual bool supportsSchur() { return true; }
    virtual bool schur() { return _doSchur; }
    virtual void setSchur(bool s) { _doSchur = s; }

    compute::LinearSolver<double> *linearSolver() const { return _linearSolver; }

    virtual void setWriteDebug(bool writeDebug);
    virtual bool writeDebug() const
    {
      return false;
    }

    virtual bool saveHessian(const std::string &fileName) const;

    virtual void multiplyHessian(double *dest, const double *src) const
    {
      throw std::runtime_error("BlockSolver2: multiplyHessian not implemented!");
    }

    // gpu timing
    void setBAStatistics(gpu::BAStats::BAStatistics *ba_stats) { _ba_stats = ba_stats; }

    void setImplicitSchur(bool enable)
    {
      _use_implicit_schur = enable;
    }

    void setAllocType(compute::BufferType allocation_type) {
      _alloc_type = allocation_type;
    }

  protected:
    void resize(const std::vector<BlockIndex> &blockPoseIndices,
                int numPoseBlocks,
                const std::vector<BlockIndex> &blockLandmarkIndices,
                int numLandmarkBlocks, int totalDim);

    void deallocate();

    bool implicit_schur();

    void mapMemory();


    compute::ComputeEngine *engine;

    compute::MatPtr<double> _Hpp;
    compute::MatPtr<double> _Hll;
    compute::MatPtr<double> _Hllinv;
    compute::MatPtr<double> _Hpl;
    compute::MatPtr<double> _HplinvHll;
    compute::MatPtr<double> _Hschur;
    compute::BufferPtr<double> _bl;
    compute::BufferPtr<double> _bschur;
    compute::BufferPtr<double> _xp;
    compute::BufferPtr<double> _xl;
    compute::BufferPtr<double> _lambda;

    std::shared_ptr<compute::SolverSeq> sync_H;
    std::shared_ptr<compute::SolverSeq> sync_x;

    std::shared_ptr<compute::SolverSeq> mult_HplinvHll;
    std::shared_ptr<compute::SolverSeq> mult_bschur;
    std::shared_ptr<compute::SolverSeq> mult_Hschur;

    std::shared_ptr<compute::SolverSeq> mult_HplTxp;
    std::shared_ptr<compute::SolverSeq> mult_Hllinv;

    std::shared_ptr<compute::SolverSeq> inversion_op;

    std::shared_ptr<compute::SolverSeq> Schur_seq2;
    std::shared_ptr<compute::SolverSeq> Schur_seq;
    std::shared_ptr<compute::SolverSeq> bschur_seq;
    std::shared_ptr<compute::SolverSeq> set_lambda_seq;
    std::shared_ptr<compute::SolverSeq> restore_diagonal_seq;

    std::shared_future<void> task_rec_Hschur;
    std::shared_future<void> task_rec_multiply;
    std::shared_future<void> task_reserve_Hschur;
    std::shared_future<void> gpu_alloc_task;
    std::shared_future<void> task_solver_setup;
    std::shared_future<void> task_rec_landmark;
    std::shared_future<void> task_init_Hschur;

    compute::LinearSolver<double> *_linearSolver;

    std::vector<VectorXd, Eigen::aligned_allocator<VectorXd>> _diagonalBackupPose;
    std::vector<VectorXd, Eigen::aligned_allocator<VectorXd>> _diagonalBackupLandmark;


    bool _doSchur;

    bool _use_implicit_schur = false;

    int _numPoses, _numLandmarks;
    int _sizePoses, _sizeLandmarks;

    // gpu
    gpu::BAStats::BAStatistics *_ba_stats;
    compute::BufferType _alloc_type;


  };

  typedef BlockSolver2 BlockSolver2X;

} // end namespace

#include "block_solver2.hpp"

#endif
