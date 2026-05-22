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

#include "../core/sparse_optimizer.h"
#include <Eigen/LU>
#include <fstream>
#include <iomanip>

#include "../stuff/timeutil.h"
#include "../stuff/macros.h"
#include "../stuff/misc.h"
#include "../gpu/BAStats.hpp"
namespace g2o {

using namespace std;
using namespace Eigen;

BlockSolver2::BlockSolver2(compute::ComputeEngine* engine, compute::LinearSolver<double>* linearSolver) :
  BlockSolverBase(),
  engine(engine),
  _linearSolver(linearSolver)
{
  // std::cout << "Constructing BlockSolver2!\n";

  _xSize=0;
  _numPoses=0;
  _numLandmarks=0;
  _sizePoses=0;
  _sizeLandmarks=0;
  _doSchur=true;
  _use_implicit_schur = false;
  // gpu
  _ba_stats = nullptr;
  _alloc_type = compute::BufferType::DeviceCached;

}


void BlockSolver2::resize(const std::vector<BlockIndex> & blockPoseIndices,
      int numPoseBlocks,
      const std::vector<BlockIndex> & blockLandmarkIndices,
      int numLandmarkBlocks, int s
      ) {
  deallocate();

  resizeVector(s);

  assert(_sizePoses > 0 && "allocating with wrong size");
  _bschur = engine->create_buffer<double>(nullptr, _sizePoses, _alloc_type); // same size as _bp
  _xp = engine->create_buffer<double>(nullptr, _sizePoses, _alloc_type);

  if (_doSchur) {
    _xl = engine->create_buffer<double>(nullptr, _sizeLandmarks, _alloc_type);
    _bl = engine->create_buffer<double>(nullptr, _sizeLandmarks, _alloc_type);
    sync_x = engine->create_op_sequence();
    sync_x->sync_device<double>({_xp, _xl});
  }

  // _Hpp=new PoseHessianType(blockPoseIndices, blockPoseIndices, numPoseBlocks, numPoseBlocks);
  _Hpp = std::make_shared<compute::SparseBlockMatrix<double>>(*engine);
  _Hpp->resize(blockPoseIndices, blockPoseIndices);
  if (_doSchur) {
    if (!implicit_schur()) {
    _Hschur = std::make_shared<compute::SparseBlockMatrix<double>>(*engine);
    _Hschur->resize(blockPoseIndices, blockPoseIndices);
    }
    _Hll = std::make_shared<compute::SparseBlockMatrix<double>>(*engine);
    _Hll->resize(blockLandmarkIndices, blockLandmarkIndices);

    _Hllinv = std::make_shared<compute::SparseBlockMatrix<double>>(*engine);
    _Hllinv->resize(blockLandmarkIndices, blockLandmarkIndices);
    _Hpl = std::make_shared<compute::SparseBlockMatrix<double>>(*engine);
    _Hpl->resize(blockPoseIndices, blockLandmarkIndices);

    if (!implicit_schur()) {
      _HplinvHll = std::make_shared<compute::SparseBlockMatrix<double>>(*engine);
      _HplinvHll->resize(blockPoseIndices, blockLandmarkIndices);
    }

  }
}

void BlockSolver2::deallocate()
{

  // std::cout << "BlockSolver2: Deallocating!\n";
  // auto t0 = get_monotonic_time();
  // wait for all tasks 
  if (gpu_alloc_task.valid()) {
    gpu_alloc_task.get();
  }

  if (task_init_Hschur.valid()) {
    task_init_Hschur.get();
  }

  if (task_reserve_Hschur.valid()) {
    task_reserve_Hschur.get();
  }

  if (task_solver_setup.valid()) {
    task_solver_setup.get();
  }

  if (task_rec_landmark.valid()) {
    task_rec_landmark.get();
  }

  if (task_rec_multiply.valid()) {
    task_rec_multiply.get();
  }
  if (task_rec_Hschur.valid()) {
    task_rec_Hschur.get();
  }
  // auto t1 = get_monotonic_time();
  // std::cout << "BlockSolver2: Waiting for tasks took " << t1-t0 << " seconds.\n";
  // Multiplication objects
  engine->recycle_sequence(mult_HplinvHll);
  engine->recycle_sequence(mult_bschur);
  engine->recycle_sequence(mult_Hschur);
  engine->recycle_sequence(mult_HplTxp);
  engine->recycle_sequence(mult_Hllinv);
  engine->recycle_sequence(inversion_op);

  mult_HplinvHll.reset();
  mult_bschur.reset();
  mult_Hschur.reset();
  mult_HplTxp.reset();
  mult_Hllinv.reset();
  inversion_op.reset();
  // auto t2 = get_monotonic_time();
  // std::cout << "BlockSolver2: Waiting for mult objects took " << t2-t1 << " seconds.\n";
  // Sequences

  engine->recycle_sequence(Schur_seq2);
  engine->recycle_sequence(Schur_seq);
  engine->recycle_sequence(bschur_seq);
  engine->recycle_sequence(set_lambda_seq);
  engine->recycle_sequence(restore_diagonal_seq);

  Schur_seq2.reset();
  Schur_seq.reset();
  bschur_seq.reset();
  set_lambda_seq.reset();
  restore_diagonal_seq.reset();

  // auto t3 = get_monotonic_time();
  // std::cout << "BlockSolver2: Waiting for sequences took " << t3-t2 << " seconds.\n";

  // Sync objects
  engine->recycle_sequence(sync_H);
  engine->recycle_sequence(sync_x);
  sync_H.reset();
  sync_x.reset();
  // auto t4 = get_monotonic_time();
  // std::cout << "BlockSolver2: Waiting for sync objects took " << t4-t3 << " seconds.\n";

  // Matrices
  _Hpp.reset();
  _Hll.reset();
  _Hpl.reset();
  _Hschur.reset();
  _Hllinv.reset();
  _HplinvHll.reset();

  // auto t5 = get_monotonic_time();
  // std::cout << "BlockSolver2: Waiting for matrices took " << t5-t4 << " seconds.\n";

  // Vectors
  _bschur.reset();
  _bl.reset();
  _xp.reset();
  _xl.reset();
  _lambda.reset();
  // auto t6 = get_monotonic_time();
  // std::cout << "BlockSolver2: Waiting for vectors took " << t6-t5 << " seconds.\n";
  // std::cout<< "Deallocated!\n";
}

BlockSolver2::~BlockSolver2()
{
  // backend-specific
  #ifndef SOLVER_CPU_BACKEND
  auto stats = _ba_stats;
  if (stats) {
    auto vma_stats = engine->get_vma_statistics();
    stats->mem_block_bytes = vma_stats.total.statistics.blockBytes;
    stats->mem_allocation_bytes = vma_stats.total.statistics.allocationBytes;
  }
  #endif

  delete _linearSolver;
  deallocate();
}

bool BlockSolver2::implicit_schur()
{
  if (_use_implicit_schur)
  {
    auto solver = dynamic_cast<compute::ImplicitSchurSolver<double> *>(_linearSolver);
    if (!solver)
    {
      throw std::runtime_error("BlockSolver2: Configured for implict method but LinearSolver does not support it!");
    }
    return true;
  }
  return false;
}

bool BlockSolver2::buildStructure(bool zeroBlocks)
{

  #ifdef ASYNC_DEFERRED
  constexpr auto async_mode = std::launch::deferred;
  #else
  constexpr auto async_mode = std::launch::async;
  #endif
  // std::cout << "BlockSolver2: buildStructure!\n";

  auto tr0 = std::chrono::high_resolution_clock::now();
  assert(_optimizer);
  size_t sparseDim = 0;
  _numPoses=0;
  _numLandmarks=0;
  _sizePoses=0;
  _sizeLandmarks=0;

  std::vector<BlockIndex> blockPoseIndices;
  blockPoseIndices.reserve(_optimizer->indexMapping().size()+1);
  std::vector<BlockIndex> blockLandmarkIndices;
  blockLandmarkIndices.reserve(_optimizer->indexMapping().size()+1);
  // std::cout << "BlockSolver2: buildStructure - Finding dimensions...\n";


  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    int dim = v->dimension();
    if (! v->marginalized()){
      v->setColInHessian(_sizePoses);
      blockPoseIndices.push_back(_sizePoses);
      _sizePoses+=dim;
      ++_numPoses;
    } else {
      v->setColInHessian(_sizeLandmarks);
      blockLandmarkIndices.push_back(_sizeLandmarks);
      _sizeLandmarks+=dim;
      ++_numLandmarks;
    }
    sparseDim += dim;
  }

  if (_sizeLandmarks && _sizePoses) {
    blockPoseIndices.push_back(_sizePoses);
    blockLandmarkIndices.push_back(_sizeLandmarks);
  }
  else if (_sizePoses && !_doSchur) {
    blockPoseIndices.push_back(_sizePoses);
  }
  else {
    throw std::runtime_error("BlockSolver2: Landmark or pose size was 0!");
  }

  // std::cout << "Resizing!";
  resize(blockPoseIndices, _numPoses, blockLandmarkIndices, _numLandmarks, sparseDim);
  // std::cout << "Done resizing!";

  // allocate the diagonal on Hpp and Hll
  int poseIdx = 0;
  int landmarkIdx = 0;
  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    if (! v->marginalized()){
      _Hpp->reserve_block(poseIdx, poseIdx);
      ++poseIdx;
    } else {
      _Hll->reserve_block(landmarkIdx, landmarkIdx);
      ++landmarkIdx;
    }
  }
    assert(poseIdx == _numPoses && landmarkIdx == _numLandmarks);

  if (_Hschur) {
    _Hschur->take_structure_from(_Hpp);
  }


  // code for reserving non-diagonals in the block matrices
  // here we assume that the landmark indices start after the pose ones
  // create the structure in Hpp, Hll and in Hpl
  for (SparseOptimizer::EdgeContainer::const_iterator it=_optimizer->activeEdges().begin(); it!=_optimizer->activeEdges().end(); ++it){
    OptimizableGraph::Edge* e = *it;

    for (size_t viIdx = 0; viIdx < e->vertices().size(); ++viIdx) {
      OptimizableGraph::Vertex* v1 = (OptimizableGraph::Vertex*) e->vertex(viIdx);
      int ind1 = v1->hessianIndex();
      if (ind1 == -1)
        continue;
      int indexV1Bak = ind1;
      for (size_t vjIdx = viIdx + 1; vjIdx < e->vertices().size(); ++vjIdx) {
        OptimizableGraph::Vertex* v2 = (OptimizableGraph::Vertex*) e->vertex(vjIdx);
        int ind2 = v2->hessianIndex();
        if (ind2 == -1)
          continue;
        ind1 = indexV1Bak;
        bool transposedBlock = ind1 > ind2;
        if (transposedBlock){ // make sure, we allocate the upper triangle block
          swap(ind1, ind2);
        }
        if (! v1->marginalized() && !v2->marginalized()){
          _Hpp->reserve_block(ind1, ind2);
          if (_Hschur) {// assume this is only needed in case we solve with the schur complement
            _Hschur->reserve_block(ind1, ind2);
          }
        } else if (v1->marginalized() && v2->marginalized()){
          _Hll->reserve_block(ind1-_numPoses, ind2-_numPoses);
        } else { 
          if (v1->marginalized()){ 
            _Hpl->reserve_block(v2->hessianIndex(), v1->hessianIndex()-_numPoses);
          } else {
            _Hpl->reserve_block(v1->hessianIndex(),v2->hessianIndex()-_numPoses);
          }
        }
      }
    }
  }

  if (! _doSchur) {
    /* 
      Since Hpp structure is known, we can exit early, but we need to do a few things first:
      - Buffers for Hpp must be created
      - Linear solver needs to be set up for Hpp
      - Need to also set up the set/restore operations
      - And also the transfer operations
      - Finally, map H
    */

    // Allocate Hpp
    gpu_alloc_task = std::async(async_mode, [_alloc_type=_alloc_type, engine=engine, _Hpp=_Hpp, zeroBlocks, 
    &sync_H=sync_H, &set_lambda_seq=set_lambda_seq, &restore_diagonal_seq=restore_diagonal_seq, &_lambda=_lambda, 
    &sync_x=sync_x, _xp=_xp, _bschur=_bschur, _linearSolver=_linearSolver, &Schur_seq=Schur_seq]() {
      _Hpp->allocate_memory(_alloc_type);

      if (zeroBlocks) {
        _Hpp->zero_memory();
      }

      // Set up the set/restore diagonal operations
      _lambda = engine->create_buffer<double>(nullptr, 1, compute::BufferType::DeviceCached);
      set_lambda_seq = engine->create_op_sequence();
      restore_diagonal_seq = engine->create_op_sequence();
      set_lambda_seq->sync_device<double>({_lambda});
      set_lambda_seq->insert_tc_barrier();
      _Hpp->set_lambda(_lambda, set_lambda_seq, restore_diagonal_seq);

      // Record transfer operations

      // we need to transfer _Hpp and _bp to device memory
      sync_H = engine->create_op_sequence();
      sync_H->sync_device<double>({_Hpp->get_buffer(), _bschur});

      // an operation for syncing _Hpp back
      if (!_linearSolver->result_gpu()) { // this implies it's a CPU solver
        Schur_seq = engine->create_op_sequence();
        Schur_seq->sync_local<double>({_Hpp->get_buffer()});
      }

      // we also need to possibly transfer _xp back to host memory
      if (_linearSolver->result_gpu()) {
        sync_x = engine->create_op_sequence();
        sync_x->sync_local<double>({_xp});
      }

    });

    #ifdef SERIALIZE_PIPELINE
    gpu_alloc_task.wait();
    #endif

    // Set up linear solver
    task_solver_setup = std::async(async_mode, [_linearSolver=_linearSolver, 
    gpu_alloc_task=gpu_alloc_task, _Hpp=_Hpp, _xp=_xp, _bschur=_bschur]() {
      gpu_alloc_task.get();
      auto row_sort = std::async(async_mode, [&]() {_Hpp->sort_row_indices();});
      auto col_sort = std::async(async_mode, [&]() {_Hpp->sort_col_indices();});
      row_sort.get();
      col_sort.get();
      _linearSolver->setup(_Hpp, _xp, _bschur);
    });

    #ifdef SERIALIZE_PIPELINE
    task_solver_setup.wait();
    #endif

    mapMemory();

    return true;
  }


    auto tr1 = std::chrono::high_resolution_clock::now();
  // std::cout << "Reserve time: " << std::chrono::duration<double>(tr1-tr0).count() << "\n";

  // end of code for reserving matrices
  auto tprep = get_monotonic_time();
  bool explicit_schur = !implicit_schur();
  // code for allocating matrices (slow)
  gpu_alloc_task = std::async(async_mode, [_alloc_type=_alloc_type, engine=engine, _Hpp=_Hpp, _Hll=_Hll, _Hpl=_Hpl, zeroBlocks, 
   &sync_H=sync_H, &set_lambda_seq=set_lambda_seq, &restore_diagonal_seq=restore_diagonal_seq, &_lambda=_lambda]() {
    _Hpp->allocate_memory(_alloc_type);
    _Hll->allocate_memory(_alloc_type);
    _Hpl->allocate_memory(_alloc_type);
    // prepare sync stuff
    // std::cout << "Rec sync H!";
    sync_H = engine->create_op_sequence();
    sync_H->sync_device<double>({_Hpp->get_buffer(), _Hll->get_buffer(), _Hpl->get_buffer()});
    // std::cout << "Done rec sync H!";
    if (zeroBlocks) {
      _Hpl->zero_memory();
      _Hll->zero_memory();
      _Hpp->zero_memory();
    }

    _lambda = engine->create_buffer<double>(nullptr, 1, compute::BufferType::DeviceCached);
    set_lambda_seq = engine->create_op_sequence();
    restore_diagonal_seq = engine->create_op_sequence();
    set_lambda_seq->sync_device<double>({_lambda});
    set_lambda_seq->insert_tc_barrier();

    _Hpp->set_lambda(_lambda, set_lambda_seq, restore_diagonal_seq);
    _Hll->set_lambda(_lambda, set_lambda_seq, restore_diagonal_seq);
  });

  #ifdef SERIALIZE_PIPELINE
  gpu_alloc_task.wait();
  #endif

  
  std::shared_future<void> gpu_alloc2 = std::async(async_mode, [_Hpl=_Hpl, _Hll=_Hll, _HplinvHll=_HplinvHll, _Hllinv=_Hllinv, gpu_alloc_task=gpu_alloc_task]() {
      gpu_alloc_task.get();
      auto buf_type = _Hpl->get_buffer()->get_buffer_type() == compute::BufferType::DeviceCached ? compute::BufferType::Storage : compute::BufferType::Host;
      if (_HplinvHll) {
        _HplinvHll->take_structure_from(_Hpl);
        _HplinvHll->allocate_memory(buf_type);
      }

      _Hllinv->take_structure_from(_Hll);
      _Hllinv->allocate_memory(buf_type);
  });

  #ifdef SERIALIZE_PIPELINE
  gpu_alloc2.wait();
  #endif


  task_reserve_Hschur = std::async(async_mode, [_alloc_type=_alloc_type, _Hschur=_Hschur, _optimizer=_optimizer, explicit_schur]() {

    if (explicit_schur) {
        for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
        OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
        if (v->marginalized()){
          const HyperGraph::EdgeSet& vedges=v->edges();
          for (HyperGraph::EdgeSet::const_iterator it1=vedges.begin(); it1!=vedges.end(); ++it1){
            for (size_t i=0; i<(*it1)->vertices().size(); ++i)
            {
              OptimizableGraph::Vertex* v1= (OptimizableGraph::Vertex*) (*it1)->vertex(i);
              if (v1->hessianIndex()==-1 || v1==v)
                continue;
              for  (HyperGraph::EdgeSet::const_iterator it2=vedges.begin(); it2!=vedges.end(); ++it2){
                for (size_t j=0; j<(*it2)->vertices().size(); ++j)
                {
                  OptimizableGraph::Vertex* v2= (OptimizableGraph::Vertex*) (*it2)->vertex(j);
                  if (v2->hessianIndex()==-1 || v2==v)
                    continue;
                  int i1=v1->hessianIndex();
                  int i2=v2->hessianIndex();
                  if (i1<=i2) {
                    _Hschur->reserve_block(i1, i2);
                  }
                }
              }
            }
          }
        }
      }

        // also sort Hschur column indices for conversion to CSC later
      auto row_sort = std::async(async_mode, [&]() {_Hschur->sort_row_indices();});
      auto col_sort = std::async(async_mode, [&]() {_Hschur->sort_col_indices();});
      _Hschur->allocate_memory(_alloc_type);

      row_sort.get();
      col_sort.get();
    }


  });


    #ifdef SERIALIZE_PIPELINE
    task_reserve_Hschur.wait();
    #endif

    task_solver_setup = std::async(async_mode, [task_reserve_Hschur=task_reserve_Hschur, _linearSolver=_linearSolver, 
    gpu_alloc_task=gpu_alloc_task, gpu_alloc2, _Hschur=_Hschur, _Hpp=_Hpp, _Hpl=_Hpl, _Hllinv=_Hllinv, _xp=_xp, _bschur=_bschur, explicit_schur]() {
    
    if (explicit_schur) {
      task_reserve_Hschur.get();
      _linearSolver->setup(_Hschur, _xp, _bschur);
    }
    else {
      task_reserve_Hschur.get();
      gpu_alloc_task.get();
      gpu_alloc2.get();
      auto solver = dynamic_cast<compute::ImplicitSchurSolver<double>*>(_linearSolver);
      solver->setup(_Hpp, _Hpl, _Hllinv, _xp, _bschur);
      }
  });

  #ifdef SERIALIZE_PIPELINE
  task_solver_setup.wait();
  #endif


  task_rec_multiply = std::async(async_mode, [gpu_alloc_task=gpu_alloc_task, gpu_alloc2,  explicit_schur, 
  &mult_HplinvHll=mult_HplinvHll, &mult_bschur=mult_bschur, &inversion_op=inversion_op,
  _bschur=_bschur, _bl=_bl, _xl=_xl, _xp=_xp,
  _HplinvHll=_HplinvHll, _Hllinv=_Hllinv, _Hpl=_Hpl, _Hll=_Hll, engine=engine]() {
    
    gpu_alloc_task.get();
    gpu_alloc2.get();
    // Queue up multiplications, can be done in parallel since structures are known

    // inversion op
    auto f6 = std::async(async_mode, [&]() {
      inversion_op = engine->create_op_sequence();
      _Hll->create_inversion_op(inversion_op, _Hllinv);
    });

    if (explicit_schur) {
      // HplinvHll += Hpl*Hll^-1;
      auto f1 = std::async(async_mode, [&]() {
        mult_HplinvHll = engine->create_op_sequence();
        _Hpl->subgroup_block_diagonal_multiply_add(mult_HplinvHll, _HplinvHll, _Hllinv);
      });
      // bschur -= HplinvHll*bl
      auto f2 = std::async(async_mode, [&]() {
        mult_bschur = engine->create_op_sequence();
        _HplinvHll->subgroup_multiply_vec_add(mult_bschur, _bschur, _bl, false); // subtraction into _coeff
      });

      f1.get();
      f2.get();

    }

    f6.get();

  });

  #ifdef SERIALIZE_PIPELINE
  task_rec_multiply.wait();
  #endif

task_rec_landmark = std::async(async_mode, [gpu_alloc_task=gpu_alloc_task, gpu_alloc2, 
  &mult_HplTxp=mult_HplTxp, &mult_Hllinv=mult_Hllinv,
  _bl=_bl, _xl=_xl, _xp=_xp,
  _Hllinv=_Hllinv, _Hpl=_Hpl, engine=engine]() {

    gpu_alloc_task.get();
    gpu_alloc2.get();
    // Queue up multiplications, can be done in parallel since structures are known

    // reuse _bl to store result of -bl+HplT*xp
    // c = bl
    // c -= xp*Hpl
    auto f4 = std::async(async_mode, [&]() {
      mult_HplTxp = engine->create_op_sequence();
      _Hpl->subgroup_right_multiply_vec_add(mult_HplTxp, _bl, _xp, false); // store result in bl 

    });
    // xl = Hll^-1 * c
    auto f5 = std::async(async_mode, [&]() {
      mult_Hllinv = engine->create_op_sequence();
      _Hllinv->subgroup_block_diagonal_multiply_vec_add(mult_Hllinv, _xl, _bl); // final multiplication
      mult_Hllinv->insert_ct_barrier(); // may be redundant for vk
      mult_Hllinv->sync_local<double>({_xp, _xl});
    });
        

    f4.get();
    f5.get();    

  });


  #ifdef SERIALIZE_PIPELINE
  task_rec_landmark.wait();
  #endif

  // Hschur -= HplinvHll*HplT
  task_rec_Hschur = std::async(async_mode, [gpu_alloc_task=gpu_alloc_task, gpu_alloc2, 
  task_reserve_Hschur=task_reserve_Hschur, explicit_schur,
  _HplinvHll=_HplinvHll, _Hschur=_Hschur, _Hpl=_Hpl, _Hllinv=_Hllinv, engine=engine, 
  _xl=_xl, _bl=_bl, _bschur=_bschur, &bschur_seq=bschur_seq]() {
    
    auto f3 = get_monotonic_time();
    gpu_alloc_task.get();
    gpu_alloc2.get();
    task_reserve_Hschur.get();
    auto f3a = get_monotonic_time();


    auto f3b = get_monotonic_time();
    if (explicit_schur) {
      bschur_seq = engine->create_op_sequence();
      _HplinvHll->subgroup_transposed_multiply_add3(bschur_seq, _Hschur, _Hpl, false, true);
    }
    else {
      // record bschur seq
      bschur_seq = engine->create_op_sequence();
      // sync bschur
      bschur_seq->sync_device<double>({_bschur, _bl});
      bschur_seq->insert_tc_barrier();
      // bschur_seq->insert_cc_barrier();
      

      // borrow xl memory
      bschur_seq->fill_vec(_xl, 0.0);
      // bschur = -bp at this point
      bschur_seq->insert_cc_barrier();

      _Hllinv->subgroup_block_diagonal_multiply_vec_add(bschur_seq, _xl, _bl);
      bschur_seq->insert_cc_barrier();

      _Hpl->subgroup_multiply_vec_add(bschur_seq, _bschur, _xl, false);
      bschur_seq->insert_cc_barrier();

      bschur_seq->fill_vec(_xl, 0.0); // reset
      bschur_seq->insert_cc_barrier();
    }
    auto f3c = get_monotonic_time();
  });

  #ifdef SERIALIZE_PIPELINE
  task_rec_Hschur.wait();
  #endif

  // start of code for mapping allocated blocks
  mapMemory();


  task_init_Hschur = std::async(async_mode, [this, alloc_task1 = gpu_alloc_task,
  alloc_task2 = task_reserve_Hschur]() {
    alloc_task1.get();
    alloc_task2.get();
    if (_Hschur && !Schur_seq) {
      Schur_seq = engine->create_op_sequence();
      Schur_seq->fill_vec(_Hschur->get_buffer(), 0.0);
      Schur_seq->insert_cc_barrier();

      // _Hschur = _Hpp, but keeping the pattern of _Hschur
      _Hpp->copy_blocks_into(Schur_seq, _Hschur);
      Schur_seq->sync_device<double>({_bschur, _bl});

      Schur_seq2 = engine->create_op_sequence();
      if (!_linearSolver->result_gpu()) {
        Schur_seq2->sync_local<double>({_Hschur->get_buffer(), _bschur});
      }
      else {
        Schur_seq2->sync_local<double>({_Hschur->get_buffer()}); // Invert diagonal for PCG on CPU
      }
    }
    else if (!_Hschur && !Schur_seq2) {
      Schur_seq2 = engine->create_op_sequence();
      Schur_seq2->sync_local<double>({_Hpp->get_buffer()}); // Must readback Hpp diagonal on CPU for implicit schur
    }

  });

  #ifdef SERIALIZE_PIPELINE
  task_init_Hschur.wait();
  #endif

  auto tm0 = std::chrono::high_resolution_clock::now();
  // end of mapping code


  auto tmult = std::chrono::high_resolution_clock::now();


  // std::cout << "Built structure!\n";
  return true;
}

bool BlockSolver2::updateStructure(const std::vector<HyperGraph::Vertex*>& vset, const HyperGraph::EdgeSet& edges)
{
  throw std::runtime_error("BlockSolver2: updateStructure is not implemented!");
  return false;
}

bool BlockSolver2::solve(){

  auto stats = _ba_stats;
  double t=get_monotonic_time();
  double t0 = t;

  if (! _doSchur){

    // Hpp and bp/bschur are already synced
    task_solver_setup.get();
    // Solve
    if (Schur_seq) {
      Schur_seq->execute();
    }
    bool ok = _linearSolver->solve(_Hpp, _xp, _bschur);
    // Transfer back
    if (_linearSolver->result_gpu()) {
      sync_x->execute();
    }
    memcpy(_x, _xp->map(), _sizePoses*sizeof(double));

    // Log old and new stats
    G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
    if (globalStats) {
      globalStats->timeLinearSolver = get_monotonic_time() - t;
      globalStats->hessianDimension = globalStats->hessianPoseDimension =
          _Hpp->num_scalar_cols();
    }

    if (stats) {
      auto & bstats = stats->solver_stats.back();
      bstats.time_total = get_monotonic_time()-t0;
      bstats.time = t0 - g2o::gpu::BAStats::getInitTime();

      // record dimensions
      bstats.hpp_rows = _Hpp->num_scalar_rows();
      bstats.hpp_cols = _Hpp->num_scalar_cols();
      bstats.hpp_nnz = _Hpp->num_non_zeros();

    }

    return ok;
  }


  auto init_b = [&](){
    // copy _bp into _bschur
    memcpy(_bschur->map(), _b, _sizePoses * sizeof(double));

    // copy the landmark part of _b into _bl
    memcpy(_bl->map(), _b + _sizePoses, _sizeLandmarks * sizeof(double));
  };
 
  if (!implicit_schur()) {
    // launch calculations
    
    // compute Hll^-1
    task_rec_multiply.get();
    inversion_op->execute();


    // HplinvHll += Hpl*(Hll^-1) (wait)
    mult_HplinvHll->execute();

    // Hschur -= (Hpl*(Hll^-1))*HplT (async)
    task_init_Hschur.get();
    init_b();


    if (Schur_seq) {
      Schur_seq->execute();
    }

    task_rec_Hschur.get();
    bschur_seq->execute();

    // bschur = -bp + (Hpl*(Hll^-1))*bl (async)
    // note that signs are flipped (bp - ....)
    mult_bschur->execute();
  }
  else {
    task_init_Hschur.get();
    init_b();
    task_rec_multiply.get();
    inversion_op->execute();

    task_rec_Hschur.get();
    bschur_seq->execute();
  } 


  #ifdef PRINT_TIMINGS
  std::cout << std::setprecision(3) << "schur_time: " << 1000.0*(get_monotonic_time() - t) << " ms\n";
  #endif

  if (Schur_seq2) { // assume task_init_Hschur completed
    Schur_seq2->execute();
  }

  if (stats) {
    stats->solver_stats.back().time_schur_complement = get_monotonic_time() - t;
  }

  G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
  if (globalStats){
    globalStats->timeSchurComplement = get_monotonic_time() - t;
  }

  t=get_monotonic_time();

  task_solver_setup.get();
  if (!implicit_schur()) {
    if (!_linearSolver->solve(_Hschur, _xp, _bschur)) {
      return false;
    };
  }
  else {
    auto solver = dynamic_cast<compute::ImplicitSchurSolver<double>*>(_linearSolver);
    if (!solver) {
      throw std::runtime_error("Failed to cast to implicit solver!");
    }
    if (!solver->solve(_Hpp, _Hpl, _Hllinv, _xp, _bschur)) {
      return false;
    };
  }

  #ifdef PRINT_TIMINGS
  std::cout << std::setprecision(3) << "linsolver_time: " << 1000.0*(get_monotonic_time() - t) << " ms\n";
  #endif

  if (stats) {
    stats->solver_stats.back().time_linear_solver = get_monotonic_time() - t;
  }

  if (globalStats) {
    globalStats->timeLinearSolver = get_monotonic_time() - t;
    globalStats->hessianPoseDimension = _Hpp->num_scalar_cols();
    globalStats->hessianLandmarkDimension = _Hll->num_scalar_cols();
    globalStats->hessianDimension = globalStats->hessianPoseDimension + globalStats->hessianLandmarkDimension;
  }

  t=get_monotonic_time();


  // bl - HplT*xp

  // copy xp into GPU
  if (!_linearSolver->result_gpu()) {
    memcpy(_x, _xp->map(), _sizePoses*sizeof(double)); // copy from local xp to output _x
    sync_x->execute(); // also sync local xp to device
  }

  auto tl0 = get_monotonic_time();
  task_rec_landmark.get();
  mult_HplTxp->execute();

  // destination xl must be cleared before adding Hll^-1*c into it
  mult_Hllinv->execute(); // also syncs x back to host

  // copy solved parameters into _x
  if (_linearSolver->result_gpu()) {
    memcpy(_x, _xp->map(), _sizePoses*sizeof(double)); // copy from synced xp to output _x
  }
  memcpy(_x + _sizePoses, _xl->map(), _sizeLandmarks*sizeof(double));

  auto t2 = get_monotonic_time();
  #ifdef PRINT_TIMINGS
  std::cout << std::setprecision(3) << "landmark_time: " << 1000.0*(t2 - t) << " ms\n";
  #endif

  if (stats) {
    auto & bstats = stats->solver_stats.back();
    bstats.time_landmark_delta = t2-t;
    bstats.time_total = t2-t0;
    bstats.time = t0 - g2o::gpu::BAStats::getInitTime();

    // record dimensions
    bstats.hpp_rows = _Hpp->num_scalar_rows();
    bstats.hpp_cols = _Hpp->num_scalar_cols();
    bstats.hpp_nnz = _Hpp->num_non_zeros();

    bstats.hpl_rows = _Hpl->num_scalar_rows();
    bstats.hpl_cols = _Hpl->num_scalar_cols();
    bstats.hpl_nnz = _Hpl->num_non_zeros();

    bstats.hll_rows = _Hll->num_scalar_rows();
    bstats.hll_cols = _Hll->num_scalar_cols();
    bstats.hll_nnz = _Hll->num_non_zeros();

    if (_Hschur) {
      bstats.hschur_rows = _Hschur->num_scalar_rows();
      bstats.hschur_cols = _Hschur->num_scalar_cols();
      bstats.hschur_nnz = _Hschur->num_non_zeros();
    }
  }
  return true;
}


bool BlockSolver2::computeMarginals(SparseBlockMatrix<MatrixXd>& spinv, const std::vector<std::pair<int, int> >& blockIndices)
{
  throw std::runtime_error("BlockSolver2: computeMarginals not implemented!");
  return false;
}

bool BlockSolver2::buildSystem()
{
  auto ta = get_monotonic_time();
  gpu_alloc_task.get();
  // clear b vector
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_optimizer->indexMapping().size() > 1000)
# endif
  for (int i = 0; i < static_cast<int>(_optimizer->indexMapping().size()); ++i) {
    OptimizableGraph::Vertex* v=_optimizer->indexMapping()[i];
    assert(v);
    v->clearQuadraticForm();
  }

  // zero memory
  _Hpp->zero_memory();
  if (_doSchur) {
    _Hll->zero_memory();
    _Hpl->zero_memory();
  }
  auto tb = get_monotonic_time();

  // resetting the terms for the pairwise constraints
  // built up the current system by storing the Hessian blocks in the edges and vertices
# ifndef G2O_OPENMP
  // no threading, we do not need to copy the workspace
  JacobianWorkspace& jacobianWorkspace = _optimizer->jacobianWorkspace();
# else
  // if running with threads need to produce copies of the workspace for each thread
  JacobianWorkspace jacobianWorkspace = _optimizer->jacobianWorkspace();
# pragma omp parallel for default (shared) firstprivate(jacobianWorkspace) if (_optimizer->activeEdges().size() > 100)
# endif
  for (int k = 0; k < static_cast<int>(_optimizer->activeEdges().size()); ++k) {
    OptimizableGraph::Edge* e = _optimizer->activeEdges()[k];
    e->linearizeOplus(jacobianWorkspace); // jacobian of the nodes' oplus (manifold)
    e->constructQuadraticForm();
#  ifndef NDEBUG
    for (size_t i = 0; i < e->vertices().size(); ++i) {
      const OptimizableGraph::Vertex* v = static_cast<const OptimizableGraph::Vertex*>(e->vertex(i));
      if (! v->fixed()) {
        bool hasANan = arrayHasNaN(jacobianWorkspace.workspaceForVertex(i), e->dimension() * v->dimension());
        if (hasANan) {
          cerr << "buildSystem(): NaN within Jacobian for edge " << e << " for vertex " << i << endl;
          break;
        }
      }
    }
#  endif
  }

auto tc = get_monotonic_time();

  // flush the current system in a sparse block matrix
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_optimizer->indexMapping().size() > 1000)
# endif
  for (int i = 0; i < static_cast<int>(_optimizer->indexMapping().size()); ++i) {
    OptimizableGraph::Vertex* v=_optimizer->indexMapping()[i];
    int iBase = v->colInHessian();
    if (v->marginalized())
      iBase+=_sizePoses;
    v->copyB(_b+iBase);
  }

  if (!_doSchur) { // TODO: Make other path do that transfer here as well
    memcpy(_bschur->map(), _b, _sizePoses * sizeof(double));
  }

  sync_H->execute();
  auto td = get_monotonic_time();
  return 0;
}


bool BlockSolver2::setLambda(double lambda, bool backup)
{
  // TODO: Allow disabling backup (always backed up by default)
  _lambda->map()[0] = lambda;
  set_lambda_seq->execute();
  return true;
}

void BlockSolver2::restoreDiagonal()
{
  gpu_alloc_task.get();
  restore_diagonal_seq->execute();
}

bool BlockSolver2::init(SparseOptimizer* optimizer, bool online)
{
  double t0 = 0.0;
  if (_ba_stats) {
    t0 = get_monotonic_time();
  }

  _optimizer = optimizer;
  if (! online) {
    if (_Hpp) {
      gpu_alloc_task.get();
      _Hpp->zero_memory();
    }
    if (_Hpl) {
      gpu_alloc_task.get();
      _Hpl->zero_memory();
    }
    if (_Hll) {
      gpu_alloc_task.get();
      _Hll->zero_memory();
    }
  }

  if (_ba_stats) {
    _ba_stats->time_solver_init = get_monotonic_time() - t0;
  }

  return true;
}

void BlockSolver2::setWriteDebug(bool writeDebug)
{
  // throw std::runtime_error("BlockSolver2: setWriteDebug not implemented!");
}

bool BlockSolver2::saveHessian(const std::string& fileName) const
{
  throw std::runtime_error("BlockSolver2: saveHessian not implemented!");
}

void BlockSolver2::mapMemory() {
  gpu_alloc_task.get();
  // now map memory of diagonals
  int poseIdx = 0;
  int landmarkIdx = 0;
  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    if (! v->marginalized()){
      auto ptr = _Hpp->get_block_ptr(poseIdx, poseIdx);
      v->mapHessianMemory(ptr);
      ++poseIdx;
    } else {
      auto ptr = _Hll->get_block_ptr(landmarkIdx, landmarkIdx);
      v->mapHessianMemory(ptr);
      ++landmarkIdx;
    }
  }
  assert(poseIdx == _numPoses && landmarkIdx == _numLandmarks);

  // now map the Hpl blocks
  for (SparseOptimizer::EdgeContainer::const_iterator it=_optimizer->activeEdges().begin(); it!=_optimizer->activeEdges().end(); ++it){
    OptimizableGraph::Edge* e = *it;

    for (size_t viIdx = 0; viIdx < e->vertices().size(); ++viIdx) {
      OptimizableGraph::Vertex* v1 = (OptimizableGraph::Vertex*) e->vertex(viIdx);
      int ind1 = v1->hessianIndex();
      if (ind1 == -1)
        continue;
      int indexV1Bak = ind1;
      for (size_t vjIdx = viIdx + 1; vjIdx < e->vertices().size(); ++vjIdx) {
        OptimizableGraph::Vertex* v2 = (OptimizableGraph::Vertex*) e->vertex(vjIdx);
        int ind2 = v2->hessianIndex();
        if (ind2 == -1)
          continue;
        ind1 = indexV1Bak;
        bool transposedBlock = ind1 > ind2;
        if (transposedBlock){ // make sure, we allocate the upper triangle block
          swap(ind1, ind2);
        }
        if (! v1->marginalized() && !v2->marginalized()){
          auto ptr = _Hpp->get_block_ptr(ind1, ind2);
          e->mapHessianMemory(ptr, viIdx, vjIdx, transposedBlock);
        } else if (v1->marginalized() && v2->marginalized()){
          auto ptr = _Hll->get_block_ptr(ind1-_numPoses, ind2-_numPoses);
          e->mapHessianMemory(ptr, viIdx, vjIdx, false);
        } else { 
          if (v1->marginalized()){ 
            auto ptr = _Hpl->get_block_ptr(v2->hessianIndex(),v1->hessianIndex()-_numPoses);
            e->mapHessianMemory(ptr, viIdx, vjIdx, true); // transpose the block before writing to it
          } else {
            auto ptr = _Hpl->get_block_ptr(v1->hessianIndex(),v2->hessianIndex()-_numPoses);
            e->mapHessianMemory(ptr, viIdx, vjIdx, false); // directly the block
          }
        }
      }
    }
  }
}

}