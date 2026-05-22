/// @file pcg.hpp
#pragma once
#include <graphite/ops/vector.hpp>
#include <graphite/preconditioner/preconditioner.hpp>
#include <graphite/solver/solver.hpp>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

namespace graphite {

/// @brief Preconditioned Conjugate Gradient (PCG) solver
template <typename T, typename S> class PCGSolver : public Solver<T, S> {
private:
  thrust::device_vector<T> v;

  // Need vectors for residuals of each factor
  thrust::device_vector<T> v1;   // v1 = Jv (dimension same as r)
  thrust::device_vector<T> v2;   // v2 = J^T v1 (dimension same as x)
  thrust::device_vector<T> r;    // residual
  thrust::device_vector<T> p;    // search direction
  thrust::device_vector<T> z;    // preconditioned residual
  thrust::device_vector<T> diag; // diagonal of Hessian
  thrust::device_vector<T> x_backup;
  thrust::device_vector<T> y;

  size_t max_iter;
  T tol;
  T rejection_ratio;
  T damping_factor;

  Preconditioner<T, S> *preconditioner;

public:
  PCGSolver(size_t max_iter, T tol, T rejection_ratio,
            Preconditioner<T, S> *preconditioner)
      : max_iter(max_iter), tol(tol), rejection_ratio(rejection_ratio),
        damping_factor(0), preconditioner(preconditioner) {}

  virtual void update_structure(Graph<T, S> *graph,
                                StreamPool &streams) override {

    preconditioner->update_structure(graph, streams);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    preconditioner->update_values(graph, streams);
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  StreamPool &streams) override {
    this->damping_factor = damping_factor;
    preconditioner->set_damping_factor(graph, damping_factor, streams);
  }

  // Assumes that x is already initialized
  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {

    auto &vertex_descriptors = graph->get_vertex_descriptors();
    auto &factor_descriptors = graph->get_factor_descriptors();
    T *b = graph->get_b().data().get();
    size_t dim_h = graph->get_hessian_dimension();

    size_t dim_r = 0;
    for (size_t i = 0; i < factor_descriptors.size(); i++) {
      dim_r += factor_descriptors[i]->get_residual_size();
    }

    // Resize vectors (assuming this causes host synchronization)
    v1.resize(dim_r);
    v2.resize(dim_h);
    r.resize(dim_h); // dim h because dim(r) = dim(Ax) = dim(b)
    diag.resize(dim_h);

    const cudaStream_t stream = 0;

    thrust::fill(thrust::cuda::par_nosync.on(stream), x, x + dim_h, 0.0);
    thrust::fill(thrust::cuda::par_nosync.on(stream), v1.begin(), v1.end(),
                 0.0);
    thrust::fill(thrust::cuda::par_nosync.on(stream), v2.begin(), v2.end(),
                 0.0);

    // Compute residual
    thrust::copy(thrust::cuda::par_nosync.on(stream), graph->get_b().begin(),
                 graph->get_b().end(), r.begin());

    // 3. Add damping factor
    // v2 += damping_factor*diag(H)*x
    thrust::fill(thrust::cuda::par_nosync.on(stream), diag.begin(), diag.end(),
                 0.0);
    for (size_t i = 0; i < factor_descriptors.size(); i++) {
      factor_descriptors[i]->compute_hessian_scalar_diagonal_async(
          diag.data().get(),
          graph->get_jacobian_scales().data().get()); // also on default stream
    }

    // Check for negative values in diag and print an error if found
    T min_diag = static_cast<T>(1.0e-6);
    T max_diag = static_cast<T>(1.0e32);
    ops::clamp_async(stream, dim_h, min_diag, max_diag, diag.data().get());

    cudaStreamSynchronize(stream);

    // Rescale r
    y.resize(dim_h);
    auto rnorm = thrust::inner_product(thrust::device, r.begin(), r.end(),
                                       r.begin(), static_cast<T>(0.0));
    rnorm = std::sqrt(rnorm);
    auto scale = 1.0 / rnorm;
    ops::rescale_vec_async<T>(stream, dim_h, y.data().get(), scale,
                              r.data().get());
    cudaStreamSynchronize(stream);
    // Apply preconditioner
    z.resize(dim_h);

    thrust::fill(z.begin(), z.end(), 0.0);
    preconditioner->apply(graph, z.data().get(), y.data().get(), streams);

    p.resize(dim_h);
    thrust::copy(z.begin(), z.end(), p.begin()); // p = z

    x_backup.resize(dim_h);

    // 1. First compute dot(r, z)
    T rz = (T)thrust::inner_product(r.begin(), r.end(), z.begin(),
                                    static_cast<T>(0.0));

    T rz_0 = std::numeric_limits<T>::infinity();

    for (size_t k = 0; k < max_iter; k++) {

      if (rz == 0) {
        // std::cout << "rz is zero, stopping at iteration " << k << std::endl;
        break;
      }

      // auto t_jv_start = std::chrono::steady_clock::now();
      // 2. Compute v1 = Jp
      thrust::fill(v1.begin(), v1.end(), 0.0);
      auto v1_ptr = v1.data().get(); // reset
      for (size_t i = 0; i < factor_descriptors.size(); i++) {
        factor_descriptors[i]->compute_Jv(
            v1_ptr, p.data().get(), graph->get_jacobian_scales().data().get(),
            streams);
        v1_ptr += factor_descriptors[i]->get_residual_size();
      }
      // auto t_jv_end = std::chrono::steady_clock::now();
      // std::cout << "Time for Jv: "
      //           << std::chrono::duration<double>(t_jv_end -
      //           t_jv_start).count()
      //           << " seconds" << std::endl;

      // 3. Compute v2 = J^T v1
      thrust::fill(v2.begin(), v2.end(), 0.0);
      v1_ptr = v1.data().get(); // reset
      for (size_t i = 0; i < factor_descriptors.size(); i++) {
        factor_descriptors[i]->compute_Jtv(
            v2.data().get(), v1_ptr, graph->get_jacobian_scales().data().get(),
            streams);
        v1_ptr += factor_descriptors[i]->get_residual_size();
      }
      // Add damping factor
      // v2 += damping_factor*diag(H)*p
      ops::damp_by_factor_async(stream, dim_h, v2.data().get(), damping_factor,
                                diag.data().get(), p.data().get());

      // 4. Compute alpha = dot(r, z) / dot(p, v2)
      T alpha = (rz) / thrust::inner_product(thrust::cuda::par.on(stream),
                                             p.begin(), p.end(), v2.begin(),
                                             static_cast<T>(0.0));
      // 5. x  += alpha * p
      thrust::copy(thrust::cuda::par.on(stream), x, x + dim_h,
                   x_backup.begin());
      ops::axpy_async(stream, dim_h, x, alpha, p.data().get(), x);

      // 6. r -= alpha * v2
      ops::axpy_async(stream, dim_h, r.data().get(), -alpha, v2.data().get(),
                      r.data().get());
      // cudaStreamSynchronize(0);

      rnorm = (T)thrust::inner_product(thrust::cuda::par.on(stream), r.begin(),
                                       r.end(), r.begin(), static_cast<T>(0.0));
      rnorm = std::sqrt(rnorm);
      scale = 1.0 / rnorm;
      ops::rescale_vec_async<T>(stream, dim_h, y.data().get(), scale,
                                r.data().get());

      // Apply preconditioner again
      thrust::fill(thrust::cuda::par.on(stream), z.begin(), z.end(), 0.0);
      preconditioner->apply(graph, z.data().get(), y.data().get(), streams);
      T rz_new = thrust::inner_product(thrust::cuda::par.on(stream), r.begin(),
                                       r.end(), z.begin(), static_cast<T>(0.0));

      // if (rz_new > rejection_ratio * rz_0) {
      if (std::abs(rz_new) > rejection_ratio * rz_0 || std::isnan(rz_new)) {
        thrust::copy(thrust::device, x_backup.begin(), x_backup.end(), x);
        // std::cout << "Rejection: rz_new = " << rz_new
        //           << ", rz_0 = " << rz_0 << " at iteration " << k + 1 <<
        //           std::endl;
        break;
      }
      rz_0 = std::min(rz_0, std::abs(rz_new));

      // 8. Compute beta
      // std::cout << "rz_new: " << rz_new << ", rz: " << rz
      //           << ", at iteration " << k + 1 << std::endl;

      T beta = rz_new / rz;
      rz = rz_new;

      // 9. Update p
      ops::axpy_async(stream, dim_h, p.data().get(), beta, p.data().get(),
                      z.data().get());
      cudaStreamSynchronize(stream);

      if (std::abs(static_cast<T>(rz_new)) < tol) {
        // std::cout << "Converged after " << k + 1
        //           << " iterations with residual: " << rz_new << std::endl;
        break;
      }
      // if (k == max_iter - 1) {
      //   std::cout << "Reached maximum iterations: " << max_iter
      //             << " with residual: " << rz_new << std::endl;
      // }
    }
    // TODO: Figure out failure cases
    return true;
  }
};
} // namespace graphite