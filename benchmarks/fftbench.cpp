/* Single-rank FFT microbenchmark: compares the heffte pipeline used by
 * petapm against a fused fftw 3D plan (the best case, and roughly what
 * pfft did when one rank owned the whole mesh).
 *
 * heffte always runs three separate batched 1-D FFT stages with pack /
 * reshape copies between them, so on one rank it is memory-bandwidth
 * bound and slower than the fused plan; the gap closes when the mesh is
 * distributed and every library must pack and exchange anyway. This
 * benchmark exists to check that heffte tuning changes (threaded
 * packers, plan_options) behave as expected: run it before and after.
 *
 * Usage: ./fftbench [Nmesh] [iterations]
 * Times are seconds per forward+backward pair.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <omp.h>
#include <fftw3.h>
#include <mpi.h>
#include <heffte.h>

/* Fused, in-place, threaded fftw 3D r2c/c2r plan. */
static double bench_fftw(int N, int iters)
{
    ptrdiff_t nreal = (ptrdiff_t) N * N * (N/2+1) * 2;
    double * real = fftw_alloc_real(nreal);
    fftw_complex * cplx = (fftw_complex *) real;
    fftw_plan fwd = fftw_plan_dft_r2c_3d(N, N, N, real, cplx, FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_c2r_3d(N, N, N, cplx, real, FFTW_ESTIMATE);
    ptrdiff_t i;
    for(i = 0; i < nreal; i++) real[i] = i % 17;
    fftw_execute(fwd); fftw_execute(bwd); /* warmup */
    double t0 = omp_get_wtime();
    for(int it = 0; it < iters; it++) {
        fftw_execute(fwd);
        fftw_execute(bwd);
    }
    double dt = (omp_get_wtime() - t0) / iters;
    fftw_destroy_plan(fwd); fftw_destroy_plan(bwd); fftw_free(real);
    return dt;
}

/* heffte with the same boxes petapm_init builds: whole mesh on one rank,
 * input z fastest, transposed output with x fastest (order {0,2,1}). */
static double bench_heffte(int N, int iters, heffte::plan_options opts)
{
    heffte::box3d<> world({0,0,0},{N-1,N-1,N-1});
    heffte::box3d<> rbox = heffte::split_world(world, {1,1,1})[0];
    heffte::box3d<> fbox = heffte::split_world(world.r2c(2), {1,1,1})[0];
    heffte::box3d<> inbox(rbox.low, rbox.high, {2,1,0});
    heffte::box3d<> outbox(fbox.low, fbox.high, {0,2,1});
    heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox, outbox, 2, MPI_COMM_SELF, opts);
    std::vector<double> real(fft.size_inbox());
    std::vector<std::complex<double>> cplx(fft.size_outbox());
    std::vector<std::complex<double>> work(fft.size_workspace());
    size_t i;
    for(i = 0; i < real.size(); i++) real[i] = i % 17;
    fft.forward(real.data(), cplx.data(), work.data());
    fft.backward(cplx.data(), real.data(), work.data());
    double t0 = omp_get_wtime();
    for(int it = 0; it < iters; it++) {
        fft.forward(real.data(), cplx.data(), work.data());
        fft.backward(cplx.data(), real.data(), work.data());
    }
    return (omp_get_wtime() - t0) / iters;
}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    int N = argc > 1 ? atoi(argv[1]) : 256;
    int iters = argc > 2 ? atoi(argv[2]) : 5;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    printf("N=%d threads=%d\n", N, omp_get_max_threads());
    printf("fftw fused 3d plan:       %.4f s\n", bench_fftw(N, iters));
    heffte::plan_options def = heffte::default_options<heffte::backend::fftw>();
    printf("heffte default:           %.4f s\n", bench_heffte(N, iters, def));
    heffte::plan_options noreord = def;
    noreord.use_reorder = false;
    printf("heffte use_reorder=false: %.4f s (petapm setting)\n", bench_heffte(N, iters, noreord));
    heffte::plan_options slabs = def;
    slabs.use_pencils = false;
    printf("heffte use_pencils=false: %.4f s\n", bench_heffte(N, iters, slabs));
    heffte::plan_options both = slabs;
    both.use_reorder = false;
    printf("heffte slabs, no reorder: %.4f s\n", bench_heffte(N, iters, both));
    MPI_Finalize();
    return 0;
}
