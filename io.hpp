#ifndef IO_HPP
#define IO_HPP

#define BOUNDARY_UP    1
#define BOUNDARY_DOWN  2
#define BOUNDARY_LEFT  3
#define BOUNDARY_RIGHT 4
#define BOUNDARY_FRONT 5
#define BOUNDARY_BACK  6

/* Datatype for temperature field */
typedef struct {
    /* nx and ny are the true dimensions of the field. The array data
     * contains also ghost layers, so it will have dimensions nx+2 x ny+2 */
    size_t nx;                     /* Local dimensions of the field */
    size_t ny;
    size_t nz;
    size_t nx_start;               /* Global index at the start of the field */
    size_t ny_start;
    size_t nz_start;
    size_t nx_full;                /* Global dimensions of the field */
    size_t ny_full;                /* Global dimensions of the field */
    size_t nz_full;
    size_t ghostZ_len;             /* Width of ghost zone used for MPI communication */
} field;

/* Datatype for basic parallelization information */
template <typename Real>
struct parallel_data {
    int size;                   /* Number of MPI tasks */
    int rank;
    size_t ncol;
    size_t nrow;
    int up, down, left, right; /* Ranks of neighbouring MPI tasks */
    int left_up, left_down, right_up, right_down; /* Ranks of neighbouring in the diagnol directions */
    int front, back;           /* save these entries for 3D */ 
    // boundary
    int boundary_rank;         /* 0 not at boundary; 1 up; 2 down; 3 right; 4 left; 5 front; 6 back*/ 
    MPI_Comm comm;             /* Cartesian communicator */
    MPI_Datatype datatype;     /* MPI Datatype for file view in restart I/O */
    int compression;           /* 0: non MPI compression; 1: MGARD compress for MPI; 2: SZ compression */
    Real tol_u;                /* Absolute error bound defined under snorm */
    Real tol_v;                
    Real snorm;                /* MGARD compression snorm */
} ;

// initialize the neighboring ranks
// mpi_double: 1 double | 0 float
template <typename Real>
void initialize_mpi_bound(parallel_data<Real> &parallelization, int nDim,
                        int rank, int np_size, MPI_Comm comm,
                        int compression, Real tol_u, Real tol_v, Real snorm);

void initialize_mpi_dataField(field &fieldData, size_t Dx, size_t Dy, size_t Dz,
                              size_t ghostZ_len, int *dims, int *coords);

#include "io.tpp"
#endif 
