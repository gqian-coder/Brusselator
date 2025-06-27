#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include "io.hpp"

template <typename Real>
void initialize_mpi_bound(parallel_data <Real>&parallel, int nDim, int rank, 
                        int np_size, MPI_Comm cart_comm, 
                        bool compression, Real tol_u, Real tol_v, Real snorm)
{
    parallel.comm        = cart_comm;
    parallel.rank        = rank;
    parallel.size        = np_size;
    parallel.compression = compression;
    parallel.tol_u       = tol_u;
    parallel.tol_v       = tol_v;
    parallel.snorm       = snorm;
    
    if (sizeof(Real)==sizeof(double)) parallel.datatype = MPI_DOUBLE;
    else if (sizeof(Real)==sizeof(float)) parallel.datatype = MPI_FLOAT; 
    else {
        std::cout << "This simulatino only support data type double | float\n";
        return;
    }
    MPI_Cart_shift(cart_comm, 0, 1, &parallel.up, &parallel.down);
    MPI_Cart_shift(cart_comm, 1, 1, &parallel.left, &parallel.right);    
    if (nDim==3) {
        MPI_Cart_shift(cart_comm, 2, 1, &parallel.front, &parallel.back);
    }
}

void initialize_mpi_dataField(field &fieldData, size_t Dx, size_t Dy, size_t Dz, 
                                    int *dims, int *coords)
{
    fieldData.nx_full = Dx;
    fieldData.ny_full = Dy;
    fieldData.nz_full = Dz; 

    fieldData.nx = (Dx>1) ? Dx / dims[0] : 1;
    fieldData.ny = (Dy>1) ? Dy / dims[1] : 1;
    fieldData.nz = (Dz>1) ? Dz / dims[2] : 1;

    fieldData.nx_start = (Dx>1) ? fieldData.nx * coords[0] : 0; 
    fieldData.ny_start = (Dy>1) ? fieldData.ny * coords[1] : 0; 
    fieldData.nz_start = (Dz>1) ? fieldData.nz * coords[2] : 0;
}


