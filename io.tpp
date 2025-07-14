#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include "io.hpp"

int get_neighbor_rank_periodic(MPI_Comm cart_comm, int dx, int dy) {
    int coords[2], new_coords[2], dims[2];
    int periods[2]={1,1};
    int my_rank, size;

    // Get rank and cartesian info
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_coords(cart_comm, my_rank, 2, coords);

    // Shift coordinates with wraparound if periodic
    new_coords[0] = coords[0] + dx;
    new_coords[1] = coords[1] + dy;
    // std::cout << "rank " << my_rank << ", original coords: " << coords[0] << ", " << coords[1] << ", shifted coords: " << new_coords[0] << ", " << new_coords[1] << "\n";

    if (periods[0]) {
        new_coords[0] = (new_coords[0] + dims[0]) % dims[0];
    } else if (new_coords[0] < 0 || new_coords[0] >= dims[0]) {
        return MPI_PROC_NULL;
    }

    if (periods[1]) {
        new_coords[1] = (new_coords[1] + dims[1]) % dims[1];
    } else if (new_coords[1] < 0 || new_coords[1] >= dims[1]) {
        return MPI_PROC_NULL;
    }

    int neighbor_rank;
    MPI_Cart_rank(cart_comm, new_coords, &neighbor_rank);
    return neighbor_rank;
}

template <typename Real>
void initialize_mpi_bound(parallel_data <Real>&parallel, int nDim, int rank, 
                        int np_size, MPI_Comm cart_comm, 
                        int compression, Real tol_u, Real tol_v, Real snorm)
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

    parallel.left_up    = get_neighbor_rank_periodic(cart_comm, -1, -1);
    parallel.left_down  = get_neighbor_rank_periodic(cart_comm, +1, -1);
    parallel.right_up   = get_neighbor_rank_periodic(cart_comm, -1, +1);
    parallel.right_down = get_neighbor_rank_periodic(cart_comm, +1, +1);

    if (nDim==3) {
        MPI_Cart_shift(cart_comm, 2, 1, &parallel.front, &parallel.back);
    }
}

void initialize_mpi_dataField(field &fieldData, size_t Dx, size_t Dy, size_t Dz, 
                              size_t ghostZ_L, int *dims, int *coords)
{
    fieldData.nx_full    = Dx;
    fieldData.ny_full    = Dy;
    fieldData.nz_full    = Dz; 
    fieldData.ghostZ_len = ghostZ_L;

    fieldData.nx = (Dx>1) ? Dx / dims[0] : 1;
    fieldData.ny = (Dy>1) ? Dy / dims[1] : 1;
    fieldData.nz = (Dz>1) ? Dz / dims[2] : 1;

    fieldData.nx_start = (Dx>1) ? fieldData.nx * coords[0] : 0; 
    fieldData.ny_start = (Dy>1) ? fieldData.ny * coords[1] : 0; 
    fieldData.nz_start = (Dz>1) ? fieldData.nz * coords[2] : 0;
}


