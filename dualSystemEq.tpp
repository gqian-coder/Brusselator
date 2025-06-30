#include <array>
#include <ostream>
#include <vector>
#include "io.hpp"
#include "mgard/compress_x.hpp"
#include "SZ3/api/sz.hpp"

// Helper to index 2D arrays stored in 1D
inline size_t idx(size_t i, size_t j, size_t ny) { return i * ny + j; }

template<typename Real>
dualSystemEquation<Real>::dualSystemEquation(field fieldData,
                   Real dt, Real dh, Real Du, Real Dv, Real A, Real B)
               : dField(fieldData), dt(dt), dh(dh), Du(Du), Dv(Dv), A(A), B(B) 
{
    // create a ghost zone to host the boundary data for mpi exchange
    data_size = dField.nz>1 ? (dField.nx+2) * (dField.ny+2) * (dField.nz+2) : (dField.nx+2) * (dField.ny+2);
    //std::cout << "data_size = " << data_size << "\n";
    u_n.resize(data_size, A);//+0.1);
    v_n.resize(data_size, B / A);//+0.2);
}


template<typename Real>
void dualSystemEquation<Real>::init_u_2d(Real *data)
{
    for (size_t i=0; i<dField.nx; i++) {
        std::copy(data+i*dField.ny, data+i*dField.ny+dField.ny, u_n.data()+(i+1)*(dField.ny+2) + 1);
    }
}


template<typename Real>
void dualSystemEquation<Real>::init_v_2d(Real *data)
{
    for (size_t i=0; i<dField.nx; i++) {
        std::copy(data+i*dField.ny, data+i*dField.ny+dField.ny, v_n.data()+(i+1)*(dField.ny+2) + 1);
    }
}


// Compute Laplacian using 5-posize_t stencil
template<typename Real>
void dualSystemEquation<Real>::compute_laplacian(const std::vector<Real>& grid, std::vector<Real>& lap, 
                                                size_t nx, size_t ny, Real dh) 
{
    Real dh2 = dh*dh;
    for (size_t i = 1; i <= nx; ++i) {
        for (size_t j = 1; j <= ny; ++j) {
            lap[idx(i, j, ny + 2)] = (
                grid[idx(i + 1, j, ny + 2)] + grid[idx(i - 1, j, ny + 2)] 
                + grid[idx(i, j + 1, ny + 2)] + grid[idx(i, j - 1, ny + 2)] 
                - 4 * grid[idx(i, j, ny + 2)]) / dh2;
        }
    }
}


// Exchange ghost cells with neighbors with compression on the whole sub-domain 
template<typename Real>
size_t dualSystemEquation<Real>::exchange_ghost_cells_mgr(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right, Real tol, Real s)
{
    MPI_Status status;
    size_t total_buffer = (local_ny+2)*(local_nx+2);
    std::vector<Real> recv_buf(total_buffer*4);
    // compressed buffer
    std::vector<unsigned char> recv_buf_compressed(total_buffer*4*sizeof(Real));

    // compression parameters
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::SERIAL;
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    config.normalize_coordinates = true;
    std::vector<mgard_x::SIZE> data_shape{local_nx+2, local_ny+2};
    size_t compressed_size = total_buffer * sizeof(Real);
    size_t recv_size_up, recv_size_down, recv_size_left, recv_size_right;

    // compression the entire sub-domain 
    char *bufferOut = (char *) malloc(compressed_size);
    void *compressed_data = bufferOut; 
    
    mgard_x::compress(2, mgard_x::data_type::Double, data_shape, tol, s,
                mgard_x::error_bound_type::ABS, grid.data(),
                compressed_data, compressed_size, config, true);
    
    MPI_Barrier(cart_comm);
    // Up/down communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, up, 0,
                 &recv_size_down, 1, MPI_UNSIGNED_LONG, down, 0, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, down, 1,
                 &recv_size_up, 1, MPI_UNSIGNED_LONG, up, 1, cart_comm, &status);
    // Left/right communication (columns) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, left, 2,
                 &recv_size_right, 1, MPI_UNSIGNED_LONG, right, 2, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, right, 3,
                 &recv_size_left, 1, MPI_UNSIGNED_LONG, left, 3, cart_comm, &status);

    MPI_Barrier(cart_comm);
    //std::cout << "compressed data = " << compressed_size << ", received size up = " << recv_size_up << ", down = " << recv_size_down <<", left = " << recv_size_left << ", right = " << recv_size_right << "\n";

    // Up/down communication (rows) the compressed data
    unsigned char *recv_compressed_up = recv_buf_compressed.data();
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, down, 5,
                 recv_compressed_up, recv_size_up, MPI_BYTE, up, 5, cart_comm, &status);
    unsigned char *recv_compressed_down = recv_buf_compressed.data() + recv_size_up;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, up, 4,
                 recv_compressed_down, recv_size_down, MPI_BYTE, down, 4, cart_comm, &status);
    // Left/right communication (columns) the compressed data
    unsigned char *recv_compressed_left = recv_buf_compressed.data() + recv_size_up + recv_size_down;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, right, 7,
                 recv_compressed_left, recv_size_left, MPI_BYTE, left, 7, cart_comm, &status);
    unsigned char *recv_compressed_right = recv_buf_compressed.data() +recv_size_up+recv_size_down+recv_size_left;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, left, 6,
                 recv_compressed_right, recv_size_right, MPI_BYTE, right, 6, cart_comm, &status);

    // Decompression
    void *recv_buf_up = static_cast<void*>(recv_buf.data());
    mgard_x::decompress(recv_compressed_up, recv_size_up, recv_buf_up, config, true);

    void *recv_buf_down = static_cast<void*>(recv_buf.data() + total_buffer);
    mgard_x::decompress(recv_compressed_down, recv_size_down, recv_buf_down, config, true);

    void *recv_buf_left = static_cast<void*>(recv_buf.data() + 2*total_buffer);
    mgard_x::decompress(recv_compressed_left, recv_size_left, recv_buf_left, config, true);

    void *recv_buf_right = static_cast<void*>(recv_buf.data() + 3*total_buffer);
    mgard_x::decompress(recv_compressed_right, recv_size_right, recv_buf_right, config, true);
    
    // copy back to the current grid
    // Up
    double *recv_buf_ptr = recv_buf.data() + total_buffer - 2*ny + 1;
    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(0, j + 1, ny)] = *(recv_buf_ptr++);
    }
    // Down
    recv_buf_ptr = recv_buf.data() + total_buffer + ny + 1;
    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(local_nx + 1, j + 1, ny)] = *(recv_buf_ptr++);
    }
    // Left
    recv_buf_ptr = recv_buf.data() + (total_buffer  + ny - 1) * 2;
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, 0, ny)] = *recv_buf_ptr;
        recv_buf_ptr += ny;
    }
    // Right
    recv_buf_ptr = recv_buf.data() + total_buffer * 3 + ny + 1;
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, local_ny + 1, ny)] = *recv_buf_ptr;
        recv_buf_ptr += ny;
    }
    free(bufferOut);
    return compressed_size;
}


// Exchange ghost cells with neighbors with compression on the whole sub-domain
template<typename Real>
size_t dualSystemEquation<Real>::exchange_ghost_cells_SZ(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right, Real tol)
{
    MPI_Status status;
    size_t total_buffer = (local_ny+2)*(local_nx+2);
    std::vector<Real> recv_buf(total_buffer*4);
    // compressed buffer
    std::vector<char> recv_buf_compressed(total_buffer*4*sizeof(Real));

    // compression parameters
    SZ3::Config conf(local_nx+2, local_ny+2); 
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO; 
    conf.errorBoundMode = SZ3::EB_ABS; 
    conf.absErrorBound = tol; // absolute error bound
    size_t compressed_size = total_buffer * sizeof(Real);
    size_t recv_size_up, recv_size_down, recv_size_left, recv_size_right;

    // compression the entire sub-domain
    char *bufferOut = (char *) malloc(compressed_size);
    void *compressed_data = bufferOut;

    compressed_data = SZ_compress(conf, grid.data(), compressed_size);

    MPI_Barrier(cart_comm);
    // Up/down communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, up, 0,
                 &recv_size_down, 1, MPI_UNSIGNED_LONG, down, 0, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, down, 1,
                 &recv_size_up, 1, MPI_UNSIGNED_LONG, up, 1, cart_comm, &status);
    // Left/right communication (columns) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, left, 2,
                 &recv_size_right, 1, MPI_UNSIGNED_LONG, right, 2, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, right, 3,
                 &recv_size_left, 1, MPI_UNSIGNED_LONG, left, 3, cart_comm, &status);

    MPI_Barrier(cart_comm);
    //std::cout << "compressed data = " << compressed_size << ", received size up = " << recv_size_up << ", down = " << recv_size_down <<", left = " << recv_size_left << ", right = " << recv_size_right << "\n";

    // Up/down communication (rows) the compressed data
    char *recv_compressed_up = recv_buf_compressed.data();
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, down, 5,
                 recv_compressed_up, recv_size_up, MPI_BYTE, up, 5, cart_comm, &status);
    char *recv_compressed_down = recv_buf_compressed.data() + recv_size_up;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, up, 4,
                 recv_compressed_down, recv_size_down, MPI_BYTE, down, 4, cart_comm, &status);
    // Left/right communication (columns) the compressed data
    char *recv_compressed_left = recv_buf_compressed.data() + recv_size_up + recv_size_down;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, right, 7,
                 recv_compressed_left, recv_size_left, MPI_BYTE, left, 7, cart_comm, &status);
    char *recv_compressed_right = recv_buf_compressed.data() +recv_size_up+recv_size_down+recv_size_left;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, left, 6,
                 recv_compressed_right, recv_size_right, MPI_BYTE, right, 6, cart_comm, &status);

    // Decompression
    Real *recv_buf_up = recv_buf.data();
    SZ_decompress(conf, recv_compressed_up, recv_size_up, recv_buf_up);

    Real *recv_buf_down = (recv_buf.data() + total_buffer);
    SZ_decompress(conf, recv_compressed_down, recv_size_down, recv_buf_down);

    Real *recv_buf_left = (recv_buf.data() + 2*total_buffer);
    SZ_decompress(conf, recv_compressed_left, recv_size_left, recv_buf_left);

    Real *recv_buf_right = (recv_buf.data() + 3*total_buffer);
    SZ_decompress(conf, recv_compressed_right, recv_size_right, recv_buf_right);

    // copy back to the current grid
    // Up
    double *recv_buf_ptr = recv_buf.data() + total_buffer - 2*ny + 1;
    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(0, j + 1, ny)] = *(recv_buf_ptr++);
    }
    // Down
    recv_buf_ptr = recv_buf.data() + total_buffer + ny + 1;
    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(local_nx + 1, j + 1, ny)] = *(recv_buf_ptr++);
    }
    // Left
    recv_buf_ptr = recv_buf.data() + (total_buffer  + ny - 1) * 2;
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, 0, ny)] = *recv_buf_ptr;
        recv_buf_ptr += ny;
    }
    // Right
    recv_buf_ptr = recv_buf.data() + total_buffer * 3 + ny + 1;
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, local_ny + 1, ny)] = *recv_buf_ptr;
        recv_buf_ptr += ny;
    }
    free(bufferOut);
    return compressed_size;
}

    
/*
// Exchange ghost cells with neighbors with compression on edge nodes only
template<typename Real>
void dualSystemEquation<Real>::exchange_ghost_cells_mgr(std::vector<Real>& grid, 
                          size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right, Real tol, Real s)
{
    MPI_Status status;
    size_t total_buffer = local_ny*2+local_nx*2;
    std::vector<Real> send_buf(total_buffer), recv_buf(total_buffer);
    // compressed buffer
    std::vector<unsigned char>send_buf_compressed(total_buffer*sizeof(Real)), recv_buf_compressed(total_buffer*sizeof(Real));
    Real *send_data_ptr = send_buf.data();

    // compression parameters
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::SERIAL;
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    config.normalize_coordinates = true;
    std::vector<mgard_x::SIZE> shape_nx{local_nx}, shape_ny{local_ny};
    size_t compressed_size_up, compressed_size_down, compressed_size_left, compressed_size_right;
    size_t recv_size_up, recv_size_down, recv_size_left, recv_size_right;

    // Up/down boundary (rows)
    for (size_t j = 0; j < local_ny; ++j) {
        *(send_data_ptr++) = grid[idx(1, j + 1, ny)];
    }
    for (size_t j = 0; j < local_ny; ++j) {
        *(send_data_ptr++) = grid[idx(local_nx, j + 1, ny)];
    }
    // Left/right boundary (columns)
    for (size_t i = 0; i < local_nx; ++i) {
        *(send_data_ptr++) = grid[idx(i + 1, 1, ny)];
    }
    for (size_t i = 0; i < local_nx; ++i) {
        *(send_data_ptr++) = grid[idx(i + 1, local_ny, ny)];
    }
    
    // Up/down compression (rows)
    void *send_compressed_up = static_cast<void*>(send_buf_compressed.data());
    if (sizeof(Real)==sizeof(double)) {
        mgard_x::compress(1, mgard_x::data_type::Double, shape_ny, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data(),
                send_compressed_up, compressed_size_up, config, true);
    } else if (sizeof(Real)==sizeof(float)) {
        mgard_x::compress(1, mgard_x::data_type::Float, shape_ny, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data(),
                send_compressed_up, compressed_size_up, config, true);
    } 
    void *send_compressed_down = static_cast<void*>(send_buf_compressed.data() + compressed_size_up);
    if (sizeof(Real)==sizeof(double)) {
        mgard_x::compress(1, mgard_x::data_type::Double, shape_ny, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data()+local_ny,
                send_compressed_down, compressed_size_down, config, true);
    } else if (sizeof(Real)==sizeof(float)) {
        mgard_x::compress(1, mgard_x::data_type::Float, shape_ny, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data()+local_ny,
                send_compressed_down, compressed_size_down, config, true);
    } 

    // Left/right compression (columns)
    void *send_compressed_left = static_cast<void*>(send_buf_compressed.data() + compressed_size_up + compressed_size_down);
    if (sizeof(Real)==sizeof(double)) {
        mgard_x::compress(1, mgard_x::data_type::Double, shape_nx, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data() + 2*local_ny,
                send_compressed_left, compressed_size_left, config, true);
    } else if (sizeof(Real)==sizeof(float)) {
        mgard_x::compress(1, mgard_x::data_type::Float, shape_nx, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data() + 2*local_ny,
                send_compressed_left, compressed_size_left, config, true);
    }
    void *send_compressed_right = static_cast<void*>(send_buf_compressed.data() + compressed_size_up + compressed_size_down + compressed_size_left);
    if (sizeof(Real)==sizeof(double)) {
        mgard_x::compress(1, mgard_x::data_type::Double, shape_nx, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data()+2*local_ny+local_nx,
                send_compressed_right, compressed_size_right, config, true);
    } else if (sizeof(Real)==sizeof(float)) {
        mgard_x::compress(1, mgard_x::data_type::Float, shape_nx, tol, s,
                mgard_x::error_bound_type::ABS, send_buf.data()+2*local_ny+local_nx,
                send_compressed_right, compressed_size_right, config, true);
    }    
    
    std::cout << "compressed size up = " << compressed_size_up << ", down = " << compressed_size_down << ", left = " << compressed_size_left << ", right = " << compressed_size_right << "\n"; 

    MPI_Barrier(cart_comm);
    // Up/down communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size_up, 1, MPI_UNSIGNED_LONG, up, 0,
                 &recv_size_down, 1, MPI_UNSIGNED_LONG, down, 0, cart_comm, &status); 
    MPI_Sendrecv(&compressed_size_down, 1, MPI_UNSIGNED_LONG, down, 1,
                 &recv_size_up, 1, MPI_UNSIGNED_LONG, up, 1, cart_comm, &status);
    // Left/right communication (columns) the compressed size
    MPI_Sendrecv(&compressed_size_left, 1, MPI_UNSIGNED_LONG, left, 2,
                 &recv_size_right, 1, MPI_UNSIGNED_LONG, right, 2, cart_comm, &status);
    MPI_Sendrecv(&compressed_size_right, 1, MPI_UNSIGNED_LONG, right, 3,
                 &recv_size_left, 1, MPI_UNSIGNED_LONG, left, 3, cart_comm, &status);

    MPI_Barrier(cart_comm);
    std::cout << "received size up = " << recv_size_up << ", down = " << recv_size_down <<", left = " << recv_size_left << ", right = " << recv_size_right << "\n";
    // Up/down communication (rows) the compressed data
    //void *recv_compressed_up = static_cast<void*>(recv_buf_compressed.data());
    unsigned char *recv_compressed_up = recv_buf_compressed.data();
    MPI_Sendrecv(send_compressed_down, compressed_size_down, MPI_BYTE, down, 5,
                 recv_compressed_up, recv_size_up, MPI_BYTE, up, 5, cart_comm, &status);
    //void *recv_compressed_down = static_cast<void*>(recv_buf_compressed.data() + recv_size_up);
    unsigned char *recv_compressed_down = recv_buf_compressed.data() + recv_size_up;
    MPI_Sendrecv(send_compressed_up, compressed_size_up, MPI_BYTE, up, 4,
                 recv_compressed_down, recv_size_down, MPI_BYTE, down, 4, cart_comm, &status);
    // Left/right communication (columns) the compressed data
    //void *recv_compressed_left = static_cast<void*>(recv_buf_compressed.data() + recv_size_up + recv_size_down);
    unsigned char *recv_compressed_left = recv_buf_compressed.data() + recv_size_up + recv_size_down;
    MPI_Sendrecv(send_compressed_right, compressed_size_right, MPI_BYTE, right, 7,
                 recv_compressed_left, recv_size_left, MPI_BYTE, left, 7, cart_comm, &status);
    //void *recv_compressed_right = static_cast<void*>(recv_buf_compressed.data() +recv_size_up+recv_size_down+recv_size_left);
    unsigned char *recv_compressed_right = recv_buf_compressed.data() +recv_size_up+recv_size_down+recv_size_left;
    MPI_Sendrecv(send_compressed_left, compressed_size_left, MPI_BYTE, left, 6,
                 recv_compressed_right, recv_size_right, MPI_BYTE, right, 6, cart_comm, &status);

    // Decompression
    void *recv_buf_up = static_cast<void*>(recv_buf.data());
    mgard_x::decompress((void**)recv_compressed_up, compressed_size_up, recv_buf_up, config, true);
    void *recv_buf_down = static_cast<void*>(recv_buf.data() + local_ny);
    mgard_x::decompress((void *)recv_compressed_down, compressed_size_down, recv_buf_down, config, true);
    void *recv_buf_left = static_cast<void*>(recv_buf.data() + 2*local_ny);
    mgard_x::decompress((void *)recv_compressed_left, compressed_size_left, recv_buf_left, config, true);
    void *recv_buf_right = static_cast<void*>(recv_buf.data() + 2*local_ny + local_nx);
    mgard_x::decompress((void *)recv_compressed_right, compressed_size_right, recv_buf_right, config, true);
     
    double *recv_buf_ptr = recv_buf.data();
    // copy back to the current grid
    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(0, j + 1, ny)] = *(recv_buf_ptr++);
    }
    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(local_nx + 1, j + 1, ny)] = *(recv_buf_ptr++); 
    }
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, 0, ny)] = *(recv_buf_ptr++); 
    }
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, local_ny + 1, ny)] = *(recv_buf_ptr++); 
    }
}
*/


// Exchange ghost cells with neighbors
template<typename Real>
void dualSystemEquation<Real>::exchange_ghost_cells(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Datatype datatype, MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right)
{
    MPI_Status status;
    std::vector<Real> send_buf(local_ny), recv_buf(local_ny);

    // Up/down communication (rows)
    for (size_t j = 0; j < local_ny; ++j) {
        send_buf[j] = grid[idx(1, j + 1, ny)];
    }
    
    MPI_Sendrecv(send_buf.data(), local_ny, datatype, up, 0,
                 recv_buf.data(), local_ny, datatype, down, 0, cart_comm, &status);

    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(local_nx + 1, j + 1, ny)] = recv_buf[j];
    }

    for (size_t j = 0; j < local_ny; ++j) {
        send_buf[j] = grid[idx(local_nx, j + 1, ny)];
    }
    MPI_Sendrecv(send_buf.data(), local_ny, datatype, down, 1,
                 recv_buf.data(), local_ny, datatype, up, 1, cart_comm, &status);
    for (size_t j = 0; j < local_ny; ++j) {
        grid[idx(0, j + 1, ny)] = recv_buf[j];
    }

    // Left/right communication (columns)
    send_buf.resize(local_nx);
    recv_buf.resize(local_nx);
    for (size_t i = 0; i < local_nx; ++i) {
        send_buf[i] = grid[idx(i + 1, 1, ny)];
    }
    MPI_Sendrecv(send_buf.data(), local_nx, datatype, left, 2,
                 recv_buf.data(), local_nx, datatype, right, 2, cart_comm, &status);
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, local_ny + 1, ny)] = recv_buf[i];
    }

    for (size_t i = 0; i < local_nx; ++i) {
        send_buf[i] = grid[idx(i + 1, local_ny, ny)];
    }
    MPI_Sendrecv(send_buf.data(), local_nx, datatype, right, 3,
                 recv_buf.data(), local_nx, datatype, left, 3, cart_comm, &status);
    for (size_t i = 0; i < local_nx; ++i) {
        grid[idx(i + 1, 0, ny)] = recv_buf[i];
    }
}

// Runge-Kutta 4 step
template<typename Real>
size_t dualSystemEquation<Real>::rk4_step_2d(parallel_data<Real> parallel) 
{
    std::pair<std::vector<double>::iterator, std::vector<double>::iterator> mnmx_u, mnmx_v;

    size_t nx = dField.nx;
    size_t ny = dField.ny;
    size_t size = (nx + 2) * (ny + 2);
    std::vector<double> k1u(size), k2u(size), k3u(size), k4u(size);
    std::vector<double> k1v(size), k2v(size), k3v(size), k4v(size);
    std::vector<double> Lu(size), Lv(size), ut(size), vt(size);
    
    MPI_Datatype datatype = parallel.datatype;
    MPI_Comm cart_comm    = parallel.comm;
    size_t up    = parallel.up;
    size_t down  = parallel.down;
    size_t left  = parallel.left;
    size_t right = parallel.right;
    Real tol_u   = parallel.tol_u;
    Real tol_v   = parallel.tol_v;
    Real mgr_s   = parallel.snorm;
    
    size_t mpi_size = 0;

    // k1
    if ((parallel.compression==1) && (tol_u>0)) {
        mpi_size += exchange_ghost_cells_mgr(u_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if ((parallel.compression==2) && (tol_u>0)) { 
        mpi_size += exchange_ghost_cells_SZ(u_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    } else {
        exchange_ghost_cells(u_n, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    if ((parallel.compression==1) && (tol_v>0)) {
        mpi_size += exchange_ghost_cells_mgr(v_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if ((parallel.compression==2) && (tol_v>0)) {
        mpi_size += exchange_ghost_cells_SZ(v_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    } else {
        exchange_ghost_cells(v_n, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    compute_laplacian(u_n, Lu, nx, ny, dh);
    compute_laplacian(v_n, Lv, nx, ny, dh);

    for (size_t i = 1; i <= nx; ++i) {
        for (size_t j = 1; j <= ny; ++j) {
            size_t id = idx(i, j, ny + 2);
            k1u[id] = A - (B + 1) * u_n[id] + u_n[id] * u_n[id] * v_n[id] + Du * Lu[id];
            k1v[id] = B * u_n[id] - u_n[id] * u_n[id] * v_n[id] + Dv * Lv[id];
            ut[id]  = u_n[id] + 0.5 * dt * k1u[id];
            vt[id]  = v_n[id] + 0.5 * dt * k1v[id];
        }
    }
    
    // k2
    if ((parallel.compression==1) && (tol_u>0)) {
        mpi_size += exchange_ghost_cells_mgr(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if ((parallel.compression==2) && (tol_u>0)) { 
        mpi_size += exchange_ghost_cells_SZ(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    } else {
        exchange_ghost_cells(ut, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    if (parallel.compression && (tol_v>0)) {
        mpi_size += exchange_ghost_cells_mgr(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if ((parallel.compression==2) && (tol_v>0)) { 
        mpi_size += exchange_ghost_cells_SZ(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    } else {
        exchange_ghost_cells(vt, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    compute_laplacian(ut, Lu, nx, ny, dh);
    compute_laplacian(vt, Lv, nx, ny, dh);
    for (size_t i = 1; i <= nx; ++i) {
        for (size_t j = 1; j <= ny; ++j) {
            size_t id = idx(i, j, ny + 2);
            k2u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k2v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
            ut[id] = u_n[id] + 0.5 * dt * k2u[id];
            vt[id] = v_n[id] + 0.5 * dt * k2v[id];
        }
    }

    // k3
    if (parallel.compression && (tol_u>0)) {
        mpi_size += exchange_ghost_cells_mgr(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if ((parallel.compression==2) && (tol_u>0)) {
        mpi_size += exchange_ghost_cells_SZ(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    } else {
        exchange_ghost_cells(ut, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    if (parallel.compression && (tol_v>0)) {
        mpi_size += exchange_ghost_cells_mgr(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if ((parallel.compression==2) && (tol_v>0)) {
        mpi_size += exchange_ghost_cells_SZ(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    } else {
        exchange_ghost_cells(vt, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    compute_laplacian(ut, Lu, nx, ny, dh);
    compute_laplacian(vt, Lv, nx, ny, dh);
    for (size_t i = 1; i <= nx; ++i) {
        for (size_t j = 1; j <= ny; ++j) {
            size_t id = idx(i, j, ny + 2);
            k3u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k3v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
            ut[id] = u_n[id] + dt * k3u[id];
            vt[id] = v_n[id] + dt * k3v[id];
        }
    }

    // k4
    if (parallel.compression && (tol_u>0)) {
        mpi_size += exchange_ghost_cells_mgr(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if ((parallel.compression==2) && (tol_u>0)) {
        mpi_size += exchange_ghost_cells_SZ(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    } else {
        exchange_ghost_cells(ut, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    if (parallel.compression && (tol_v>0)) {
        mpi_size += exchange_ghost_cells_mgr(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if ((parallel.compression==2) && (tol_v>0)) {
        mpi_size += exchange_ghost_cells_SZ(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    } else {
        exchange_ghost_cells(vt, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    compute_laplacian(ut, Lu, nx, ny, dh);
    compute_laplacian(vt, Lv, nx, ny, dh);
    for (size_t i = 1; i <= nx; ++i) {
        for (size_t j = 1; j <= ny; ++j) {
            size_t id = idx(i, j, ny + 2);
            k4u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k4v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
        }
    }

    // Final update
    for (size_t i = 1; i <= nx; ++i) {
        for (size_t j = 1; j <= ny; ++j) {
            size_t id = idx(i, j, ny + 2);
            u_n[id] += dt / 6.0 * (k1u[id] + 2 * k2u[id] + 2 * k3u[id] + k4u[id]);
            v_n[id] += dt / 6.0 * (k1v[id] + 2 * k2v[id] + 2 * k3v[id] + k4v[id]);
        }
    }

    return mpi_size;
}

