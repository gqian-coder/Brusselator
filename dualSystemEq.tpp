#include <array>
#include <ostream>
#include <vector>
#include <Eigen/Core>
#include "io.hpp"
#include "mgard/compress_x.hpp"
#include "SZ3/api/sz.hpp"

// Helper to index 2D arrays stored in 1D
inline size_t idx(size_t i, size_t j, size_t ny) { return i * ny + j; }


template<typename Real>
dualSystemEquation<Real>::dualSystemEquation(field fieldData, Real dt, Real dh, 
                size_t ghostZ_len, Real Du, Real Dv, Real A, Real B)
               : dField(fieldData), dt(dt), dh(dh), ghostZ_len(ghostZ_len),
                 Du(Du), Dv(Dv), A(A), B(B) 
{
    // create a ghost zone to host the boundary data for mpi exchange
    size_t buffer_L = ghostZ_len * 2;
    data_size = dField.nz>1 ? (dField.nx+buffer_L) * (dField.ny+buffer_L) * (dField.nz+buffer_L) 
                            : (dField.nx+buffer_L) * (dField.ny+buffer_L);
    //std::cout << "data_size = " << data_size << "\n";
    u_n.resize(data_size, A);//+0.1);
    v_n.resize(data_size, B / A);//+0.2);
}


template<typename Real>
void dualSystemEquation<Real>::init_u_2d(Real *data)
{
    size_t buffer_L = ghostZ_len * 2;
    for (size_t i=0; i<dField.nx; i++) {
        std::copy(data+i*dField.ny, data+i*dField.ny+dField.ny, 
                  u_n.data()+(i+ghostZ_len)*(dField.ny+buffer_L) + ghostZ_len);
    }
}

template<typename Real>
void dualSystemEquation<Real>::init_v_2d(Real *data)
{
    size_t buffer_L = ghostZ_len * 2;
    for (size_t i=0; i<dField.nx; i++) {
        std::copy(data+i*dField.ny, data+i*dField.ny+dField.ny, 
                  v_n.data()+(i+ghostZ_len)*(dField.ny+buffer_L) + ghostZ_len);
    }
}

// Compute Laplacian using 5-posize_t stencil
template<typename Real>
void dualSystemEquation<Real>::compute_laplacian(const std::vector<Real>& grid, std::vector<Real>& lap,
                                                size_t edge_nx, size_t edge_ny, size_t ny, 
                                                size_t init_pos, Real dh)
{
    Real dh2 = dh*dh;
    for (size_t i = init_pos; i <= edge_nx; ++i) {
        for (size_t j = init_pos; j <= edge_ny; ++j) {
            lap[idx(i, j, ny)] = (
                grid[idx(i + 1, j, ny)] + grid[idx(i - 1, j, ny)]
                + grid[idx(i, j + 1, ny)] + grid[idx(i, j - 1, ny)]
                - 4 * grid[idx(i, j, ny)]) / dh2;
        }
    }
}

// Compute Laplacian using 5-posize_t stencil
template<typename Real>
void dualSystemEquation<Real>::compute_laplacian_float16(const std::vector<Real>& grid, std::vector<Real>& lap,
                                                size_t edge_nx, size_t edge_ny, size_t ny,
                                                size_t init_pos, Real dh)
{
    std::vector<Eigen::half> lap_half(lap.size());
    Real dh2 = dh*dh;
    for (size_t i = init_pos; i <= edge_nx; ++i) {
        for (size_t j = init_pos; j <= edge_ny; ++j) {
            lap_half[idx(i, j, ny)] = Eigen::half(
               (grid[idx(i + 1, j, ny)] + grid[idx(i - 1, j, ny)]
                + grid[idx(i, j + 1, ny)] + grid[idx(i, j - 1, ny)]
                - 4 * grid[idx(i, j, ny)]) / dh2);
        }
    }
    // convert float16 back to float32
    std::transform(lap_half.begin(), lap_half.end(), lap.begin(),
               [](Eigen::half h){ return static_cast<Real>(h); });
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
    Real *recv_buf_ptr = recv_buf.data() + total_buffer - 2*ny + 1;
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
    size_t compressed_size = total_buffer * sizeof(Real) * 2;
    size_t recv_size_up, recv_size_down, recv_size_left, recv_size_right;

    // compression the entire sub-domain
    char *bufferOut = (char *) malloc(compressed_size);
    char *compressed_data = bufferOut;

    compressed_size = SZ_compress(conf, grid.data(), compressed_data, compressed_size);

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
    Real *recv_buf_ptr = recv_buf.data() + total_buffer - 2*ny + 1;
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

// Exchange extended ghost zone (L=4 for 2D) with compression along edges
// No compression 
// to down, left_down, right_down, then right boundary to right and left boundary to left
// number of rows: nx
// number of columns: ny
template<typename Real>
size_t dualSystemEquation<Real>::exchange_ghost_extended(std::vector<Real>& grid, size_t nx, size_t ny,
                          MPI_Datatype datatype, MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right,
                          size_t left_up, size_t left_down, size_t right_up, size_t right_down)
{
    MPI_Status status;
    size_t buffer_L = ghostZ_len * 2;
    size_t local_nx     = nx - buffer_L;
    size_t local_ny     = ny - buffer_L;
    size_t total_buffer = local_nx * local_ny;
    // receiver buffer: up, down, left, right, left up, left down, right up, right down
    std::vector<Real> recv_buf(total_buffer*8); // considering the ranks in diagnol directions
    std::vector<Real> data_buffer(total_buffer);
    size_t offset = 0;

    // trim the buffer zone --> communicate data only
    Real *src_ptr = grid.data() + ghostZ_len * ny + ghostZ_len;
    for (size_t i=0; i<local_nx; i++) {
        std::copy(src_ptr, src_ptr + local_ny, &data_buffer[idx(i, 0, local_ny)]);
        src_ptr += ny;
    } 
    // Up/down communication (rows) the compressed data
    Real *recv_buf_up = recv_buf.data();
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, down, 8,
                 recv_buf_up, total_buffer, datatype, up, 8, cart_comm, &status);

    offset = total_buffer;
    Real *recv_buf_down = recv_buf.data() + offset;
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, up, 9,
                 recv_buf_down, total_buffer, datatype, down, 9, cart_comm, &status);

    // Left/right communication (columns) the compressed data
    offset += total_buffer;
    Real *recv_buf_left = recv_buf.data() + offset;
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, right, 10,
                 recv_buf_left, total_buffer, datatype, left, 10, cart_comm, &status);

    offset += total_buffer;
    Real *recv_buf_right = recv_buf.data() + offset;
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, left, 11,
                 recv_buf_right, total_buffer, datatype, right, 11, cart_comm, &status);

    // Left up/ left down communication (rows) the compressed data
    offset += total_buffer;
    Real *recv_buf_left_up = recv_buf.data() + offset;
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, right_down, 15,
                 recv_buf_left_up, total_buffer, datatype, left_up, 15, cart_comm, &status);

    offset += total_buffer;
    Real *recv_buf_left_down = recv_buf.data() + offset;
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, right_up, 14,
                 recv_buf_left_down, total_buffer, datatype, left_down, 14, cart_comm, &status);

    // right up/ left down communication (rows) the compressed data
    offset += total_buffer;
    Real *recv_buf_right_up = recv_buf.data() + offset;
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, left_down, 13,
                 recv_buf_right_up, total_buffer, datatype, right_up, 13, cart_comm, &status);

    offset += total_buffer;
    Real *recv_buf_right_down = recv_buf.data() + offset;
    MPI_Sendrecv(data_buffer.data(), total_buffer, datatype, left_up, 12,
                 recv_buf_right_down, total_buffer, datatype, right_down, 12, cart_comm, &status);

    MPI_Barrier(cart_comm);

    // copy back to the current grid
    size_t start_row     = local_nx + ghostZ_len;
    size_t start_col     = ghostZ_len + local_ny;
    size_t skipZone_up   = (local_nx-ghostZ_len)*local_ny;
    // Up: skip the ghost zone on the bottom
    Real *recv_buf_ptr = recv_buf.data() + skipZone_up;
    for (size_t i=0; i<ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + local_ny, &grid[idx(i, ghostZ_len, ny)]);
        recv_buf_ptr += local_ny;
    }
    /* 
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    if ((my_rank==0) || (my_rank==4) || (my_rank==15) || (my_rank==19) || (my_rank==20) || (my_rank==24)) {
        char filename[256];
        sprintf(filename, "viz_UpMPI_rank_%d.bin", my_rank);
        FILE *fp = fopen(filename, "wb");
        fwrite(recv_buf.data(), sizeof(Real), local_nx*local_ny, fp);
        fclose(fp);
    }
    */
    
    // Down: skip the ghost zone on the top
    recv_buf_ptr     = recv_buf.data() + total_buffer;
    for (size_t i=0; i<ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + local_ny, &grid[idx(start_row+i, ghostZ_len, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Left: skip the ghost zone on the right
    recv_buf_ptr = recv_buf.data() + total_buffer*2 + local_nx-ghostZ_len;
    for (size_t i = 0; i < local_nx; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(ghostZ_len+i, 0, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Right: skip the ghost zone on the left
    recv_buf_ptr = recv_buf.data() + total_buffer*3;
    for (size_t i = 0; i < local_nx; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(ghostZ_len+i, start_col, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Left up: the bottom piece from the left up neighbor
    recv_buf_ptr = recv_buf.data() + total_buffer*4 + skipZone_up + local_ny - ghostZ_len;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(i, 0, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Left down: the up piece from the left down neighbor
    recv_buf_ptr = recv_buf.data() + total_buffer*5 + local_ny - ghostZ_len;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(start_row+i, 0, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Right up: the bottom piece from the right up neighbor
    recv_buf_ptr = recv_buf.data() + total_buffer*6 + skipZone_up;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(i, start_col, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Right down: the up piece from the right down neighbor
    recv_buf_ptr = recv_buf.data() + total_buffer*7;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(start_row+i, start_col, ny)]);
        recv_buf_ptr += local_ny;
    }

    return 0;
}

// Exchange extended ghost zone (L=4 for 2D) with compression along edges
// Compress the entire rank data + buffer zone data (from the last timestep)
// to down, left_down, right_down, then right boundary to right and left boundary to left
// number of rows: nx
// number of columns: ny
template<typename Real>
size_t dualSystemEquation<Real>::exchange_ghost_extended_mgr(std::vector<Real>& grid, size_t nx, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right,
                          size_t left_up, size_t left_down, size_t right_up, size_t right_down,
                          Real tol, Real s)
{
    MPI_Status status;
    size_t buffer_L = ghostZ_len * 2;
    size_t local_nx     = nx - buffer_L;
    size_t local_ny     = ny - buffer_L;
    size_t total_buffer = local_nx * local_ny;
    // receiver buffer: up, down, left, right, left up, left down, right up, right down
    std::vector<Real> recv_buf(total_buffer*8); // considering the ranks in diagnol directions
    std::vector<Real> data_buffer(total_buffer);
    size_t total_buffer_bytes = total_buffer*sizeof(Real);
    size_t offset = 0;

    // trim the buffer zone --> communicate data only
    Real *src_ptr = grid.data() + ghostZ_len * ny + ghostZ_len;
    for (size_t i=0; i<local_nx; i++) {
        std::copy(src_ptr, src_ptr + local_ny, &data_buffer[idx(i, 0, local_ny)]);
        src_ptr += ny;
    }

    // compression parameters
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::SERIAL;
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    config.normalize_coordinates = true;
    std::vector<mgard_x::SIZE> data_shape{local_nx, local_ny};

    // Make more room for compressed data as the lossless coding may lead to a larger compressed
    // size when the input data size is small
    // received size: up, down, left, right, left up, left down, right up, right down
    size_t compressed_size = total_buffer_bytes;
    // received data size
    std::vector<size_t> recv_size(8, 0);

    // compression the entire sub-domain
    char *bufferOut = (char *) malloc(compressed_size);
    std::vector<unsigned char> recv_buf_compressed(total_buffer_bytes*8);

    void *compressed_data = bufferOut;

    // compress the entire sub-domain 
    mgard_x::compress(2, mgard_x::data_type::Double, data_shape, tol, s,
                mgard_x::error_bound_type::ABS, data_buffer.data(),
                compressed_data, compressed_size, config, true);

    MPI_Barrier(cart_comm);
    // Up/down communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, up, 0,
                 &recv_size[1], 1, MPI_UNSIGNED_LONG, down, 0, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, down, 1,
                 &recv_size[0], 1, MPI_UNSIGNED_LONG, up, 1, cart_comm, &status);
    // std::cout << "communicate size up and down\n";

    // Left/right communication (columns) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, left, 2,
                 &recv_size[3], 1, MPI_UNSIGNED_LONG, right, 2, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, right, 3,
                 &recv_size[2], 1, MPI_UNSIGNED_LONG, left, 3, cart_comm, &status);
    // std::cout << "communicate size left and right\n";

    // Up left /down left communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, left_up, 4,
                 &recv_size[7], 1, MPI_UNSIGNED_LONG, right_down, 4, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, left_down, 5,
                 &recv_size[6], 1, MPI_UNSIGNED_LONG, right_up, 5, cart_comm, &status);

    // Up right /down right communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, right_up, 6,
                 &recv_size[5], 1, MPI_UNSIGNED_LONG, left_down, 6, cart_comm, &status);
    MPI_Sendrecv(&compressed_size, 1, MPI_UNSIGNED_LONG, right_down, 7,
                 &recv_size[4], 1, MPI_UNSIGNED_LONG, left_up, 7, cart_comm, &status);

    MPI_Barrier(cart_comm);

    // Up/down communication (rows) the compressed data
    unsigned char *recv_compressed_up = recv_buf_compressed.data();
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, down, 8,
                 recv_compressed_up, recv_size[0], MPI_BYTE, up, 8, cart_comm, &status);

    offset = recv_size[0];
    unsigned char *recv_compressed_down = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, up, 9,
                 recv_compressed_down, recv_size[1], MPI_BYTE, down, 9, cart_comm, &status);

    // Left/right communication (columns) the compressed data
    offset += recv_size[1];
    unsigned char *recv_compressed_left = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, right, 10,
                 recv_compressed_left, recv_size[2], MPI_BYTE, left, 10, cart_comm, &status);

    offset += recv_size[2];
    unsigned char *recv_compressed_right = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, left, 11,
                 recv_compressed_right, recv_size[3], MPI_BYTE, right, 11, cart_comm, &status);

    // Left up/ left down communication (rows) the compressed data
    offset += recv_size[3];
    unsigned char *recv_compressed_left_up = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, right_down, 15,
                 recv_compressed_left_up, recv_size[4], MPI_BYTE, left_up, 15, cart_comm, &status);

    offset += recv_size[4];
    unsigned char *recv_compressed_left_down = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, right_up, 14,
                 recv_compressed_left_down, recv_size[5], MPI_BYTE, left_down, 14, cart_comm, &status);

    // right up/ left down communication (rows) the compressed data
    offset += recv_size[5];
    unsigned char *recv_compressed_right_up = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, left_down, 13,
                 recv_compressed_right_up, recv_size[6], MPI_BYTE, right_up, 13, cart_comm, &status);

    offset += recv_size[6];
    unsigned char *recv_compressed_right_down = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data, compressed_size, MPI_BYTE, left_up, 12,
                 recv_compressed_right_down, recv_size[7], MPI_BYTE, right_down, 12, cart_comm, &status);

    // Decompression
    void *recv_buf_up = static_cast<void*>(recv_buf.data());
    mgard_x::decompress(recv_compressed_up, recv_size[0], recv_buf_up, config, true);

    offset = total_buffer;
    void *recv_buf_down = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_down, recv_size[1], recv_buf_down, config, true);

    offset += total_buffer;
    void *recv_buf_left = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_left, recv_size[2], recv_buf_left, config, true);

    offset += total_buffer;
    void *recv_buf_right = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_right, recv_size[3], recv_buf_right, config, true);

    offset += total_buffer;
    void *recv_buf_left_up = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_left_up, recv_size[4], recv_buf_left_up, config, true);

    offset += total_buffer;
    void *recv_buf_left_down = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_left_down, recv_size[5], recv_buf_left_down, config, true);

    offset += total_buffer;
    void *recv_buf_right_up = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_right_up, recv_size[6], recv_buf_right_up, config, true);

    offset += total_buffer;
    void *recv_buf_right_down = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_right_down, recv_size[7], recv_buf_right_down, config, true);

    // copy back to the current grid
    size_t start_row     = local_nx + ghostZ_len;
    size_t start_col     = ghostZ_len + local_ny;
    size_t skipZone_up   = (local_nx - ghostZ_len) * local_ny;
    // Up
    Real *recv_buf_ptr = recv_buf.data() + skipZone_up;
    for (size_t i=0; i<ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + local_ny, &grid[idx(i, ghostZ_len, ny)]);
        recv_buf_ptr += local_ny;
    }

    /* 
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    if ((my_rank==0) || (my_rank==4) || (my_rank==15) || (my_rank==19) || (my_rank==20) || (my_rank==24)) {
        char filename[256];
        sprintf(filename, "viz_UpMPI_rank_%d.bin", my_rank);
        FILE *fp = fopen(filename, "wb");
        fwrite(recv_buf.data(), sizeof(Real), local_nx*local_ny, fp);
        fclose(fp);
    } 
    */
    
    // Down
    recv_buf_ptr     = recv_buf.data() + total_buffer;
    for (size_t i=0; i<ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + local_ny, &grid[idx(start_row+i, ghostZ_len, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Left
    recv_buf_ptr = recv_buf.data() + total_buffer*2 + local_nx - ghostZ_len;
    for (size_t i = 0; i < local_nx; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(ghostZ_len+i, 0, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Right
    recv_buf_ptr = recv_buf.data() + total_buffer*3;
    for (size_t i = 0; i < local_nx; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(ghostZ_len+i, start_col, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Left up
    recv_buf_ptr = recv_buf.data() + total_buffer*4 + skipZone_up + local_ny - ghostZ_len;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(i, 0, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Left down
    recv_buf_ptr = recv_buf.data() + total_buffer*5 + local_ny - ghostZ_len;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(start_row+i, 0, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Right up
    recv_buf_ptr = recv_buf.data() + total_buffer*6 + skipZone_up;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(i, start_col, ny)]);
        recv_buf_ptr += local_ny;
    }
    // Right down
    recv_buf_ptr = recv_buf.data() + total_buffer*7;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(start_row+i, start_col, ny)]);
        recv_buf_ptr += local_ny;
    }

    free(bufferOut);

    return compressed_size;
}
/*    
// Exchange extended ghost zone (L=4 for 2D) with compression along edges
// Only compress the extended edges + buffer zone data (from the last timestep)
// Send extended up boundary to up, left_up, right_up ranks, the extended down boundary 
// to down, left_down, right_down, then right boundary to right and left boundary to left
// number of rows: nx
// number of columns: ny
template<typename Real>
size_t dualSystemEquation<Real>::exchange_ghost_extended_mgr(std::vector<Real>& grid, size_t nx, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right, 
                          size_t left_up, size_t left_down, size_t right_up, size_t right_down,
                          Real tol, Real s)
{
    MPI_Status status;
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    size_t buffer_L = ghostZ_len * 2;
    size_t total_buffer_nx = buffer_L * nx;
    size_t total_buffer_ny = buffer_L * ny;
    size_t local_nx        = nx - buffer_L;
    size_t local_ny        = ny - buffer_L;
    size_t total_buffer    = 6*total_buffer_ny + 2*total_buffer_nx;
    // receiver buffer: up, down, left, right, left up, left down, right up, right down
    std::vector<Real> recv_buf(total_buffer); // considering the ranks in diagnol directions
    size_t total_buffer_bytes = total_buffer*sizeof(Real);
    size_t up_buffer_bytes    = total_buffer_nx * sizeof(Real);
    size_t left_buffer_bytes  = total_buffer_ny * sizeof(Real);

    // compression parameters
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::SERIAL;
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    config.normalize_coordinates = true;
    std::vector<mgard_x::SIZE> data_shape_nx{nx, buffer_L};
    std::vector<mgard_x::SIZE> data_shape_ny{buffer_L, ny};

    // Make more room for compressed data as the lossless coding may lead to a larger compressed
    // size when the input data size is small
    size_t m_enLarge = 2;
    // compressed size: up, down, left, right, left up, left down, right up, right down
    std::vector<size_t> compressed_size(4, up_buffer_bytes*m_enLarge);
    compressed_size[2] = left_buffer_bytes*m_enLarge;
    compressed_size[3] = left_buffer_bytes*m_enLarge;
    // received data size
    std::vector<size_t> recv_size(8, 0);
    
    // compression the entire sub-domain
    char *bufferOut = (char *) malloc((left_buffer_bytes + up_buffer_bytes ) * 2 * m_enLarge);
    std::vector<unsigned char> recv_buf_compressed(total_buffer_bytes * m_enLarge);

    size_t offset = compressed_size[0];
    void *compressed_data_up         = bufferOut;
    void *compressed_data_down       = bufferOut + offset;
    offset += compressed_size[1];
    void *compressed_data_left       = bufferOut + offset; 
    offset += compressed_size[2];
    void *compressed_data_right      = bufferOut + offset; 

    // compress up 
    mgard_x::compress(2, mgard_x::data_type::Double, data_shape_ny, tol, s,
                mgard_x::error_bound_type::ABS, grid.data(),
                compressed_data_up, compressed_size[0], config, true);
    // compress down
    mgard_x::compress(2, mgard_x::data_type::Double, data_shape_ny, tol, s,
                mgard_x::error_bound_type::ABS, grid.data() + local_nx * ny, 
                compressed_data_down, compressed_size[1], config, true);

    Real *data_p, *data_l, *data_r;
    Real *buffer_in_left  = (Real *) malloc(left_buffer_bytes);
    Real *buffer_in_right = (Real *) malloc(left_buffer_bytes);
    data_p = grid.data();
    data_l = buffer_in_left;
    data_r = buffer_in_right;
    offset = local_ny + buffer_L;
    for (size_t i=0; i<nx; i++) {
        std::copy(data_p, data_p + buffer_L, data_l);
        std::copy(data_p + local_ny, data_p + offset, data_r);
        data_p += ny;
        data_l += buffer_L;
        data_r += buffer_L;
    }    
    // compress left
    mgard_x::compress(2, mgard_x::data_type::Double, data_shape_nx, tol, s,
                mgard_x::error_bound_type::ABS, buffer_in_left,  
                compressed_data_left, compressed_size[2], config, true);

    // compress right
    mgard_x::compress(2, mgard_x::data_type::Double, data_shape_nx, tol, s,
                mgard_x::error_bound_type::ABS, buffer_in_right,
                compressed_data_right, compressed_size[3], config, true);
    
    //std::cout << "compressed size = " << compressed_size[0] << ", " << compressed_size[1] << ", " << compressed_size[2] << ", " << compressed_size[3] << "\n";
    MPI_Barrier(cart_comm);
    // Up/down communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size[0], 1, MPI_UNSIGNED_LONG, up, 0,
                 &recv_size[1], 1, MPI_UNSIGNED_LONG, down, 0, cart_comm, &status);
    MPI_Sendrecv(&compressed_size[1], 1, MPI_UNSIGNED_LONG, down, 1,
                 &recv_size[0], 1, MPI_UNSIGNED_LONG, up, 1, cart_comm, &status);
    // std::cout << "communicate size up and down\n";

    // Left/right communication (columns) the compressed size
    MPI_Sendrecv(&compressed_size[2], 1, MPI_UNSIGNED_LONG, left, 2,
                 &recv_size[3], 1, MPI_UNSIGNED_LONG, right, 2, cart_comm, &status);
    MPI_Sendrecv(&compressed_size[3], 1, MPI_UNSIGNED_LONG, right, 3,
                 &recv_size[2], 1, MPI_UNSIGNED_LONG, left, 3, cart_comm, &status);
    // std::cout << "communicate size left and right\n";   

    // Up left /down left communication (rows) the compressed size
    // std::cout << "rank " << my_rank << " step1: send to rank " << left_up << ", receive from rank " << right_down << "\n";
    MPI_Sendrecv(&compressed_size[0], 1, MPI_UNSIGNED_LONG, left_up, 4,
                 &recv_size[7], 1, MPI_UNSIGNED_LONG, right_down, 4, cart_comm, &status);
    // std::cout << "step2: send to rank " << left_down << ", receive from rank " << right_up << "\n";
    MPI_Sendrecv(&compressed_size[1], 1, MPI_UNSIGNED_LONG, left_down, 5,
                 &recv_size[6], 1, MPI_UNSIGNED_LONG, right_up, 5, cart_comm, &status);
 
    // Up right /down right communication (rows) the compressed size
    MPI_Sendrecv(&compressed_size[0], 1, MPI_UNSIGNED_LONG, right_up, 6,
                 &recv_size[5], 1, MPI_UNSIGNED_LONG, left_down, 6, cart_comm, &status);
    MPI_Sendrecv(&compressed_size[1], 1, MPI_UNSIGNED_LONG, right_down, 7,
                 &recv_size[4], 1, MPI_UNSIGNED_LONG, left_up, 7, cart_comm, &status);    
   
    std::cout << "rank " << my_rank << " send " << compressed_size[0] << " to rank " << up << ", " << compressed_size[0] << " to rank " << left_up << ", " << compressed_size[0] << " to rank " << right_up << ", " << compressed_size[1] << " to rank " << down << ", " << compressed_size[1] << " to rank " << left_down << ", " << compressed_size[1] << " to rank " << right_down << ", " << compressed_size[2] << " to rank " << left << ", " << compressed_size[3] << " to rank " << right << "\n";
    std::cout << "rank " << my_rank << " received " << recv_size[0] << " from rank " << up << ", " << recv_size[4] << " from rank " << left_up << ", " << recv_size[6] << " from rank " << right_up << ", " << recv_size[1] << " from rank " << down << ", " << recv_size[4] << " from rank " << left_down << ", " << recv_size[7] << " from rank " << right_down << ", " << recv_size[2] << " from rank " << left << ", " << recv_size[3] << " from rank " << right << "\n";

    MPI_Barrier(cart_comm);
    
    // Up/down communication (rows) the compressed data
    offset = 0;
    unsigned char *recv_compressed_up = recv_buf_compressed.data();
    MPI_Sendrecv(compressed_data_down, compressed_size[1], MPI_BYTE, down, 8,
                 recv_compressed_up, recv_size[0], MPI_BYTE, up, 8, cart_comm, &status);

    offset += recv_size[0];
    unsigned char *recv_compressed_down = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data_up, compressed_size[0], MPI_BYTE, up, 9,
                 recv_compressed_down, recv_size[1], MPI_BYTE, down, 9, cart_comm, &status);
    
    // Left/right communication (columns) the compressed data
    offset += recv_size[1];
    unsigned char *recv_compressed_left = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data_right, compressed_size[3], MPI_BYTE, right, 10,
                 recv_compressed_left, recv_size[2], MPI_BYTE, left, 10, cart_comm, &status);

    offset += recv_size[2];
    unsigned char *recv_compressed_right = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data_left, compressed_size[2], MPI_BYTE, left, 11,
                 recv_compressed_right, recv_size[3], MPI_BYTE, right, 11, cart_comm, &status);
    
    // Left up/ left down communication (rows) the compressed data
    offset += recv_size[3];
    unsigned char *recv_compressed_left_up = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data_down, compressed_size[1], MPI_BYTE, right_down, 15,
                 recv_compressed_left_up, recv_size[4], MPI_BYTE, left_up, 15, cart_comm, &status);
    
    offset += recv_size[4];
    unsigned char *recv_compressed_left_down = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data_up, compressed_size[0], MPI_BYTE, right_up, 14,
                 recv_compressed_left_down, recv_size[5], MPI_BYTE, left_down, 14, cart_comm, &status);
 
    // right up/ left down communication (rows) the compressed data 
    offset += recv_size[5];
    unsigned char *recv_compressed_right_up = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data_down, compressed_size[1], MPI_BYTE, left_down, 13,
                 recv_compressed_right_up, recv_size[6], MPI_BYTE, right_up, 13, cart_comm, &status);

    offset += recv_size[6];
    unsigned char *recv_compressed_right_down = recv_buf_compressed.data() + offset;
    MPI_Sendrecv(compressed_data_up, compressed_size[0], MPI_BYTE, left_up, 12,
                 recv_compressed_right_down, recv_size[7], MPI_BYTE, right_down, 12, cart_comm, &status);


    // Decompression
    std::cout << "rank " << my_rank << " received size " << recv_size[0] << "\n";
    void *recv_buf_up = static_cast<void*>(recv_buf.data());
    mgard_x::decompress(recv_compressed_up, recv_size[0], recv_buf_up, config, true);
    std::cout << "rank " << my_rank << " decompression finished up\n";

    offset = total_buffer_ny;
    void *recv_buf_down = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_down, recv_size[1], recv_buf_down, config, true);
    std::cout << "rank " << my_rank << " received size " << recv_size[1] << " decompression finished down\n";

    offset += total_buffer_ny;
    void *recv_buf_left = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_left, recv_size[2], recv_buf_left, config, true);

    offset += total_buffer_nx;
    void *recv_buf_right = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_right, recv_size[3], recv_buf_right, config, true);

    offset += total_buffer_nx;
    std::cout << "rank " << my_rank << " decompression finished up, down, left, right\n";

    void *recv_buf_left_up = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_left_up, recv_size[4], recv_buf_left_up, config, true);

    offset += total_buffer_ny;
    void *recv_buf_left_down = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_left_down, recv_size[5], recv_buf_left_down, config, true);

    offset += total_buffer_ny;
    void *recv_buf_right_up = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_right_up, recv_size[6], recv_buf_right_up, config, true);

    offset += total_buffer_ny;
    void *recv_buf_right_down = static_cast<void*>(recv_buf.data() + offset);
    mgard_x::decompress(recv_compressed_right_down, recv_size[7], recv_buf_right_down, config, true);
    std::cout << "rank " << my_rank << " decompression finished\n";

    // copy back to the current grid
    // Up: skip the ghost zone on the bottom 
    Real *recv_buf_ptr = recv_buf.data() + ghostZ_len;
    for (size_t i=0; i<ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + local_ny, &grid[idx(i, ghostZ_len, ny)]);
        recv_buf_ptr += ny;
    }
    // Down: skip the ghost zone on the top
    offset           = total_buffer_ny;
    recv_buf_ptr     = recv_buf.data() + offset + ghostZ_len*ny + ghostZ_len;
    size_t start_row = local_nx + ghostZ_len;
    size_t start_col = ghostZ_len + local_ny;
    for (size_t i=0; i<ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + local_ny, &grid[idx(start_row+i, ghostZ_len, ny)]);
        recv_buf_ptr += ny;
    }
    // Left: skip the ghost zone on the right
    offset       += total_buffer_ny;
    recv_buf_ptr = recv_buf.data() + offset + ghostZ_len*buffer_L;  
    for (size_t i = 0; i < local_nx; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(ghostZ_len+i, 0, ny)]);
        recv_buf_ptr += buffer_L;
    }
    // Right: skip the ghost zone on the left
    offset      += total_buffer_nx;
    recv_buf_ptr = recv_buf.data() + offset + ghostZ_len*buffer_L + ghostZ_len;
    for (size_t i = 0; i < local_nx; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(ghostZ_len+i, start_col, ny)]);
        recv_buf_ptr += buffer_L;
    }
    // Left up: the bottom piece from the left up neighbor
    offset      += total_buffer_nx;
    recv_buf_ptr = recv_buf.data() + offset + local_ny;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(i, 0, ny)]);
        recv_buf_ptr += ny;
    }
    // Left down: the up piece from the left down neighbor
    offset      += total_buffer_ny;
    recv_buf_ptr = recv_buf.data() + offset + ghostZ_len*ny + local_ny;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(start_row+i, 0, ny)]);
        recv_buf_ptr += ny;
    } 
    // Right up: the bottom piece from the right up neighbor
    offset      += total_buffer_ny;
    recv_buf_ptr = recv_buf.data() + offset + ghostZ_len;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(i, start_col, ny)]);
        recv_buf_ptr += ny;
    }
    // Right down: the up piece from the right down neighbor
    offset      += total_buffer_ny;
    recv_buf_ptr = recv_buf.data() + offset + ghostZ_len*ny + ghostZ_len;
    for (size_t i = 0; i < ghostZ_len; i++) {
        std::copy(recv_buf_ptr, recv_buf_ptr + ghostZ_len, &grid[idx(start_row+i, start_col, ny)]);
        recv_buf_ptr += ny;
    }
    
    free(buffer_in_left);
    free(buffer_in_right); 
    free(bufferOut);

    size_t total_compressed_size = std::accumulate(compressed_size.begin(), compressed_size.end(), 0);
    return total_compressed_size; 
}
*/

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
     
    Real *recv_buf_ptr = recv_buf.data();
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
    size_t nx = dField.nx;
    size_t ny = dField.ny;
    size_t size = (nx + 2) * (ny + 2);
    std::vector<Real> k1u(size), k2u(size), k3u(size), k4u(size);
    std::vector<Real> k1v(size), k2v(size), k3v(size), k4v(size);
    std::vector<Real> Lu(size), Lv(size), ut(size), vt(size);
    
    MPI_Datatype datatype = parallel.datatype;
    MPI_Comm cart_comm    = parallel.comm;
    size_t up    = parallel.up;
    size_t down  = parallel.down;
    size_t left  = parallel.left;
    size_t right = parallel.right;
    Real tol_u   = parallel.tol_u;
    Real tol_v   = parallel.tol_v;
    Real mgr_s   = parallel.snorm;
    
    size_t mpi_size    = 0;
    size_t extended_ny = ny + 2; 

    // k1
    // fill up the ghost zone 
    if ((parallel.compression==0) || (tol_u==0)) {
        exchange_ghost_cells(u_n, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(u_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if (parallel.compression==2) {
        mpi_size += exchange_ghost_cells_SZ(u_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    }
    if ((parallel.compression==0) || (tol_v==0)) {
        exchange_ghost_cells(v_n, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(v_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if (parallel.compression==2) {
        mpi_size += exchange_ghost_cells_SZ(v_n, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    }

    compute_laplacian(u_n, Lu, nx, ny, extended_ny, 1, dh);
    compute_laplacian(v_n, Lv, nx, ny, extended_ny, 1, dh);
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
    if ((parallel.compression==0) || (tol_u==0)) {
        exchange_ghost_cells(ut, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if (parallel.compression==2) { 
        mpi_size += exchange_ghost_cells_SZ(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    } 
    
    if ((parallel.compression==0) || (tol_v==0)) {
        exchange_ghost_cells(vt, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if (parallel.compression==2) { 
        mpi_size += exchange_ghost_cells_SZ(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    }
    
    compute_laplacian(ut, Lu, nx, ny, extended_ny, 1, dh);
    compute_laplacian(vt, Lv, nx, ny, extended_ny, 1, dh);
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
    if ((parallel.compression==0) || (tol_u==0)) {
        exchange_ghost_cells(ut, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if (parallel.compression==2) {
        mpi_size += exchange_ghost_cells_SZ(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    }
    if ((parallel.compression==0) || (tol_v==0)) {
        exchange_ghost_cells(vt, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if (parallel.compression==2) {
        mpi_size += exchange_ghost_cells_SZ(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    } 
    
    compute_laplacian(ut, Lu, nx, ny, extended_ny, 1, dh);
    compute_laplacian(vt, Lv, nx, ny, extended_ny, 1, dh);
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
    if ((parallel.compression==0) || (tol_u==0)) {
        exchange_ghost_cells(ut, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u, mgr_s);
    } else if (parallel.compression==2) {
        mpi_size += exchange_ghost_cells_SZ(ut, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_u);
    }
    if ((parallel.compression==0) || (tol_v==0)) {
        exchange_ghost_cells(vt, nx, ny, ny + 2, datatype, cart_comm, up, down, left, right);
    }
    else if (parallel.compression==1) {
        mpi_size += exchange_ghost_cells_mgr(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v, mgr_s);
    } else if (parallel.compression==2) {
        mpi_size += exchange_ghost_cells_SZ(vt, nx, ny, ny + 2, cart_comm, up, down, left, right, tol_v);
    }
    
    compute_laplacian(ut, Lu, nx, ny, extended_ny, 1, dh);
    compute_laplacian(vt, Lv, nx, ny, extended_ny, 1, dh);
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


// MPI communication after the entire RK4 procedure
// gradually reduce the zone of calculation from k1 --> k4 
// Note the ghost zone in this case is 4 at each side
// compression is only for the area that needs to be exchanged plus ghost zone
template<typename Real>
size_t dualSystemEquation<Real>::rk4_step_2d_extendedGhostZ(parallel_data<Real> parallel)
{
    size_t nx = dField.nx;
    size_t ny = dField.ny;
    size_t bufferL = 2*ghostZ_len;
    size_t size = (nx + bufferL) * (ny + bufferL);
    size_t extended_ny = ny + bufferL;
    size_t extended_nx = nx + bufferL;
    std::vector<Real> k1u(size), k2u(size), k3u(size), k4u(size);
    std::vector<Real> k1v(size), k2v(size), k3v(size), k4v(size);
    std::vector<Real> Lu(size), Lv(size), ut(size), vt(size);    

    MPI_Datatype datatype = parallel.datatype;
    MPI_Comm cart_comm    = parallel.comm;
    size_t up             = parallel.up;
    size_t down           = parallel.down;
    size_t left           = parallel.left;
    size_t right          = parallel.right;
    size_t left_up        = parallel.left_up;
    size_t left_down      = parallel.left_down; 
    size_t right_up       = parallel.right_up;
    size_t right_down     = parallel.right_down;
    Real tol_u            = parallel.tol_u;
    Real tol_v            = parallel.tol_v;
    Real mgr_s            = parallel.snorm;
    
    size_t mpi_size = 0;
    size_t edge_nx  = extended_nx-2;
    size_t edge_ny  = extended_ny-2;

    // Only exchange buffer zone at the beginning of the RK4
    if (parallel.compression==0) {
        mpi_size += exchange_ghost_extended(u_n, extended_nx, extended_ny, datatype, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down);
    } else if (parallel.compression==1) {
        mpi_size += exchange_ghost_extended_mgr(u_n, extended_nx, extended_ny, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down, tol_u, mgr_s);
    } 
    if (parallel.compression==0) {
        mpi_size += exchange_ghost_extended(v_n, extended_nx, extended_ny, datatype, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down);
    } else if (parallel.compression==1) {
        mpi_size += exchange_ghost_extended_mgr(v_n, extended_nx, extended_ny, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down, tol_v, mgr_s);
    }

    /* 
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    if ((my_rank==0) || (my_rank==4) || (my_rank==15) || (my_rank==19) || (my_rank==20) || (my_rank==24)) {
        char filename[256];
        sprintf(filename, "viz_ghostEx_rank_%d.bin", my_rank);
        FILE *fp = fopen(filename, "wb");
        fwrite(u_n.data(), sizeof(Real), extended_nx*extended_ny, fp);
        fclose(fp);
    }
    */

    // k1
    // ghost zone has been filled up with boundary data
    compute_laplacian(u_n, Lu, edge_nx, edge_ny, extended_ny, 1, dh);
    compute_laplacian(v_n, Lv, edge_nx, edge_ny, extended_ny, 1, dh);

    for (size_t i = 1; i <= edge_nx ; ++i) {
        for (size_t j = 1; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k1u[id] = A - (B + 1) * u_n[id] + u_n[id] * u_n[id] * v_n[id] + Du * Lu[id];
            k1v[id] = B * u_n[id] - u_n[id] * u_n[id] * v_n[id] + Dv * Lv[id];
            ut[id]  = u_n[id] + 0.5 * dt * k1u[id];
            vt[id]  = v_n[id] + 0.5 * dt * k1v[id];
        }
    }

    // k2: nx-2, ny-2
    edge_nx  = extended_nx-3;
    edge_ny  = extended_ny-3;
    compute_laplacian(ut, Lu, edge_nx, edge_ny, extended_ny, 2, dh);
    compute_laplacian(vt, Lv, edge_nx, edge_ny, extended_ny, 2, dh);
    
    for (size_t i = 2; i <= edge_nx; ++i) {
        for (size_t j = 2; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k2u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k2v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
            ut[id] = u_n[id] + 0.5 * dt * k2u[id];
            vt[id] = v_n[id] + 0.5 * dt * k2v[id];
        }
    }

    // k3: nx-3, ny-3
    edge_nx  = extended_nx-4;
    edge_ny  = extended_ny-4;
    compute_laplacian(ut, Lu, edge_nx, edge_ny, extended_ny, 3, dh);
    compute_laplacian(vt, Lv, edge_nx, edge_ny, extended_ny, 3, dh);
    
    for (size_t i = 3; i <= edge_nx; ++i) {
        for (size_t j = 3; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k3u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k3v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
            ut[id] = u_n[id] + 0.5 * dt * k3u[id];
            vt[id] = v_n[id] + 0.5 * dt * k3v[id];
        }
    }
        
    // k4: nx-4, ny-4
    edge_nx  = extended_nx-5;
    edge_ny  = extended_ny-5;
    compute_laplacian(ut, Lu, edge_nx, edge_ny, extended_ny, 4, dh);
    compute_laplacian(vt, Lv, edge_nx, edge_ny, extended_ny, 4, dh);
    
    for (size_t i = 4; i <= edge_nx; ++i) {
        for (size_t j = 4; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k4u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k4v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
        }
    }

    // Final update
    for (size_t i = 4; i <= edge_nx; ++i) {
        for (size_t j = 4; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            u_n[id] += dt / 6.0 * (k1u[id] + 2 * k2u[id] + 2 * k3u[id] + k4u[id]);
            v_n[id] += dt / 6.0 * (k1v[id] + 2 * k2v[id] + 2 * k3v[id] + k4v[id]);
        }
    }

    return mpi_size;
}

template<typename Real>
size_t dualSystemEquation<Real>::rk4_step_2d_extendedGhostZ_mixPrec(parallel_data<Real> parallel)
{
    size_t nx = dField.nx;
    size_t ny = dField.ny;
    size_t bufferL = 2*ghostZ_len;
    size_t size = (nx + bufferL) * (ny + bufferL);
    size_t extended_ny = ny + bufferL;
    size_t extended_nx = nx + bufferL;
    std::vector<Real> k1u(size), k2u(size), k3u(size), k4u(size);
    std::vector<Real> k1v(size), k2v(size), k3v(size), k4v(size);
    std::vector<Real> Lu(size), Lv(size), ut(size), vt(size);

    MPI_Datatype datatype = parallel.datatype;
    MPI_Comm cart_comm    = parallel.comm;
    size_t up             = parallel.up;
    size_t down           = parallel.down;
    size_t left           = parallel.left;
    size_t right          = parallel.right;
    size_t left_up        = parallel.left_up;
    size_t left_down      = parallel.left_down;
    size_t right_up       = parallel.right_up;
    size_t right_down     = parallel.right_down;
    Real tol_u            = parallel.tol_u;
    Real tol_v            = parallel.tol_v;
    Real mgr_s            = parallel.snorm;

    size_t mpi_size = 0;
    size_t edge_nx  = extended_nx-2;
    size_t edge_ny  = extended_ny-2;

    // Only exchange buffer zone at the beginning of the RK4
    if (parallel.compression==0) {
        mpi_size += exchange_ghost_extended(u_n, extended_nx, extended_ny, datatype, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down);
    } else if (parallel.compression==1) {
        mpi_size += exchange_ghost_extended_mgr(u_n, extended_nx, extended_ny, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down, tol_u, mgr_s);
    }
    if (parallel.compression==0) {
        mpi_size += exchange_ghost_extended(v_n, extended_nx, extended_ny, datatype, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down);
    } else if (parallel.compression==1) {
        mpi_size += exchange_ghost_extended_mgr(v_n, extended_nx, extended_ny, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down, tol_v, mgr_s);
    }

    // k1
    // ghost zone has been filled up with boundary data
    compute_laplacian(u_n, Lu, edge_nx, edge_ny, extended_ny, 1, dh);
    compute_laplacian(v_n, Lv, edge_nx, edge_ny, extended_ny, 1, dh);
    
    for (size_t i = 1; i <= edge_nx ; ++i) {
        for (size_t j = 1; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k1u[id] = A - (B + 1) * u_n[id] + u_n[id] * u_n[id] * v_n[id] + Du * Lu[id];
            k1v[id] = B * u_n[id] - u_n[id] * u_n[id] * v_n[id] + Dv * Lv[id];
            ut[id]  = u_n[id] + 0.5 * dt * k1u[id];
            vt[id]  = v_n[id] + 0.5 * dt * k1v[id];
        }
    }

    // k2: nx-2, ny-2
    edge_nx  = extended_nx-3;
    edge_ny  = extended_ny-3;
    compute_laplacian(ut, Lu, edge_nx, edge_ny, extended_ny, 2, dh);
    compute_laplacian(vt, Lv, edge_nx, edge_ny, extended_ny, 2, dh);

    for (size_t i = 2; i <= edge_nx; ++i) {
        for (size_t j = 2; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k2u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k2v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
            ut[id] = u_n[id] + 0.5 * dt * k2u[id];
            vt[id] = v_n[id] + 0.5 * dt * k2v[id];
        }
    }

    // k3: nx-3, ny-3
    edge_nx  = extended_nx-4;
    edge_ny  = extended_ny-4;
    compute_laplacian(ut, Lu, edge_nx, edge_ny, extended_ny, 3, dh);
    compute_laplacian(vt, Lv, edge_nx, edge_ny, extended_ny, 3, dh);
    
    for (size_t i = 3; i <= edge_nx; ++i) {
        for (size_t j = 3; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k3u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k3v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
            ut[id] = u_n[id] + 0.5 * dt * k3u[id];
            vt[id] = v_n[id] + 0.5 * dt * k3v[id];
        }
    }

    // k4: nx-4, ny-4
    edge_nx  = extended_nx-5;
    edge_ny  = extended_ny-5;
    compute_laplacian(ut, Lu, edge_nx, edge_ny, extended_ny, 4, dh);
    compute_laplacian(vt, Lv, edge_nx, edge_ny, extended_ny, 4, dh);
    
    for (size_t i = 4; i <= edge_nx; ++i) {
        for (size_t j = 4; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            k4u[id] = A - (B + 1) * ut[id] + ut[id] * ut[id] * vt[id] + Du * Lu[id];
            k4v[id] = B * ut[id] - ut[id] * ut[id] * vt[id] + Dv * Lv[id];
        }
    }

    // Final update
    for (size_t i = 4; i <= edge_nx; ++i) {
        for (size_t j = 4; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            u_n[id] += dt / 6.0 * (k1u[id] + 2 * k2u[id] + 2 * k3u[id] + k4u[id]);
            v_n[id] += dt / 6.0 * (k1v[id] + 2 * k2v[id] + 2 * k3v[id] + k4v[id]);
        }
    }

    return mpi_size;
}


// time integration through Euler method
template<typename Real>
size_t dualSystemEquation<Real>::Euler_step_2d_extendedGhostZ(parallel_data<Real> parallel)
{
    size_t nx = dField.nx;
    size_t ny = dField.ny;
    size_t bufferL = 2*ghostZ_len;
    size_t size = (nx + bufferL) * (ny + bufferL);
    size_t extended_ny = ny + bufferL;
    size_t extended_nx = nx + bufferL;
    std::vector<Real> Lu(size), Lv(size), ut(size), vt(size);

    MPI_Datatype datatype = parallel.datatype;
    MPI_Comm cart_comm    = parallel.comm;
    size_t up             = parallel.up;
    size_t down           = parallel.down;
    size_t left           = parallel.left;
    size_t right          = parallel.right;
    size_t left_up        = parallel.left_up;
    size_t left_down      = parallel.left_down;
    size_t right_up       = parallel.right_up;
    size_t right_down     = parallel.right_down;
    Real tol_u            = parallel.tol_u;
    Real tol_v            = parallel.tol_v;
    Real mgr_s            = parallel.snorm;

    size_t mpi_size = 0;
    size_t edge_nx  = extended_nx-2;
    size_t edge_ny  = extended_ny-2;


    // Only exchange buffer zone at the beginning of the RK4
    if (parallel.compression==0) {
        mpi_size += exchange_ghost_extended(u_n, extended_nx, extended_ny, datatype, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down);
    } else if (parallel.compression==1) {
        mpi_size += exchange_ghost_extended_mgr(u_n, extended_nx, extended_ny, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down, tol_u, mgr_s);
    }
    if (parallel.compression==0) {
        mpi_size += exchange_ghost_extended(v_n, extended_nx, extended_ny, datatype, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down);
    } else if (parallel.compression==1) {
        mpi_size += exchange_ghost_extended_mgr(v_n, extended_nx, extended_ny, cart_comm, up, down, left, right,
                                                left_up, left_down, right_up, right_down, tol_v, mgr_s);
    }

    // k1
    // ghost zone has been filled up with boundary data
    compute_laplacian(u_n, Lu, edge_nx, edge_ny, extended_ny, 1, dh);
    compute_laplacian(v_n, Lv, edge_nx, edge_ny, extended_ny, 1, dh);

    for (size_t i = 1; i <= edge_nx ; ++i) {
        for (size_t j = 1; j <= edge_ny; ++j) {
            size_t id = idx(i, j, extended_ny);
            ut[id]  = A - (B + 1) * u_n[id] + u_n[id] * u_n[id] * v_n[id] + Du * Lu[id];
            vt[id]  = B * u_n[id] - u_n[id] * u_n[id] * v_n[id] + Dv * Lv[id];
            u_n[id] += dt * ut[id];
            v_n[id] += dt * vt[id];
        }
    }

    return mpi_size;
}
