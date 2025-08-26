// Parallel 2D Brusselator solver using MPI
// Domain is partitioned in 2D grid across MPI processes

#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

#include <adios2.h>
#include "mgard/compress_x.hpp"
#include "io.hpp"
#include "funInit.hpp"
#include "dualSystemEq.hpp"

using namespace std;

// Brusselator parameters
// Turing spots
//const double A = 0.5;
//const double B = 3.0;
// Turing strips
const double A = 5;
const double B = 10;
double Du = 1.0;
double Dv = 9.0;

/* suggested inputs */
// Lx  = 10.0, Ly = 10.0, dh = 1e-4, dt = 5e-4, T=2,  wt_interval = 100

// Helper to index 2D arrays stored in 1D
inline int idx(int i, int j, int ny) { return i * ny + j; }

void copy_internal_data(double *copy_buff, double *data_buff, size_t nx, size_t ny, size_t ghostZ)
{
    size_t offset, offset_full;
    size_t extended_ny = ghostZ*2 + ny;
    for (size_t i=ghostZ; i<nx+ghostZ; i++) {
        offset      = (i-ghostZ) * ny;
        offset_full = i*extended_ny+ghostZ;
        memcpy(&copy_buff[offset], &data_buff[offset_full], sizeof(double)*ny);
    }
}


// only support zm_factor in the format of integer
// zm_factor: > 1: zm_var is the zoomed-in version of in_var; vice verse 
// nx, ny: the dimension of zm_var
void repeatCopy(double *in_var, double *zm_var, double zm_factor, size_t nx, size_t ny)
{
    size_t ny_in, offset_zm, offset_in;
    ny_in = static_cast<size_t>(static_cast<double>(ny) / zm_factor); 
    for (size_t i=0; i<nx; i++) {
        size_t i_in = static_cast<size_t>(static_cast<double>(i) / zm_factor);
        offset_in   = i_in * ny_in;
        offset_zm   = i    * ny;
        for (size_t j=0; j<ny; j++) {
            size_t j_in = static_cast<size_t>(static_cast<double>(j) / zm_factor);    
            zm_var[offset_zm + j] = in_var[offset_in + j_in];
        }
    }
}


int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> dims = {0, 0};
    MPI_Dims_create(size, 2, dims.data());
    int periods[2] = {1, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods, 0, &cart_comm);

    std::vector<int> coords(2);
    MPI_Cart_coords(cart_comm, rank, 2, coords.data());
    if (rank==0) {
        std::cout << "Dims {" << dims[0] << ", " << dims[1] << "}\n";
        std::cout << "Periods: {" << periods[0] << ", " << periods[1] << "}\n";
    }

    int cnt_argv = 1;
    // compress checkpoint data (timestep output compressed)
    int compression_cpt = std::stoi(argv[cnt_argv++]);
    // when checkpoint compression is on, compression_mpi will be off 
    int compression_mpi = std::stoi(argv[cnt_argv++]);
    if (compression_cpt) {
        compression_mpi = false;
    }
    // tolerance used for either checkpoint or mpi compression
    double tol_u     = std::stof(argv[cnt_argv++]);
    double tol_v     = std::stof(argv[cnt_argv++]);
    double snorm = std::stof(argv[cnt_argv++]);   

    parallel_data<double> parallelization; //!< Parallelization info
    initialize_mpi_bound<double>(parallelization, 2, rank, size, cart_comm, compression_mpi, tol_u, tol_v, snorm);
    std::cout << "Rank " << rank << " coordinates: {" << coords[0] << ", " << coords[1] << "}, up = " << parallelization.up << ", down = " << parallelization.down << ", left = " << parallelization.left << ", right = " << parallelization.right<< ", ";
    std::cout << "left up = " << parallelization.left_up << ", left down = " << parallelization.left_down << ", right up = " << parallelization.right_up << ", right down = " << parallelization.right_down << "\n";

    int init_fun     = std::stoi(argv[cnt_argv++]);
    size_t init_ts   = 0;
    double zm_factor = 1.0;
    if (init_fun==0) {
        init_ts = std::stoi(argv[cnt_argv++]);
    } else if (init_fun==5) {
        init_ts   = std::stoi(argv[cnt_argv++]);
        // zm_factor > 1: load checkpoint and make it zm_factor times larger through repeat copy
        // zm_factor < 1: skip copy
        zm_factor = std::stof(argv[cnt_argv++]); 
    }
    double Lx = std::stof(argv[cnt_argv++]);
    double Ly = std::stof(argv[cnt_argv++]);
    double dh = std::stof(argv[cnt_argv++]);
    double dt = std::stof(argv[cnt_argv++]);
    double T  = std::stof(argv[cnt_argv++]);
    size_t wt_interval = (size_t) std::stof(argv[cnt_argv++]);
    std::string filename = argv[cnt_argv++];
    size_t steps = (size_t)(T/dt);
    size_t Nx = (size_t)std::ceil((double)Lx / dh) ;
    size_t Ny = (size_t)std::ceil((double)Ly / dh) ;
    if (rank==0) {
        std::cout << "Checkpoint compression flag " << (int)compression_cpt << ", MPI compression flag " << (int)compression_mpi << ", eb = " << tol_u << ", snorm = " << snorm << "\n"; 
        std::cout << "init fun = " << init_fun << ", " << "Lx = " << Lx << ", Ly = " << Ly << ", dh = " << dh << ", dt = " << dt <<  ", T = " << T << ", total steps = " << steps <<  ", wt_interval = " << wt_interval << ", total output steps = " << steps / wt_interval << ", Nx = " << Nx << ", " << Ny << "\n";
    }

    // Set Dv to generate Turing instability
    double D_thresh = A*A / (std::sqrt(B)-1) / (std::sqrt(B)-1) * Du; 
    //Dv = D_thresh * 0.9;
    // Check for CFL 
    double dt_cfl = std::min(dh*dh/4.0/ Du, dh*dh/4.0/Dv);
    if (rank==0) std::cout << "CFL required dt " << dt_cfl << "\n";
    if (dt >= dt_cfl) {
        std::cout << "CFL condition violated\n";    
        MPI_Finalize();
        return -1;
    }

    field fieldData;
    // using extended ghost zone for optimized compression 
    size_t ghostZ_len = 4; 
    initialize_mpi_dataField(fieldData, Nx, Ny, 1, ghostZ_len, dims.data(), coords.data());
    dualSystemEquation<double> dualSys(fieldData, dt, dh, ghostZ_len, Du, Dv, A, B);
    if (rank==0) {
        std::cout << "local data domain = {" << fieldData.nx << ", " << fieldData.ny << "}\n";
        std::cout << "A = " << A << ", B = " << B << ", Du = " << Du << ", Dv = " << Dv << "\n";
        if (Dv > D_thresh) std::cout << "Turing instability occurred: Dv = " << Dv << "\n";
    }

    size_t data_size = (fieldData.nx+2) * (fieldData.ny+2);
    std::vector<double> gauss_template;
    double NDx, NDy;
    double max_intensity = 1.0;

    // Initialize ADIOS2
    adios2::ADIOS adios(MPI_COMM_WORLD);
    adios2::IO io = adios.DeclareIO("Brusselator");
    std::vector<std::size_t> shape = {(std::size_t)Nx, (std::size_t)Ny};
    std::vector<std::size_t> start = {(std::size_t)(fieldData.nx_start), (std::size_t)(fieldData.ny_start)};
    std::vector<std::size_t> count = {(std::size_t)fieldData.nx, (std::size_t)fieldData.ny};
    
    auto var_u = io.DefineVariable<double>("u", shape, start, count);
    auto var_v = io.DefineVariable<double>("v", shape, start, count);
    //std::cout << "shape = {" << shape[0] << ", " << shape[1] << "}, start = {" << start[0] << ", " << start[1] << "}, count = {" << count[0] << ", " << count[1] << "}\n";

    if ((init_fun==1) || (init_fun==2)) {
        // Width of the Gaussian profile for each initial drop.
        double drop_width = fieldData.nx_full/3.0*dh;
        // Size of the Gaussian template each drop is based on.
        NDx = (size_t) std::ceil(drop_width / dh);
        NDy = (size_t) std::ceil(drop_width / dh);
        size_t cx = (size_t)(NDx/2);
        size_t cy = (size_t)(NDy/2);
        gauss_template.resize(NDx*NDy, 0);
        std::vector <double> px(NDx), py(NDy);
        for (size_t r=0; r<NDx; r++) {
            px[r] = ((double)r - cx)/NDx*16;
        }
        for (size_t c=0; c<NDy; c++) {
            py[c] = (double(c)-cy)/NDy*16;
        }
        for (size_t r=0; r<NDx; r++) {
            for (size_t c=0; c<NDy; c++) {
                if ((r-cx)*(r-cx)+(c-cy)*(c-cy) < 0.25*NDx*NDx) {
                    gauss_template[r*NDy+c] = max_intensity * exp(-(px[r]*px[r] + py[c]*py[c])/100.0);
                }
            }
        }
    }

    // domain value initialization
    switch (init_fun) {
        case 0: {
            adios2::IO reader_io = adios.DeclareIO("Input");
            reader_io.SetEngine("BP");
            adios2::Engine reader = reader_io.Open(filename, adios2::Mode::ReadRandomAccess);
            adios2::Variable<double> variable_u, variable_v;
            variable_u = reader_io.InquireVariable<double>("u");
            variable_v = reader_io.InquireVariable<double>("v");
            if (rank==0) {
                std::cout << "total number of steps: " << variable_u.Steps() << ", read from " << init_ts << " timestep \n";
                std::cout << "Initialize the simulation from a previously saved checkpoint file\n";
            }
            std::vector<double> in_var(fieldData.nx * fieldData.ny);
            variable_u.SetSelection({start, count}); 
            variable_u.SetStepSelection({init_ts, 1});
            variable_v.SetSelection({start, count});
            variable_v.SetStepSelection({init_ts, 1});
            reader.Get(variable_u, in_var.data());
            reader.PerformGets();
            dualSys.init_u_2d(in_var.data());
            reader.Get(variable_v, in_var.data());
            reader.PerformGets();
            dualSys.init_v_2d(in_var.data());
            reader.Close();
            in_var.clear();
            filename.erase(filename.size() - 3);
            filename.append("-cr.bp");
            break;
        }
        case 1: {
            std::cout << "Rain drop of width " << NDx <<  " in rank " << rank << " (" << coords[0] << ", " << coords[1] << ")\n";
            auto [min_it, max_it] = std::minmax_element(gauss_template.begin(), gauss_template.end());
            std::cout << "min and max of gaussian source: " << *min_it << ", " << *max_it << "\n";
            fun_rainDrop<double>(dualSys.u_n.data(), dualSys.v_n.data(), dualSys.dField, NDx, NDy, 1, gauss_template.data());
            break;
        }    
        case 2: {
            size_t n_drops = 4;
            if (rank==0) {
                std::cout << n_drops << " rain drops of width " << NDx << " in rank " << rank << " (" << coords[0] << ", " << coords[1] << ")\n";
                auto [min_it, max_it] = std::minmax_element(gauss_template.begin(), gauss_template.end());
                std::cout << "min and max of gaussian source: " << *min_it << ", " << *max_it << "\n";
            }
            fun_MultiRainDrop<double>(dualSys.u_n.data(), dualSys.v_n.data(), dualSys.dField, NDx, NDy, 1, gauss_template.data(), n_drops);
            break;
        }
        case 3: {
            double freq = 15.0 * dh;
            double magn = 1;
            fun_cos_waves<double>(dualSys.u_n.data(), dualSys.v_n.data(), dualSys.dField, magn, freq); 
            break;
        }
        case 4: {
            double minv = 0, maxv = 0.01;
            generate_random_vector<double>(dualSys.u_n.data(), data_size, minv, maxv);
            generate_random_vector<double>(dualSys.v_n.data(), data_size, minv, maxv);
            break;
        }
        case 5: {
            adios2::IO reader_io = adios.DeclareIO("Input");
            reader_io.SetEngine("BP");
            adios2::Engine reader = reader_io.Open(filename, adios2::Mode::ReadRandomAccess);
            adios2::Variable<double> variable_u, variable_v;
            variable_u = reader_io.InquireVariable<double>("u");
            variable_v = reader_io.InquireVariable<double>("v");
            if (rank==0) {
                std::cout << " Load checkpoint and create a zoomed in/out shadow system\n";
                std::cout << "total number of steps: " << variable_u.Steps() << ", read from " << init_ts << " timestep \n";
                std::cout << "Initialize the simulation from a previously saved checkpoint file\n";
            }
            size_t readin_sz = (size_t) ((double)(fieldData.nx * fieldData.ny) / zm_factor / zm_factor);
            std::vector<double> in_var(readin_sz);
            std::vector<double> zm_var(fieldData.nx * fieldData.ny);
            std::vector<std::size_t> zm_start = {(std::size_t)(static_cast<double>(fieldData.nx_start) / zm_factor), (std::size_t)(static_cast<double>(fieldData.ny_start) / zm_factor)};
            std::vector<std::size_t> zm_count = {(std::size_t)(static_cast<double>(fieldData.nx) / zm_factor), (std::size_t)(static_cast<double>(fieldData.ny) / zm_factor)}; 
            
            variable_u.SetSelection({zm_start, zm_count});
            variable_u.SetStepSelection({init_ts, 1});
            variable_v.SetSelection({zm_start, zm_count});
            variable_v.SetStepSelection({init_ts, 1});

            reader.Get(variable_u, in_var.data());
            reader.PerformGets();

            repeatCopy(in_var.data(), zm_var.data(), zm_factor, fieldData.nx, fieldData.ny);
            dualSys.init_u_2d(zm_var.data());

            reader.Get(variable_v, in_var.data());
            reader.PerformGets();

            repeatCopy(in_var.data(), zm_var.data(), zm_factor, fieldData.nx, fieldData.ny); 
            dualSys.init_v_2d(zm_var.data());
            
            reader.Close();
            in_var.clear();
            zm_var.clear();
            filename.erase(filename.size() - 3);
            filename.append("-shadow-cr.bp");
            
            break;
        }
        default:
            break;
    }
    // fill up the ghost zone with neighboring data values

    adios2::Engine writer = io.Open(filename, adios2::Mode::Write);

    std::vector<double> internal_u(fieldData.nx*fieldData.ny);
    std::vector<double> internal_v(fieldData.nx*fieldData.ny);
    size_t mpi_size_rk = 0, mpi_size_total = 0;

    //bool mid_rank = ((coords[0]==(int)std::floor(dims[0]/2)) && (coords[1]==(int)std::floor(dims[1]/2)));
    void *compressed_u = NULL, *compressed_v = NULL;
    std::vector<mgard_x::SIZE> mgard_shape{fieldData.nx, fieldData.ny};
    size_t compressed_size_u = 0, compressed_size_v = 0, compressed_total_u, compressed_total_v, temp_cp_size;
    // compression parameters
    mgard_x::Config config;
    size_t fieldSz = fieldData.nx*fieldData.ny*sizeof(double);
    if (compression_cpt) {
        compressed_u = (void *)malloc(sizeof(double) * fieldData.nx*fieldData.ny);
        compressed_v = (void *)malloc(sizeof(double) * fieldData.nx*fieldData.ny);
        config.lossless = mgard_x::lossless_type::Huffman_Zstd;
        config.normalize_coordinates = true;
        config.dev_type = mgard_x::device_type::CUDA;
    }
    for (size_t t = 0; t <= steps; ++t) {
        if (t % wt_interval == 0) {
            writer.BeginStep();
            copy_internal_data(internal_u.data(), dualSys.u_n.data(), fieldData.nx, fieldData.ny, ghostZ_len);
            if (compression_cpt) {
                temp_cp_size = fieldSz;
                mgard_x::compress(2, mgard_x::data_type::Double, mgard_shape, tol_u, snorm,
                        mgard_x::error_bound_type::ABS, internal_u.data(),
                        compressed_u, temp_cp_size, config, true);
                compressed_size_u += temp_cp_size;
                // write out decompressed data to checkpoint file (so restart does not need to decompress)
                void *decompress_pt = static_cast<void*>(internal_u.data());
                mgard_x::decompress(compressed_u, temp_cp_size, decompress_pt, config, true);
            } 
            writer.Put(var_u, internal_u.data(), adios2::Mode::Sync);
            
            copy_internal_data(internal_v.data(), dualSys.v_n.data(), fieldData.nx, fieldData.ny, ghostZ_len);
            if (compression_cpt) {
                temp_cp_size = fieldSz;
                mgard_x::compress(2, mgard_x::data_type::Double, mgard_shape, tol_v, snorm,
                        mgard_x::error_bound_type::ABS, internal_v.data(),
                        compressed_v, temp_cp_size, config, true);
                compressed_size_v += temp_cp_size;
                void *decompress_pt = static_cast<void*>(internal_v.data());
                mgard_x::decompress(compressed_v, temp_cp_size, decompress_pt, config, true);
            }
            writer.Put(var_v, internal_v.data(), adios2::Mode::Sync);
            
            writer.PerformPuts();
            writer.EndStep(); 
            if (rank == 0) std::cout << "Step " << t << " written to ADIOS2." << std::endl;
        }
        mpi_size_rk += dualSys.rk4_step_2d_extendedGhostZ(parallelization);
       // mpi_size_rk += dualSys.Euler_step_2d_extendedGhostZ(parallelization);
    }
    if (compression_mpi) {
        MPI_Reduce(&mpi_size_rk, &mpi_size_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    } 
    if (compression_cpt) {
        MPI_Reduce(&compressed_size_u, &compressed_total_u, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&compressed_size_v, &compressed_total_v, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    std::cout << "Rank " << rank << ": s = " << snorm << ", MPI message CR = " << (double)steps * (double)(fieldData.nx * fieldData.ny) * sizeof(double) * 2 / (double)mpi_size_rk << "\n"; 
    if (rank == 0) {
        if (compression_mpi) {
            std::cout << "Total compressed MPI message size = " << mpi_size_total << ", CR = " << (double)steps * (double)(fieldData.nx * fieldData.ny) * size * sizeof(double) * 2 / (double)mpi_size_total <<"\n"; 
        }
        if (compression_cpt){
            std::cout << "Total compressed checkpoint CR(u) = " << (double)steps * (double)(fieldData.nx * fieldData.ny) * size * sizeof(double) / (double)compressed_total_u << ", CR(v) = " << (double)steps * (double)(fieldData.nx * fieldData.ny) * size * sizeof(double) / (double)compressed_total_v << "\n"; 
        }
    }
 
    writer.Close();
    if (compression_cpt) {
        free(compressed_u);
        free(compressed_v);
    }
    MPI_Finalize();
    return 0;
}


