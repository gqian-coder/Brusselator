#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <dirent.h>
#include <random>
#include <algorithm>  // For std::min_element and std::max_element
#include <string.h>

#include "adios2.h"

// kinetic energy
template<typename Real>
double calc_KE(Real *u_prev, Real *u, double dt, size_t num_data)
{
    double diff_t, KE = 0.0;
    double dt2 = dt*dt;
    for (size_t i=0; i<num_data; i++) {
        diff_t = static_cast<double>(u[i] - u_prev[i]);
        KE += diff_t * diff_t / dt2 ;
    }
    return KE * 0.5;
}

// potential energy
template<typename Real>
double calc_PE(Real *u, double dh, size_t Nx, size_t Ny)
{
    double PE = 0.0;
    double ux, uy;
    size_t r_curr, k;
    for (size_t r=1; r<Nx; r++) {
        r_curr = r * Ny;
        for (size_t c=1; c<Ny; c++) {
            k  = r_curr + c;
            ux = static_cast<double>(u[k] - u[k-Ny]) / dh;
            uy = static_cast<double>(u[k] - u[k-1])/dh;
            PE += ux*ux + uy*uy;
        }
    }
    return PE * 0.5;
}

// root-of-mean-square error
template<typename Real>
double calc_rmse(double *data_f, Real *data_g, double *diff, size_t num_data)
{
    double rmse = 0.0;
    for (size_t i=0; i<num_data; i++) {
        diff[i] = data_f[i] - static_cast<double>(data_g[i]);
        rmse += (diff[i]*diff[i]); 
    }
    rmse = rmse / (double)num_data;
    return std::sqrt(rmse); 
}

int main(int argc, char **argv) {

    int cnt_argv = 1;
    std::string fname_f(argv[cnt_argv++]);
    std::string fname_g(argv[cnt_argv++]);
    // simulation spatial resolution
    double dt      = std::stof(argv[cnt_argv++]);
    double dh      = std::stof(argv[cnt_argv++]);
    // from the init_ts step in frame_f to compare against frame_g 
    size_t init_ts = std::stoi(argv[cnt_argv++]);    
    std::string fname_err(argv[cnt_argv++]);
    size_t total_Steps = std::stoi(argv[cnt_argv++]);
    bool total_Energy  = bool(std::stoi(argv[cnt_argv++]));

    std::cout << "original file: " << fname_f.c_str() << "\n";
    std::cout << "mgr file: " << fname_g.c_str() << "\n";

    std::cout << "output file: " << fname_err.c_str() << "\n"; 
    std::cout << ", dh = " << dh << ", init_ts = " << init_ts << "\n";

    adios2::ADIOS ad;
    adios2::IO reader_io_f = ad.DeclareIO("Original");
    adios2::IO reader_io_g = ad.DeclareIO("Lossy");
    reader_io_f.SetEngine("BP");
    reader_io_g.SetEngine("BP");
    adios2::Engine reader_f = reader_io_f.Open(fname_f, adios2::Mode::ReadRandomAccess);
    adios2::Engine reader_g = reader_io_g.Open(fname_g, adios2::Mode::ReadRandomAccess);

    adios2::Variable<double> variable_f_u = reader_io_f.InquireVariable<double>("u");
    adios2::Variable<float>  variable_g_u = reader_io_g.InquireVariable<float>("u");
    adios2::Variable<double> variable_f_v = reader_io_f.InquireVariable<double>("v");
    adios2::Variable<float>  variable_g_v = reader_io_g.InquireVariable<float>("v");
    size_t available_Steps = std::min(variable_g_u.Steps(), variable_f_u.Steps() - init_ts); 
    if (total_Steps>0) {
        total_Steps = std::min(total_Steps, available_Steps); 
    } else {
        total_Steps = available_Steps;
    }
    std::cout << "total number of steps: " << variable_f_u.Steps() << ", read from " << init_ts << " to " << total_Steps + init_ts << " timestep \n";
    reader_g.Close();
    reader_g = reader_io_g.Open(fname_g, adios2::Mode::Read);

    std::vector<std::size_t> shape = variable_f_u.Shape();
    size_t num_data = shape[0]*shape[1];
    std::cout << "data space: " << shape[0] << "x" << shape[1] << "\n";
    std::vector<double> var_f(num_data);
    std::vector<float> var_g(num_data);
    // difference data
    std::vector<double> var_e(num_data);
    std::vector<double> rmse_u(total_Steps);
    std::vector<double> rmse_v(total_Steps);
    std::vector<double> var_prevE_u(num_data);
    std::vector<double>  var_prevF_u(num_data);
    std::vector<double> var_prevE_v(num_data);
    std::vector<double>  var_prevF_v(num_data);
    std::vector<double> PE_e_u(total_Steps);
    std::vector<double> PE_f_u(total_Steps);
    std::vector<double> PE_e_v(total_Steps);
    std::vector<double> PE_f_v(total_Steps);
    std::vector<double> KE_f_u(total_Steps);
    std::vector<double> KE_e_u(total_Steps);
    std::vector<double> KE_f_v(total_Steps);
    std::vector<double> KE_e_v(total_Steps);

    size_t cnt = 0;
    while (true) {
        // Begin step
        adios2::StepStatus read_status = reader_g.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK) {
            break;
        }
        variable_f_u.SetStepSelection({init_ts+cnt, 1});
        reader_f.Get(variable_f_u, var_f);
        reader_f.PerformGets();
        variable_g_u = reader_io_g.InquireVariable<float>("u");
        reader_g.Get(variable_g_u, var_g);
        reader_g.PerformGets();
        rmse_u[cnt] = calc_rmse(var_f.data(), var_g.data(), var_e.data(), num_data);
        if (cnt>0) {
            KE_e_u[cnt] = calc_KE(var_prevE_u.data(), var_e.data(), dt, num_data);
            if (total_Energy) {
                KE_f_u[cnt] = calc_KE(var_prevF_u.data(), var_f.data(), dt, num_data);
            }
        }
        PE_e_u[cnt] = calc_PE(var_e.data(), dh, shape[0], shape[1]);
        if (total_Energy) {
            PE_f_u[cnt] = calc_PE(var_f.data(), dh, shape[0], shape[1]);
        }
        std::copy(var_e.begin(), var_e.end(), var_prevE_u.begin());
        std::copy(var_f.begin(), var_f.end(), var_prevF_u.begin());
        variable_f_v.SetStepSelection({init_ts+cnt, 1});
        reader_f.Get(variable_f_v, var_f);
        reader_f.PerformGets();
        variable_g_v = reader_io_g.InquireVariable<float>("v");
        reader_g.Get(variable_g_v, var_g);
        reader_g.PerformGets();
        rmse_v[cnt] = calc_rmse(var_f.data(), var_g.data(), var_e.data(), num_data);
        if (cnt>0) {
            KE_e_v[cnt] = calc_KE(var_prevE_v.data(), var_e.data(), dt, num_data);
            if (total_Energy) {
                KE_f_v[cnt] = calc_KE(var_prevF_v.data(), var_f.data(), dt, num_data);
            }
        }
        PE_e_v[cnt] = calc_PE(var_e.data(), dh, shape[0], shape[1]);
        if (total_Energy) {
            PE_f_v[cnt] = calc_PE(var_f.data(), dh, shape[0], shape[1]);
        }
        std::copy(var_e.begin(), var_e.end(), var_prevE_v.begin());
        std::copy(var_f.begin(), var_f.end(), var_prevF_v.begin());
        reader_g.EndStep(); 
        if (cnt % 5 == 0) std::cout << cnt << " / " << init_ts+cnt << ", var U: l2 = " << rmse_u[cnt] << ", relative PE = " << PE_e_u[cnt]/PE_f_u[cnt] << ", relative KE = " << KE_e_u[cnt]/KE_f_u[cnt] << "; var V: l2 = "<< rmse_v[cnt] << ", relative PE = " << PE_e_v[cnt]/PE_f_v[cnt] << ", relative KE = " << KE_e_v[cnt]/KE_f_v[cnt] <<  "\n"; 
        if (cnt == total_Steps-1) {
            std::cout << cnt << " / " << init_ts+cnt << ", var U: l2 = " << rmse_u[cnt] << ", relative PE = " << PE_e_u[cnt]/PE_f_u[cnt] << ", relative KE = " << KE_e_u[cnt]/KE_f_u[cnt] << "; var V: l2 = "<< rmse_v[cnt] << ", relative PE = " << PE_e_v[cnt]/PE_f_v[cnt] << ", relative KE = " << KE_e_v[cnt]/KE_f_v[cnt] <<  "\n";
            break;
        }
        cnt ++;
    }
    reader_f.Close();
    reader_g.Close();
    
    FILE *fp = fopen((fname_err+"_l2.bin").c_str(), "w");
    fwrite(rmse_u.data(), sizeof(double), total_Steps, fp);
    fwrite(rmse_v.data(), sizeof(double), total_Steps, fp);
    fclose(fp);

    fp = fopen((fname_err+"_PE_e.bin").c_str(), "w");
    fwrite(PE_e_u.data(), sizeof(double), total_Steps, fp);
    fwrite(PE_e_v.data(), sizeof(double), total_Steps, fp);
    fclose(fp);

    if (total_Energy) {
        fp = fopen((fname_err+"_PE_f.bin").c_str(), "w");
        fwrite(PE_f_u.data(), sizeof(double), total_Steps, fp);
        fwrite(PE_f_v.data(), sizeof(double), total_Steps, fp);
        fclose(fp);
    }

    fp = fopen((fname_err+"_KE_e.bin").c_str(), "w");
    fwrite(KE_e_u.data(), sizeof(double), total_Steps, fp);
    fwrite(KE_e_v.data(), sizeof(double), total_Steps, fp);
    fclose(fp);

    if (total_Energy) {
        fp = fopen((fname_err+"_KE_f.bin").c_str(), "w");
        fwrite(KE_f_u.data(), sizeof(double), total_Steps, fp);
        fwrite(KE_f_v.data(), sizeof(double), total_Steps, fp);
        fclose(fp);
    }

    size_t n_print = std::min((int)cnt, 1000);
    //for (size_t i=0; i<n_print; i++) {
    //    std::cout << "Step " << i << " ==> var U: l2 = " << rmse_u[i] << ", PE_e = " << PE_e_u[i] << ", PE_f = " << PE_f_u[i] << "\n";
        //std::cout << "      var V: l2 = " << rmse_v[i] << ", PE_e = " << PE_e_v[i] << ", PE_f = " << PE_f_v[i] << "\n";
    //}
    return 0;
}
