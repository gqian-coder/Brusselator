#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "io.hpp"

// Helper to index 2D arrays stored in 1D
inline size_t idx_2d(size_t i, size_t j, size_t ny) { return i * ny + j; }

template<typename Real>
void generate_random_vector(Real *u, size_t size, Real min_val, Real max_val) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(min_val, max_val);

    for (size_t i = 0; i < size; ++i) {
        u[i] += distrib(gen);
    }
}

template <typename Real> 
Real src_Gaussian_pulse(Real freq, Real ts, Real A)  
{
    return (-2.0*A*ts * (freq*freq) * (std::exp(-(freq*freq) * (ts*ts)))) ;
}

// t0: nt * dt
template <typename Real>
void fun_Gaussian_pulse(Real *u, Real freq, Real t0, Real A, size_t xsrc, size_t ysrc, 
                    size_t zsrc, size_t Ny, size_t Nz)
{
    size_t k = xsrc * (Ny*Nz) + ysrc*Nz + zsrc;
    u[k] = src_Gaussian_pulse(freq, -t0, A); 
}


// an array of 2D plane wave starting from x=px 
template <typename Real>
void fun_cos_waves(Real *u, Real *v, field dField, Real A, Real freq)
{
    size_t buffer_L = dField.ghostZ_len * 2;
    for (size_t i=1; i<=dField.nx; i++) {
        for (size_t j=1; j<=dField.ny; j++) {
            size_t id = idx_2d(i, j, dField.ny+buffer_L);
            Real x = (dField.nx_start + i - dField.ghostZ_len) * freq;
            Real y = (dField.ny_start + j - dField.ghostZ_len) * freq;
            u[id] += A * cos(M_PI * x) * cos(M_PI * y); 
            v[id] += A * cos(2 * M_PI * x) * cos(2 * M_PI * y);
        }
    }
}

// drop_probability: Raindrop probability (with each time tick) and intensity.
// NDx, NDy: droplet's region
// one droplet at each time
// source global location 
template <typename Real>
void fun_rainDrop(Real *u, Real *v, field dField,  
                  size_t NDx, size_t NDy, size_t NDz,
                  Real *gauss_template)
{
    size_t cx = (size_t)(NDx/2);
    size_t cy = (size_t)(NDy/2);
    size_t cz = (size_t)(NDz/2);

    float random_x = 0.5;
    float random_y = 0.5;
    float random_z = 0.5;
    size_t x = static_cast<size_t>(random_x * dField.nx_full - cx);
    size_t y = static_cast<size_t>(random_y * dField.ny_full - cy);
    size_t z = (dField.nz_full>1) ? static_cast<size_t>(random_z * dField.nz_full - cz) : 0;
    //std::cout << "center of rain drop: " << x+cx << ", " << y+cy << ", " << z+cz << "\n";
    size_t buffer_L = dField.ghostZ_len * 2;
    size_t dim1     = (dField.nz>1) ? (dField.ny+buffer_L) * (dField.nz+buffer_L) : dField.ny+buffer_L; 
    size_t dimD1    = NDy * NDz;
    size_t offset_x, offset_y, k; 
    size_t local_x, local_y, local_z;
    for (size_t r=0; r<NDx; r++) {
        if ((r+x<dField.nx_start) || (r+x>=dField.nx_start+dField.nx)) {
            continue;
        } else {
            local_x = r+x+dField.ghostZ_len - dField.nx_start; 
            offset_x = local_x*dim1;
            for (size_t c=0; c<NDy; c++) {
                if ((c+y<dField.ny_start) || (c+y>=dField.ny_start+dField.ny)) {
                    continue;
                } else {
                    //if ((dField.ny_start==400) && (c==200)) std::cout << "edge assign values @ " << c+y << "\n";
                    local_y = c+y+dField.ghostZ_len-dField.ny_start;
                    offset_y = (dField.nz>1) ? local_y*(dField.nz+buffer_L) : local_y;
                    for (size_t h=0; h<NDz; h++) {
                        if ((dField.nz>1) && ((h+z<dField.nz_start) || (h+z>=dField.nz_start+dField.nz))) {
                            continue;
                        } else {
                            local_z = (dField.nz>1) ? (h+z+dField.ghostZ_len-dField.nz_start) : 0;
                            k = offset_x + offset_y + local_z;
                            u[k] += gauss_template[r*dimD1+c*NDz+h];
                            v[k] += gauss_template[r*dimD1+c*NDz+h];
                        }
                    }
                }      
            }
        }
    }
}

// drop_probability: Raindrop probability (with each time tick) and intensity.
// drop multiple droplets
template <typename Real>
void fun_MultiRainDrop(Real *u, Real *v, field dField, size_t NDx, size_t NDy, size_t NDz,
                       Real *gauss_template, size_t nDrops)
{
    size_t cx = (size_t)(NDx/2);
    size_t cy = (size_t)(NDy/2);
    size_t cz = (size_t)(NDz/2);
    size_t buffer_L = dField.ghostZ_len * 2;
    //std::cout << "probability: " << random_number << "\n";
    // try not to generate rain drops closing to edges
    for (size_t d=0; d<nDrops; d++) {
        std::vector<float> px  = {0.275, 0.65, 0.35, 0.7};
        std::vector<float> py  = {0.35, 0.3, 0.725, 0.65};
        std::vector<float> pz  = {0.34, 0.56, 0.75, 0.25};
        std::vector<float> mag = {0.7, 0.5, 0.85, 0.6};
        //std::vector<float> px  = {0.375, 0.65, 0.45, 0.63};
        //std::vector<float> py  = {0.45, 0.4, 0.725, 0.65};
        //std::vector<float> pz  = {0.44, 0.56, 0.75, 0.35};
        //std::vector<float> mag = {0.6, 0.5, 0.65, 0.6};
        float random_x = px[d];
        float random_y = py[d];
        float random_z = pz[d];
        size_t x = static_cast<size_t>(random_x * dField.nx_full-cx);
        size_t y = static_cast<size_t>(random_y * dField.ny_full-cy);
        size_t z = static_cast<size_t>(random_z * dField.nz_full-cz);
        //std::cout << "x, y, z = " << x << ", " << y  << ", "<< z << ", " << cx << ", " << cy << ", " << cz << ", " << NDx << ", " << NDy << ", " << NDz << ", " << dField.nx << ", " << dField.ny << ", " << dField.nz << "\n";
        size_t dim1  = (dField.nz>1) ? (dField.ny+buffer_L) * (dField.nz+buffer_L) : (dField.ny+buffer_L);
        size_t dimD1 = NDy * NDz;
        size_t offset_x, offset_y, k;
        size_t local_x, local_y, local_z;
        float intensity = mag[d];// dis(gen);
        for (size_t r=0; r<NDx; r++) {
            if ((r+x<dField.nx_start) || (r+x>=dField.nx_start+dField.nx)) {
                continue;
            } else {
                local_x = r+x+dField.ghostZ_len - dField.nx_start;
                offset_x = local_x*dim1;
                for (size_t c=0; c<NDy; c++) {
                    if ((c+y<dField.ny_start) || (c+y>=dField.ny_start+dField.ny)) {
                        continue;
                    } else {
                        local_y = c+y+dField.ghostZ_len-dField.ny_start;
                        offset_y = (dField.nz>1) ? local_y*dField.nz : local_y;
                        for (size_t h=0; h<NDz; h++) {
                            if ((dField.nz>1) && ((h+z<dField.nz_start) || (h+z>=dField.nz_start+dField.nz))) {
                                continue;
                            } else {
                                local_z = (dField.nz>1) ? (h+z+dField.ghostZ_len-dField.nz_start) : 0;
                                k = offset_x + offset_y + local_z;
                                u[k] += intensity*gauss_template[r*dimD1+c*NDz+h];
                                v[k] += intensity*gauss_template[r*dimD1+c*NDz+h];
                            }
                        }
                    }
                }
            }
        }
    }
}

