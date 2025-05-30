#ifndef FUNINIT_HPP
#define FUNINIT_HPP

template <typename Real>
Real src_Gaussian_pulse(Real freq, Real ts, Real A);

template <typename Real>
void fun_Gaussian_pulse(Real *u, Real freq, Real t0, Real A, size_t xsrc, size_t ysrc,
                    size_t zsrc, size_t Ny, size_t Nz);

template <typename Real>
void fun_cos_waves(Real *u, Real *v, field dField, Real A, Real freq);

template <typename Real>
void fun_rainDrop(Real *u, Real *v, field dField,
                  size_t NDx, size_t NDy, size_t NDz, Real *gauss_template);

template <typename Real>
void fun_MultiRainDrop(Real *u, Real *v, field dField, size_t NDx, size_t NDy, size_t NDz,
                       Real *gauss_template, size_t nDrops);

template<typename Real>
void generate_random_vector(Real *u, size_t size, Real min_val, Real max_val);

#include "funInit.tpp"
#endif
