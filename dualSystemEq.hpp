#ifndef DUALSYSTEMEQ_HPP 
#define DUALSYSTEMEQ_HPP 

template <typename Real> class dualSystemEquation {
public:
    //! Constructor.
    //!
    //!\param Nx and Ny simulation domain area
    //!\param dt and dh time and space grid spacings 
    //!\param use_condition Simulation boundary condiction
    dualSystemEquation(field data_field, Real dt, Real dh,
                       Real Du, Real Dv, Real A, Real B);
                  

    //! Update the simulation by one time tick.
    void rk4_step_2d(parallel_data<Real> parallel);
    void rk4_step_3d(parallel_data<Real> parallel);
    void compute_laplacian(const std::vector<Real>& grid, std::vector<Real>& lap, size_t nx, size_t ny, Real dh);
    void exchange_ghost_cells(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Datatype datatype, MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right);
    void exchange_ghost_cells_mgr(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right, Real tol, Real s);
    
    // initial current ts data values
    void init_u(Real *data);
    void init_v(Real *data);

    //! Return the size in bytes of the data
    std::size_t size() const;

    //! Vector array u_{i,j}^{n}
    std::vector<Real> u_n;
    //! Vector array v_{i,j}^{n}
    std::vector<Real> v_n;
    // data domain parameters (global vs local)
    field dField;

private:
    Real dt;
    Real dh;
    size_t data_size;
    // Diffusion term
    double Du;
    double Dv;
    double A;
    double B;
};

#include "dualSystemEq.tpp"
#endif
