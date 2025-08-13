#ifndef DUALSYSTEMEQ_HPP 
#define DUALSYSTEMEQ_HPP 

template <typename Real> class dualSystemEquation {
public:
    //! Constructor.
    //!
    //!\param Nx and Ny simulation domain area
    //!\param dt and dh time and space grid spacings 
    //!\param use_condition Simulation boundary condiction
    dualSystemEquation(field data_field, Real dt, Real dh, size_t ghostZ_len, 
                        Real Du, Real Dv, Real A, Real B); 
                  

    //! Update the simulation by one time tick.
    size_t rk4_step_2d(parallel_data<Real> parallel);
    void rk4_step_3d(parallel_data<Real> parallel);
    void compute_laplacian(const std::vector<Real>& grid, std::vector<Real>& lap, size_t local_nx, size_t local_ny, size_t ny, size_t init_pos, Real dh);
    // float16
    void compute_laplacian_float16(const std::vector<Real>& grid, std::vector<Real>& lap, size_t local_nx, size_t local_ny, size_t ny, size_t init_pos, Real dh);
    // non-compression MPI ghost zone exchange
    void exchange_ghost_cells(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Datatype datatype, MPI_Comm cart_comm, size_t up, size_t down, 
                          size_t left, size_t right);
    // MGARD-compression for MPI
    size_t exchange_ghost_cells_mgr(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right, 
                          Real tol, Real s);
    // SZ-compression for MPI
    size_t exchange_ghost_cells_SZ(std::vector<Real>& grid, size_t local_nx, size_t local_ny, size_t ny,
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right, Real tol);
    
    // RK4 update using extended ghost zone -- no MPI communication for intermediate steps 
    size_t rk4_step_2d_extendedGhostZ(parallel_data<Real> parallel); 

    // mixed precision version of the above  
    size_t rk4_step_2d_extendedGhostZ_mixPrec(parallel_data<Real> parallel);

    // Euler method for Brusselator time integration 
    size_t Euler_step_2d_extendedGhostZ(parallel_data<Real> parallel);
    
    // MGARD-compression with extended ghost zone for MPI
    size_t exchange_ghost_extended_mgr(std::vector<Real>& grid, size_t nx, size_t ny, 
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right,
                          size_t left_up, size_t left_down, size_t right_up, size_t right_down,
                          Real tol, Real s);

    size_t exchange_ghost_extended(std::vector<Real>& grid, size_t nx, size_t ny, MPI_Datatype datatype, 
                          MPI_Comm cart_comm, size_t up, size_t down, size_t left, size_t right,
                          size_t left_up, size_t left_down, size_t right_up, size_t right_down);

    // initial current ts data values
    void init_u_2d(Real *data);
    void init_v_2d(Real *data);

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
    size_t ghostZ_len;
    size_t data_size;
    // Diffusion term
    Real Du;
    Real Dv;
    Real A;
    Real B;
};

#include "dualSystemEq.tpp"
#endif
