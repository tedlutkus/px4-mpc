#ifndef WORKSPACE_H
#define WORKSPACE_H

/*
 * This file was autogenerated by OSQP-Python on November 05, 2024 at 14:06:33.
 * 
 * This file contains the prototypes for all the workspace variables needed
 * by OSQP. The actual data is contained inside workspace.c.
 */

#include "types.h"
#include "qdldl_interface.h"

// Data structure prototypes
extern csc Pdata;
extern csc Adata;
extern c_float qdata[366];
extern c_float ldata[406];
extern c_float udata[406];
extern OSQPData data;

// Settings structure prototype
extern OSQPSettings settings;

// Scaling structure prototypes
extern c_float Dscaling[366];
extern c_float Dinvscaling[366];
extern c_float Escaling[406];
extern c_float Einvscaling[406];
extern OSQPScaling scaling;

// Prototypes for linsys_solver structure
extern csc linsys_solver_L;
extern c_float linsys_solver_Dinv[772];
extern c_int linsys_solver_P[772];
extern c_float linsys_solver_bp[772];
extern c_float linsys_solver_sol[772];
extern c_float linsys_solver_rho_inv_vec[406];
extern c_int linsys_solver_Pdiag_idx[183];
extern csc linsys_solver_KKT;
extern c_int linsys_solver_PtoKKT[183];
extern c_int linsys_solver_AtoKKT[2799];
extern c_int linsys_solver_rhotoKKT[406];
extern QDLDL_float linsys_solver_D[772];
extern QDLDL_int linsys_solver_etree[772];
extern QDLDL_int linsys_solver_Lnz[772];
extern QDLDL_int   linsys_solver_iwork[2316];
extern QDLDL_bool  linsys_solver_bwork[772];
extern QDLDL_float linsys_solver_fwork[772];
extern qdldl_solver linsys_solver;

// Prototypes for solution
extern c_float xsolution[366];
extern c_float ysolution[406];

extern OSQPSolution solution;

// Prototype for info structure
extern OSQPInfo info;

// Prototypes for the workspace
extern c_float work_rho_vec[406];
extern c_float work_rho_inv_vec[406];
extern c_int work_constr_type[406];
extern c_float work_x[366];
extern c_float work_y[406];
extern c_float work_z[406];
extern c_float work_xz_tilde[772];
extern c_float work_x_prev[366];
extern c_float work_z_prev[406];
extern c_float work_Ax[406];
extern c_float work_Px[366];
extern c_float work_Aty[366];
extern c_float work_delta_y[406];
extern c_float work_Atdelta_y[366];
extern c_float work_delta_x[366];
extern c_float work_Pdelta_x[366];
extern c_float work_Adelta_x[406];
extern c_float work_D_temp[366];
extern c_float work_D_temp_A[366];
extern c_float work_E_temp[406];

extern OSQPWorkspace workspace;

#endif // ifndef WORKSPACE_H