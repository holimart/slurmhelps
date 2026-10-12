# shellcheck shell=bash
# Site toolchain profile: MFF Troja (Rocky 9) Fortran + MPI + math stack.
# Source of truth: slurmhelps env/toolchain-mffsrv.sh; sourced by env/env.sh
# through SH_TOOLCHAIN_PROFILE (or the $WORK_ROOT/config/toolchain-mffsrv.sh shim).
# Replicates `module load openmpi-5.0.8/tcp` + `oneapi/mkl/2023.2.0`.
export MPI_ROOT="/usr/local/openmpi-5.0.8"
case ":$PATH:" in *":$MPI_ROOT/bin:"*) ;; *) export PATH="$MPI_ROOT/bin:$PATH" ;; esac
export LD_LIBRARY_PATH="$MPI_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$MPI_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$MPI_ROOT/include${CPATH:+:$CPATH}"
# OpenMPI 5 selects the UCX PML by default; UCX's ucp_worker_iface_open raises
# a floating-point flag during MPI_Init, and any build with -ffpe-trap turns
# that into SIGFPE before the solver starts. Force the ob1 PML -- exactly what
# the cluster's `openmpi-5.0.8/tcp` module does -- so UCX is never loaded.
export OMPI_MCA_pml="ob1"
export OMPI_MCA_coll="self,basic,libnbc"
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_btl_tcp_if_exclude="ib0,lo"
export MKLROOT="/opt/intel/oneapi/mkl/2023.2.0"
export LD_LIBRARY_PATH="$MKLROOT/lib/intel64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$MKLROOT/lib/intel64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$MKLROOT/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${WORK_ROOT:?}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export FC="gfortran"
# OpenMPI 5/PRTE refuses a Lustre TMPDIR -> force node-local /tmp.
export TMPDIR="/tmp"
export OMPI_MCA_prte_tmpdir_base="/tmp"
