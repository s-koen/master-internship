!------- for use during AGB evolution ---------------------

! fixed quantities

! integer :: evol_phase
  ! x_integer_ctrl(1) = evol phase

  ! x_logical_ctrl(2) = save hdf5 files

! variable quantities- ENSURE THERE IS NO CONFLICT WITH OTHER USAGE OF THESE PARAMETERS

! xtra quantities
integer, parameter :: x_max_LHe = 1
integer, parameter :: x_he_core_mass = 2
integer, parameter :: x_interpulse_core_mass_growth = 3
integer, parameter :: x_min_core_mass_after_TDU = 4
integer, parameter :: x_max_core_mass_before_TDU = 5
! ixtra quantities
integer, parameter :: ix_TP_count = 1
! lxtra quantities
! integer, parameter :: lx_keep_hydro_off = 1
! integer, parameter :: lx_hydro_on = 2
integer, parameter :: lx_in_LHe_peak = 3


character(len=20) :: envelope_overshooting_scheme
real(dp) :: VCT
real(dp) :: MXP
! real(dp) :: overshoot_1_f
! real(dp) :: overshoot_1_f0
! real(dp) :: overshoot_2_f
! real(dp) :: overshoot_2_f0
! real(dp) :: surface_alpha_mlt
! real(dp) :: alpha_e = 2.5
real(dp) :: logP
real(dp) :: conv_env_bot
integer :: num_extra_history_columns_eagb = 4
integer :: num_extra_history_columns_tpagb = 17
integer :: num_extra_profile_columns_tpagb = 3
integer :: num_extra_history_columns_post_agb = 3
real(dp), dimension(200) :: frac_HI_old

! ----------------------------------------------------------------
