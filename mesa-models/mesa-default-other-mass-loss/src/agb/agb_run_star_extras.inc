!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Standard run_star_extra routines for TP-AGB modelling
      ! Routines explained in Rees et al. 2024 and Rees & Izzard 2024
      ! evol_phase = s% x_integer_ctrl(1)
      ! evol_phase = 1 for MS
      ! evol_phase = 2 for GB
      ! evol_phase = 3 for CHeB
      ! evol_phase = 4 for EAGB
      ! evol_phase = 5 for TPAGB
      ! evol_phase = 6 for simplified TPAGB
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   subroutine extras_controls(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         ! this is the place to set any procedure pointers you want to change
         ! e.g., other_wind, other_mixing, other_energy  (see star_data.inc)


         ! the extras functions in this file will not be called
         ! unless you set their function pointers as done below.
         ! otherwise we use a null_ version which does nothing (except warn).

         s% extras_startup => extras_startup
         s% extras_start_step => extras_start_step
         s% extras_check_model => extras_check_model
         s% extras_finish_step => extras_finish_step
         s% extras_after_evolve => extras_after_evolve
         s% how_many_extra_history_columns => how_many_extra_history_columns
         s% data_for_extra_history_columns => data_for_extra_history_columns
         s% how_many_extra_profile_columns => how_many_extra_profile_columns
         s% data_for_extra_profile_columns => data_for_extra_profile_columns

         s% how_many_extra_history_header_items => how_many_extra_history_header_items
         s% data_for_extra_history_header_items => data_for_extra_history_header_items
         s% how_many_extra_profile_header_items => how_many_extra_profile_header_items
         s% data_for_extra_profile_header_items => data_for_extra_profile_header_items
         
         evol_phase = s% x_integer_ctrl(1)
         select case(evol_phase)
         
         case(1)
         ! convective penetration
         s% other_adjust_mlt_gradT_fraction => adjust_mlt_gradT_fraction_conv_pen

         case(4,5)
         s% other_wind => vassilliadis_wood_wind
         s% other_alpha_mlt => other_alpha_mlt_prevent_FeI
         s% other_eps_grav => other_eps_grav_prevent_HRI
         s% other_timestep_limit => other_timestep_limit_resolve_TDU

         ! simplified model
         case(6)
         s% other_wind => vassilliadis_wood_wind


 !--------- for double exponential overshoot (s-process) -----------------------
         ! include 'overshoot_dbl_exp/overshoot_dbl_exp_extras_controls.inc'
 !-------------------------------------------------------------------------------
         
         end select

      end subroutine extras_controls


      subroutine extras_startup(id, restart, ierr)
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         select case(evol_phase)
         case(5)
            if (.not. restart) then
               s% ixtra(ix_TP_count) = 1
            !    s% lxtra(lx_keep_hydro_off) = .false.
            !    s% lxtra(lx_hydro_on) = .false.
               s% lxtra(lx_in_LHe_peak) = .true.
               s% xtra(x_min_core_mass_after_TDU) = s% he_core_mass
               s% xtra(x_max_core_mass_before_TDU) = s% he_core_mass
               s% xtra(x_interpulse_core_mass_growth) = 1.0
            end if
            VCT = s% x_ctrl(1)
            MXP = s% x_ctrl(2)
            envelope_overshooting_scheme = s% overshoot_scheme(2)

        case(6)
            if (.not. restart) then
               s% ixtra(ix_TP_count) = 1
               s% lxtra(lx_in_LHe_peak) = .true.
            end if


!--------- for hdf5 output for MPPNP nugrid post processing ---------
        ! if (s%x_logical_ctrl(2) .eqv. .true.) then
        !     call extras_startup_hdf5(id,restart,ierr)
        ! end if
!---------------------------------------------------------------------

         end select


      end subroutine extras_startup


      integer function extras_start_step(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_start_step = 0

         select case(evol_phase)
         case(2)
            ! controls to convergence degenerate stars through He flash
            if ((s%center_degeneracy > 5.0) .and. (s% use_gold_tolerances .eqv. .true.)) then
               write(*,*) 'easing controls to get through degenerate ignition'
               s% convergence_ignore_equL_residuals = .true.
               s% use_gold_tolerances = .false.
               s% use_gold2_tolerances = .false.
               s% varcontrol_target = 1d-3
            end if
            
         case(4)
            ! relax timestep controls to get through carbon burning
               if (s%co_core_mass>1.0) then
                  s% varcontrol_target = 1d-3
                  s% delta_lgL_He_limit = 0.025
                  s% lgL_He_burn_min = 2.5
               end if

               ! relax timestep controls for massive AGB stars
               if (s%co_core_mass>0.8) then
                  s% dX_limit_species(1) = 'h1'
                  s% dX_limit_min_X(1) = 1d99
                  s% dX_limit(1) = 1d99
                  s% dX_div_X_limit_min_X(1) = 1d-3
                  s% dX_div_X_limit(1) = 0.9d0
               end if

         end select
      end function extras_start_step


      ! returns either keep_going, retry, or terminate.
      integer function extras_check_model(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_check_model = keep_going

         ! if you want to check multiple conditions, it can be useful
         ! to set a different termination code depending on which
         ! condition was triggered.  MESA provides 9 customizeable
         ! termination codes, named t_xtra1 .. t_xtra9.  You can
         ! customize the messages that will be printed upon exit by
         ! setting the corresponding termination_code_str value.
         ! termination_code_str(t_xtra1) = 'my termination condition'

         ! by default, indicate where (in the code) MESA terminated
         if (extras_check_model == terminate) s% termination_code = t_extras_check_model
      end function extras_check_model


      integer function how_many_extra_history_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_columns = 4
         select case(evol_phase)
         case(5)
            how_many_extra_history_columns = 17

         case(6)
            how_many_extra_history_columns = 6
         end select

      end function how_many_extra_history_columns


      subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         real(dp) :: alpha_mlt_max, min_beta
         real(dp) :: c13_eff_pocket, c13_pocket_width
         integer :: i
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         ! note: do NOT add the extras names to history_columns.list
         ! the history_columns.list is only for the built-in history column options.
         ! it must not include the new column names you are adding here.

         call calc_alpha_mlt_max(s,alpha_mlt_max)

         names(1) = 'alpha_mlt_max'
         vals(1) = alpha_mlt_max

         names(2) = 'alpha_mlt_surface'
         vals(2) = s% alpha_mlt(1)

         call calc_min_beta(s,min_beta)
         names(3) = 'min_beta'
         vals(3) = min_beta

         names(4) = 'varcontrol'
         vals(4) = s% varcontrol_target

         select case(evol_phase)
         case(5)

         names(5) = 'conv_env_bot'
         vals(5) = conv_env_bot

         names(6) = 'TP_count'
         vals(6) = s% ixtra(ix_TP_count)

         names(7) = 'PDCZ_overshoot'
         vals(7) = s% overshoot_f(1)

         names(8) = 'CE_overshoot'
         vals(8) = s% overshoot_f(2)

         names(9) = 'lambda_DUP'
         if (s% lxtra(lx_in_LHe_peak) .eqv. .true.) then
             vals(9) = (s% xtra(x_max_core_mass_before_TDU)-s%he_core_mass)/ s% xtra(x_interpulse_core_mass_growth)
         else
             vals(9) = 0
         end if

         names(10) = 'mesh_dlogX_dlogP_extra'
         vals(10) = s% mesh_dlogX_dlogP_extra(1)

         call calc_c13_pocket_mass(s,c13_eff_pocket,c13_pocket_width)

         names(11) = 'c13_eff_pocket_mass'
         vals(11) = c13_eff_pocket

         names(12) = 'c13_pocket_width'
         vals(12) = c13_pocket_width

         !coordinate at bottom of all mixing in envelope
         do i = s%num_mixing_regions,1,-1
             if (s%m(s%mixing_region_bottom(i))/Msun>s%he_core_mass) exit
         end do

         names(13) = 'env_mix_bot'
         vals(13) = s%m(s%mixing_region_bottom(i))/Msun

         names(14) = 'env_mix_bot_temp'
         vals(14) = s%T(s%mixing_region_bottom(i))

         names(15) = 'logP'
         vals(15) = logP

         names(16) = 'central_degeneracy'
         vals(16) = s% center_degeneracy

         names(17) = 'max_degeneracy'
         vals(17) = MAXVAL(s% eta)

         case(6)

         names(5) = 'conv_env_bot'
         vals(5) = conv_env_bot

         names(6) = 'TP_count'
         vals(6) = s% ixtra(ix_TP_count)



         end select


      end subroutine data_for_extra_history_columns


      integer function how_many_extra_profile_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_columns = 3
      end function how_many_extra_profile_columns


      subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
         integer, intent(in) :: id, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         integer :: k
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         ! note: do NOT add the extra names to profile_columns.list
         ! the profile_columns.list is only for the built-in profile column options.
         ! it must not include the new column names you are adding here.

         ! here is an example for adding a profile column
         !if (n /= 1) stop 'data_for_extra_profile_columns'
         !names(1) = 'beta'
         !do k = 1, nz
         !   vals(k,1) = s% Pgas(k)/s% P(k)
         !end do

         names(1) = 'beta'
         do k = 1, nz
             vals(k,1) = s% Pgas(k)/s% Peos(k)
         end do

         names(2) = 'alpha_mlt'
         do k = 1, nz
             vals(k,2) = s% alpha_mlt(k)
         end do

         names(3) = 'eps_grav_ad'
         do k = 1, nz
            vals(k,3) = s% eps_grav_ad(k) %val
         end do

        ! bug with this quantity
        !  names(4) = 'eta'
        !  do k = 1, nz
        !     vals(k,1) = s% eta(k)
        !  end do

      end subroutine data_for_extra_profile_columns


      integer function how_many_extra_history_header_items(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_header_items = 0
      end function how_many_extra_history_header_items


      subroutine data_for_extra_history_header_items(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         type(star_info), pointer :: s
         integer, intent(out) :: ierr
         ierr = 0
         call star_ptr(id,s,ierr)
         if(ierr/=0) return

         ! here is an example for adding an extra history header item
         ! also set how_many_extra_history_header_items
         ! names(1) = 'mixing_length_alpha'
         ! vals(1) = s% mixing_length_alpha

      end subroutine data_for_extra_history_header_items


      integer function how_many_extra_profile_header_items(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_header_items = 0
      end function how_many_extra_profile_header_items


      subroutine data_for_extra_profile_header_items(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(n)
         type(star_info), pointer :: s
         integer, intent(out) :: ierr
         ierr = 0
         call star_ptr(id,s,ierr)
         if(ierr/=0) return

         ! here is an example for adding an extra profile header item
         ! also set how_many_extra_profile_header_items
         ! names(1) = 'mixing_length_alpha'
         ! vals(1) = s% mixing_length_alpha

      end subroutine data_for_extra_profile_header_items


      ! returns either keep_going or terminate.
      ! note: cannot request retry; extras_check_model can do that.
      integer function extras_finish_step(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         logical :: do_termination
         character(len=50) :: do_termination_code_str
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_finish_step = keep_going

         ! to save a profile,
            ! s% need_to_save_profiles_now = .true.
         ! to update the star log,
            ! s% need_to_update_history_now = .true.

         select case(evol_phase)
         case(4)
            call check_for_onset_TPs(s,do_termination,do_termination_code_str)

         case(5)
            ! check if TP started
            call check_for_TP(s)

            ! turn hydro on or off
            !call control_hydro(id,s,ierr)

            ! resolve TDU
            call resolve_TDU(s,ierr)

            ! turn off envelope overshooting for HTDU
            call check_for_HTDU(s,ierr)

        case(6)
            call check_for_TP(s)


         !terminate TP-AGB when H-rich envelope falls below 20% of current stellar mass (Dotter et al 2016)
         ! if ((s%star_mass-s%he_core_mass)/s%star_mass<0.2) then
         !     do_termination_code_str = 'terminate TP-AGB'
         !     extras_finish_step = terminate
         ! end if

!--------- for hdf5 output for MPPNP nugrid post processing ---------
         ! if (s%x_logical_ctrl(2) .eqv. .true.) then
         !     call extras_finish_step_hdf5(id)
         ! end if
!--------------------------------------------------------------------
         end select

         if (do_termination .eqv. .True.) then
            termination_code_str(t_xtra1) = do_termination_code_str
            s% termination_code = t_xtra1
            extras_finish_step = terminate
         end if

         ! see extras_check_model for information about custom termination codes
         ! by default, indicate where (in the code) MESA terminated
        !  if (extras_finish_step == terminate) s% termination_code = t_extras_finish_step
      end function extras_finish_step


      subroutine extras_after_evolve(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
      end subroutine extras_after_evolve
