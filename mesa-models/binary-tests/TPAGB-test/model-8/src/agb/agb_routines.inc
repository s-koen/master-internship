!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Extra routines for TPAGB modelling
      ! Routines explained in Rees et al. 2024 and Rees & Izzard 2024
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine check_for_onset_TPs(s,do_termination,do_termination_code_str)
         logical, intent(out) :: do_termination
         character(len=50), intent(out) :: do_termination_code_str
         type (star_info), pointer :: s

         ! terminate if logLHe <-3
         if (s%power_he_burn<1d-3) then
             do_termination_code_str = 'no_he_burn'
             do_termination = .true.
         end if

         ! check for thermal pulse, require higher he burning lum for higher core masses so not confused with carbon burning
         if ((s%he_core_mass-s%co_core_mass<0.1) .and. (((s% power_he_burn .gt. 10**(4)) .AND. (s%he_core_mass < 0.8)) .OR.((s% power_he_burn .gt. 10**(4.5)) .AND. (s%he_core_mass<= 1.2)) .OR. ((s% power_he_burn .gt. 10**(5.5)) .AND. (s%he_core_mass> 1.2)))) then
             s% need_to_update_history_now = .true.
             s% need_to_save_profiles_now = .true.
             write(*,*) 'starting thermal pulse'
             do_termination_code_str = 'TP_started'
             do_termination = .true.
         end if

     end subroutine check_for_onset_TPs

     subroutine check_for_TP(s)
      ! thermal pulse detection and counting
      ! call subroutine in extras_finish_step
      type (star_info), pointer :: s

      ! save core mass for use in resolving TDU
      s% xtra(x_he_core_mass) = s% he_core_mass

      ! update minimum core mass
      if (s% he_core_mass < s% xtra(x_min_core_mass_after_TDU)) s% xtra(x_min_core_mass_after_TDU) = s% he_core_mass
      if (s% he_core_mass > s% xtra(x_max_core_mass_before_TDU)) s% xtra(x_max_core_mass_before_TDU) = s% he_core_mass

      ! record thermal pulses
      if (.not. s% lxtra(lx_in_LHe_peak)) then
          ! check for He burning
          if (s% power_he_burn .gt. 1e4) then
              s% lxtra(lx_in_LHe_peak) = .true.
              s% ixtra(ix_TP_count) = s% ixtra(ix_TP_count) + 1
              s% need_to_update_history_now = .true.
              s% need_to_save_profiles_now = .true.
              write(*,*) 'starting thermal pulse'
              ! save core mass properties for TDU
              s% xtra(x_interpulse_core_mass_growth) = s% he_core_mass - s% xtra(x_min_core_mass_after_TDU)
              write(*,*) 'Core mass growth during interpulse = ', s% xtra(x_interpulse_core_mass_growth)
              s% xtra(x_min_core_mass_after_TDU) = s% he_core_mass
              s% xtra(x_max_core_mass_before_TDU) = s% he_core_mass
          end if
      else
          ! check if pulse over
          if ((s%power_he_burn .lt. 1d3) .and. (s% power_h_burn/s% power_he_burn .gt. 10)) s% lxtra(lx_in_LHe_peak) = .false.
      end if

  end subroutine check_for_TP

!   subroutine control_hydro(id,s,ierr)
!       ! NO LONGER REQUIRED AS PREVENT_HRI ROUTINE USED INSTEAD
!       ! turns hydro on for He shell flash to prevent convergence problems
!       ! turns hydro on at start of thermal pulse
!       ! turns hydro of once LHe drops below max value reached and is less than 1d6
!       ! call subroutine in extras_finish_step
!       ! must use with check_for_TP so that in_LHe_peak is updated
!       integer, intent(in) :: id
!       integer, intent(out) :: ierr
!       integer :: i
!       type (star_info), pointer :: s
!       ! ierr = 0
!       ! call star_ptr(id, s, ierr)
!       ! if (ierr /= 0) return

!       !write(*,*) 'hydro_on = ', s% lxtra(lx_hydro_on), 'in_LHe_peak = ', s% lxtra(lx_in_LHe_peak)
!       !write(*,*) 'max LHe = ', s% xtra(x_max_LHe), 'LHe = ', s%power_he_burn

!       !routine for turning hydro on and off during He shell flash
!       if (s% lxtra(lx_hydro_on) .eqv. .false.) then
!           call star_set_v_flag(id, .false., ierr)
!           if ((s% lxtra(lx_in_LHe_peak)) .AND. (.not. s% lxtra(lx_keep_hydro_off))) then
!               s% lxtra(lx_hydro_on) = .true.
!               s% xtra(x_max_LHe) = s% power_he_burn
!           else if (.not. s% lxtra(lx_in_LHe_peak)) then
!               s% lxtra(lx_keep_hydro_off) = .false.
!           end if
!       else
!           call star_set_v_flag(id, .true., ierr)
!           if (s% power_he_burn > s% xtra(x_max_LHe)) then
!               s% xtra(x_max_LHe) = s% power_he_burn
!           else if ((s% power_he_burn < (s% xtra(x_max_LHe)/1.1)) .AND. (s% power_he_burn < 1d6)) then
!               s% lxtra(lx_hydro_on) = .false.
!               s% lxtra(lx_keep_hydro_off) = .true.
!           end if
!       end if

!   end subroutine control_hydro

  subroutine resolve_TDU(s,ierr)
    ! routine to increase temporal and spatial resolution during TDU (Rees et al. 2024)
    ! call subroutine in extras_finish_step
    integer, intent(out) :: ierr
    integer :: i,j
    type (star_info), pointer :: s
    ierr = 0
    ! call star_ptr(id, s, ierr)
    ! if (ierr /= 0) return

    ! write(*,*) 'VCT = ', VCT

    do j = 1, s%num_mixing_regions
        if (s%cz_bot_mass(j)/Msun>s%he_core_mass) exit
    end do

    ! if convective envelope moving inwards increase resolution
    if (conv_env_bot>s%cz_bot_mass(j)) then
        if (s%varcontrol_target>VCT+1d-8) then
            s%varcontrol_target = s%varcontrol_target -1d-6
            write(*,*) 'reduced varcontrol_target = ', s%varcontrol_target
        end if
        if (s%mesh_dlogX_dlogP_extra(1)>MXP+0.0001) then
            s%mesh_dlogX_dlogP_extra(1) = s%mesh_dlogX_dlogP_extra(1)-0.01
            write(*,*) 'reduced mesh_dlogX_dlogP_extra for he4 = ', s%mesh_dlogX_dlogP_extra(1)
        end if
    else
        if (s%varcontrol_target<0.999d-4) then
            s%varcontrol_target = s%varcontrol_target +1d-6
            write(*,*) 'reduced varcontrol_target = ', s%varcontrol_target
        end if
        if (s%mesh_dlogX_dlogP_extra(1)<0.999) then
            s%mesh_dlogX_dlogP_extra(1) = s%mesh_dlogX_dlogP_extra(1)+0.01
            write(*,*) 'reduced mesh_dlogX_dlogP_extra for he4 = ', s%mesh_dlogX_dlogP_extra(1)
        end if
    end if

    conv_env_bot = s%cz_bot_mass(j)

end subroutine resolve_TDU


integer function other_timestep_limit_resolve_TDU( &
  id, skip_hard_limit, dt, dt_limit_ratio)
  ! Rees & Izzard 2024
  ! Second method developed to reesolve third dredge-up
  use const_def, only: dp
  integer, intent(in) :: id
  logical, intent(in) :: skip_hard_limit
  real(dp), intent(in) :: dt
  real(dp), intent(inout) :: dt_limit_ratio
  integer :: ierr, j
  real(dp) :: delta_conv_env_bot
  type (star_info), pointer :: s
  ierr = 0
  call star_ptr(id, s, ierr)
  if (ierr /= 0) return

  other_timestep_limit_resolve_TDU = keep_going

  ! find bottom of convective envelope
  do j = 1, s%num_mixing_regions
    if (s%cz_bot_mass(j)/Msun>s%he_core_mass) exit
  end do

  delta_conv_env_bot = (conv_env_bot-s%cz_bot_mass(j))/Msun

  !reduce timestep if change in mass of conv_env_bot is greater than x_ctrl(3)*interpulse_core_mass_growth
  dt_limit_ratio = (delta_conv_env_bot/s% xtra(x_interpulse_core_mass_growth))/s% x_ctrl(3)
  if ((dt_limit_ratio>0) .and. (s%power_he_burn<1d5)) then
    write(*,*) 'Interpulse core masss growth = ', s% xtra(x_interpulse_core_mass_growth)
    write(*,*) 'Change in conv_env_bot = ', delta_conv_env_bot
    write(*,*) 'dt_limit_ratio = ', dt_limit_ratio
  end if

  conv_env_bot = s%cz_bot_mass(j)

end function other_timestep_limit_resolve_TDU


  subroutine check_for_HTDU(s,ierr)
    ! Rees et al. 2024
      ! turns of envelope overshooting when convective boundary hot during TDU to avoid convergence problems due to HTDU
      ! turn envelope overshooting off during TDU if temp at base of convective envelope > 4e7
      ! must have envelope overshooting as overshooting controls group 2
      ! call subroutine in extras_finish_step
      integer, intent(out) :: ierr
      integer :: i
      type (star_info), pointer :: s
      ierr = 0
      ! call star_ptr(id, s, ierr)
      ! if (ierr /= 0) return

      !find zone number for bottom of all mixing in envelope
      do i = s%num_mixing_regions,1,-1
          if (s%m(s%mixing_region_bottom(i))/Msun>s%he_core_mass) exit
      end do

      ! change envelope overshooting
      if ((s%T(s%mixing_region_bottom(i))>0.4*1d8) .and. (s%varcontrol_target<9.9d-5)) then
          s% overshoot_scheme(2) = ' '
          write(*,*) 'no envelope overshooting, T=', s%T(s%mixing_region_bottom(i))
      else
          s% overshoot_scheme(2) = envelope_overshooting_scheme !'other' !'exponential'
      end if

  end subroutine check_for_HTDU

  subroutine other_alpha_mlt_prevent_FeI(id, ierr)
   ! Rees & Izzard 2024
   ! increase mixing length parameter as a function of beta=Pgas/P to avoid the Fe-peak instability
   ! in extras_controls:
   !     s% other_alpha_mlt => other_alpha_mlt_prevent_FeI
   ! in inlist:
   !     use_other_alpha_mlt = .true.

   integer, intent(in) :: id
   integer, intent(out) :: ierr
   type (star_info), pointer :: s
   integer :: i, inner_index
   real(dp) :: beta, logT, A, B, A1, A2, alpha_mlt
   ierr = 0
   call star_ptr(id, s, ierr)
   if (ierr /= 0) return

   select case (evol_phase)
   case(5)
       ! for use in TP-AGB models
       inner_index = s%he_core_k
       if (s% power_h_burn>1d1) then
           A1 = s% mixing_length_alpha
       else
           A1 = 50.0
       end if
       A2=-9*(s%star_mass/s%he_core_mass-1)+20
       A = MAX(A1,A2)
       B=0.3

   case(6)
       ! for use in post-AGB models
       inner_index =s%he_core_k
       A = 50
       B = 0.3
   end select

   call alpha_mlt_beta(id, ierr, A, B, inner_index)

end subroutine other_alpha_mlt_prevent_FeI


subroutine other_eps_grav_prevent_HRI(id, k, dt, ierr)
   ! Rees & Izzard 2024
   ! prevents HRI by damping hydrogen recombination energy in outer envelope
   ! must add use_other_eps_grav = .true. in inlist
   ! must add s% other_eps_grav => other_eps_grav_prevent_HRI in extras_controls
   use auto_diff
   integer, intent(in) :: id, k
   real(dp), intent(in) :: dt
   integer, intent(out) :: ierr
   type (star_info), pointer :: s
   real(dp) :: f, fmin

   ierr = 0
   call star_ptr(id, s, ierr)
   if (ierr /= 0) return

   ! NOTE: this is called after 1st doing the standard eps_grav calculation.
   ! so if you decide you don't want to do anything special, just return.

   ! NOTE: need to set the auto_diff variable s% eps_grav_ad(k)
   ! this is type(auto_diff_real_star_order1) so includes partials.
   ! in addition to setting the value,
   ! you must also set the partials, and there are lots of them.

   fmin = 0d0
   f = MAX(fmin,MIN(1d0,1/0.3*(safe_log10(s%T(k))-4.2)))

   s% eps_grav_ad(k) = f*s% eps_grav_ad(k)

end subroutine other_eps_grav_prevent_HRI


! wind mass loss routine
subroutine vassilliadis_wood_wind(id, Lsurf, Msurf, Rsurf, Tsurf, X, Y, Z, w, ierr)
   ! Vassilliadis & Wood 1993 wind mass loss rate
   ! in extras_controls:
   !      s% other_wind => vassilliadis_wood_wind
   ! in inlist:
   !      use_other_wind = .true.

   use star_def
   use chem_def, only: ic12, io16

   integer, intent(in) :: id
   real(dp), intent(in) :: Lsurf, Msurf, Rsurf, Tsurf, X, Y, Z ! surface values (cgs)
   ! NOTE: surface is outermost cell. not necessarily at photosphere.
   ! NOTE: don't assume that vars are set at this point.
   ! so if you want values other than those given as args,
   ! you should use values from s% xh(:,:) and s% xa(:,:) only.
   ! rather than things like s% Teff or s% lnT(:) which have not been set yet.
   real(dp), intent(out) :: w ! wind in units of Msun/year (value is >= 0)
   integer, intent(out) :: ierr
   real(dp) :: Mdot1, Mdot2, vexp, P, factor
   real(dp) :: m, r, CO
   integer :: c12, o16
   !print*, 'Msurf', Msurf,'Rsurf', Rsurf

   type(star_info), pointer :: s
   call star_ptr(id,s,ierr)
   if(ierr/=0) return

   ! period from Vassilliadis & Wood
   logP = -2.07+1.94*log10(Rsurf/(6.957D10))-0.9*log10(Msurf/(1.9885D33))
   P = 10.0**logP

   ! fundamental period from Trabucchi et al 2018
!   m = log10(Msurf/(1.9885D33))
!   r = log10(Rsurf/(6.957D10))
!   c12 = s% net_iso(ic12)
!   o16 = s% net_iso(io16)
!   CO = s% xa(c12,1)/ s% xa(o16,1)

!   logP = -1.12166+1.24449*m+1.07886*r+0.87741*m**2-1.53239*m*r+0.10382*r**2-0.07659*m**3-0.26130*m**2*r+0.26867*m*r**2+0.03278*r**3-0.02713*log10(Z)+0.14872*Y-0.01455*log10(CO/0.4125541984736385)
!   P = 10.0**logP

   vexp = -13.5 + 0.056*P

   if (vexp .lt. 3.0) then
       vexp = 3.0
   else if (vexp .gt. 15.0) then
       vexp = 15.0
   end if

   if (s% initial_mass >= 2.5) then
       Mdot1 = 10.0**(-11.4 + 0.0125*(P-100*(Msurf/1.9885D33-2.5)))
   else
       Mdot1 = 10.0**(-11.4 + 0.0123*P)
   end if

   Mdot2 = Lsurf*3.154D7/(2.9979D10*vexp*1D5*1.9885D33)

   select case(evol_phase)
   case(4)
       factor = 1.0
   case(5)
       factor = 1.0
   case(6)
       factor = 1.0
   end select
   !write(*,*) 'factor = ', factor
   w = factor*MIN(Mdot1, Mdot2)
   !write(*,*) w

   !mass-loss-period relation from Straniero et al 2006
   ! if (logP<2.5) then
   !    w = 10**(-7.7)
   ! else if (logP<=3.1) then
   !    w = 10**(-101.6+63.2*logP-10.282*logP**2)
   ! else
   !    w = 10**(-4.3)
   ! end if

   ierr = 0
end subroutine vassilliadis_wood_wind



subroutine calc_c13_pocket_mass(s,c13_eff_pocket,c13_pocket_width)
   use chem_def, only: ih1, ic13, in14
   real(dp), intent(out) :: c13_pocket_width, c13_eff_pocket
   real(dp) :: c13_eff
   integer :: k
   type (star_info), pointer :: s

   !carbon 13 pocket mass defintion of Cristallo et al 2011
   c13_eff_pocket = 0
   do k = s%he_core_k, s%co_core_k
       c13_eff = s% xa(s% net_iso(ic13),k)-s% xa(s% net_iso(in14),k)*13/14
       if (c13_eff >0) then
           c13_eff_pocket = c13_eff_pocket + c13_eff* s% dm(k)/Msun
       end if
   end do


   !carbon 13 pocket mass definition of Battino et al 2016
   c13_pocket_width = 0
   do k = s%he_core_k, s%co_core_k
       if ((s% xa(s% net_iso(ic13),k)-s% xa(s% net_iso(in14),k)>0) .and. (s%xa(s% net_iso(ic13),k)>1d-3)) then
           c13_pocket_width = c13_pocket_width + s% dm(k)/Msun
       end if
   end do

end subroutine calc_c13_pocket_mass
