!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Extra routines for binary TPAGB modelling
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine update_stellar_spin(s,ierr)
    integer, intent(out) :: ierr
    type (star_info), pointer :: s

    real(dp) :: omega, dt
    real(dp) :: domega_dt, domega_dt_structure

    ! --- get initial spin and orbital frequency
    omega = s%xtra(x_omega)
    dt = s%dt

    ! --- compute structural change
    domega_dt_structure = compute_domega_dt_structure(s)

    ! --- total spin evolution
    domega_dt = domega_dt_structure &
              + s%xtra(x_domega_dt_wind) &
              + s%xtra(x_domega_dt_tides)

    ! --- update spin
    omega = omega + dt*domega_dt
    s%xtra(x_omega) = omega

end subroutine update_stellar_spin

subroutine save_model_on_radius_increase(id, s, ierr)

    integer, intent(in) :: id
    type(star_info), pointer :: s
    integer, intent(inout) :: ierr

    character(len=200) :: fname
    real(dp) :: R_now, R_last
    real(dp) :: R_max_since_last_save
    real(dp) :: R_max
    integer :: evol_phase
    logical :: needs_save

    needs_save = s% lxtra(lx_needs_save)
    evol_phase = s%x_integer_ctrl(1)

    ! the subroutine uses the current stellar radius 
    ! and the last saved stellar radius to determine
    ! if a model should be saved.
    R_now = s% photosphere_R
    R_last = s% xtra(x_last_saved_R)
    R_max_since_last_save = s% xtra(x_max_R_since_last_save)

    ! if true: save a model

    ! this criterion produces many models, 
    ! but using this routine is only required once
    ! for every mass star!
    
    if (((R_now - R_last > 10d0) .or. (R_now > 1.25d0 * R_last)) .and. (evol_phase > 1) .and. (evol_phase /= 5)) then
        write(fname, fmt="(a7,f0.3,a7,i0,a4)") 'models/', R_now, 'Rsun_TP', s% ixtra(ix_TP_count), '.mod'

        call star_write_model(id, fname, ierr)

        s% need_to_update_history_now = .true.
        s% need_to_save_profiles_now = .true.

        open(1, file="last_saved_R.dat", status="replace")
        write(1,*) R_now
        close(1)

        if (ierr /= 0) then
            write(*,*) 'error writing model'
            ierr = 0
        end if

        s% xtra(x_last_saved_R) = R_now
        s% xtra(x_max_R_since_last_save) = R_now
    end if

    if ((R_now - R_max_since_last_save > 0d0) .and. (evol_phase == 5) .and. (s% power_h_burn / s% power_nuc_burn < 0.7)) then
        s% xtra(x_max_R_since_last_save) = R_now
        s% lxtra(lx_needs_save) = .true.
    end if

    if ((s% power_h_burn / s% power_nuc_burn > 0.7) .and. (s% lxtra(lx_needs_save)) .and. (evol_phase == 5)) then

        write(fname, fmt="(a7,f0.3,a7,i0,a4)") 'models/', R_max_since_last_save, 'Rsun_TP', s% ixtra(ix_TP_count), '.mod'

        call star_write_model(id, fname, ierr)

        s% need_to_update_history_now = .true.
        s% need_to_save_profiles_now = .true.

        open(1, file="last_saved_R.dat", status="replace")
        write(1,*) R_now
        close(1)

        if (ierr /= 0) then
            write(*,*) 'error writing model'
            ierr = 0
        end if

        s% xtra(x_last_saved_R) = R_max_since_last_save
        s% lxtra(lx_needs_save) = .false.
    end if


end subroutine save_model_on_radius_increase

subroutine load_last_saved_radius(s)

    type (star_info), pointer, intent(inout) :: s

    integer :: ios
    real(dp) :: stored_R

    open(1, file='last_saved_R.dat', status='old', action='read', iostat=ios)
    if (ios == 0) then
        read(1,*,iostat=ios) stored_R

        if (ios == 0) then
            s%xtra(x_last_saved_R) = stored_R
            write(*,*) 'loaded last_R = ', stored_R
        end if
        close(1)
    else
        write(*,*) 'no last_R.dat found, initializing with 0'
        s%xtra(x_last_saved_R) = 0
    end if

end subroutine load_last_saved_radius

subroutine add_hist(names, vals, k, name, val)
   integer, intent(inout) :: k
   character(len=*), intent(in) :: name
   character(len=maxlen_history_column_name) :: names(:)
   real(dp) :: vals(:)
   real(dp), intent(in) :: val

   k = k + 1
   names(k) = name
   vals(k)  = val
end subroutine

subroutine calc_quasi_adiabadic_critical_mass_loss(s, max_Mdot_crit)
    type (star_info), pointer, intent(inout) :: s
    real(dp), intent(out) :: max_Mdot_crit
    real(dp) :: cum_energy, tau_local, Mdot_crit
    integer :: k

    cum_energy = 0d0
    max_Mdot_crit = -1d99

    do k = 1, s% nz
        cum_energy = cum_energy + s% dm(k) * s% cp(k) * s% T(k)

        if (s% L(1) <= 0d0) cycle
        tau_local = cum_energy / s% L(1) 

        if (tau_local <= 0d0) cycle
        Mdot_crit = (s% mstar - s% m(k)) / Msun / (tau_local / secyer)
        max_Mdot_crit = max(max_Mdot_crit, Mdot_crit)
    end do
end subroutine calc_quasi_adiabadic_critical_mass_loss

real(dp) function compute_domega_dt_structure(s) result(domega_dt_structure)
    implicit none

    type(star_info), pointer :: s
    real(dp) :: omega, I_eff, I_old 
    real(dp) :: dlnI_dt

    omega = s%xtra(x_omega)
    I_eff = s%xtra(x_I_eff)

    ! --- angular momentum loss due to structural change
    I_old = s%xtra(x_I_old)
    if (I_old > 0d0 .and. s%dt > 0d0) then
        dlnI_dt = (log(I_eff) - log(I_old))/s%dt
    else
        dlnI_dt = 0d0
    end if
    domega_dt_structure = -omega*dLnI_dt
    s%xtra(x_domega_dt_structure) = domega_dt_structure
    s%xtra(x_I_old) = s%xtra(x_I_eff)

end function compute_domega_dt_structure

real(dp) function compute_vwind(s) result(vwind)
    implicit none

    type(star_info), pointer :: s

    ! ---
    ! wind velocity modeled after Lamers (1995),
    ! which is not specifically calibrated for TPAGB
    ! stars
    ! ---

    ! real(dp) :: m, R, log_Teff
    ! real(dp) :: v_esc, vw_factor

    ! m = s% star_mass * Msun
    ! R = s%photosphere_r * Rsun
    ! log_Teff = log10(s% Teff)

    ! v_esc = sqrt(2.0 * standard_cgrav * m / R)
    ! if (log_Teff < 3.5d0) then
    !     vw_factor = 0.35d0
    ! else if (log_Teff <= 4.0d0) then
    !     vw_factor = 0.7d0 + 0.7d0 * (log_Teff - 4.0d0)
    ! else if (log_Teff <= 4.35d0) then
    !     vw_factor = 1.3d0
    ! else
    !     vw_factor = 2.6d0
    ! end if 
    ! vwind = vw_factor * v_esc
    
    vwind = 1.5d6 ! cm / s
end function compute_vwind

real(dp) function compute_I_eff(s) result(I_eff)
    implicit none

    type(star_info), pointer :: s
    integer :: k

    I_eff = 0d0
    do k = 1, s%nz
        I_eff = I_eff + (2d0/3d0)*s%dm_bar(k)*s%r(k)**2
    end do
end function compute_I_eff

real(dp) function compute_rg2(s) result(rg2)
    implicit none

    type(star_info), pointer :: s
    integer :: k
    real(dp) :: I_eff, M, R

    I_eff = s%xtra(x_I_eff)
    M = s%mstar
    R = s%r(1)

    if (M > 0d0 .and. R > 0d0) then
        rg2 = I_eff/(M*R*R)
    else
        rg2 = 0d0
    end if

end function compute_rg2

real(dp) function compute_M_conv(s) result(M_conv)
    implicit none

    type(star_info), pointer :: s
    integer :: i
    real(dp) :: q_top, q_bot, q_max

    M_conv = 0
    q_max = 0
    do i = 1, s%n_conv_regions
        if (s%cz_top_mass(i)/Msun - s%he_core_mass > 8d-3) then
            ! --- starts sufficiently close to the edge
            q_top = s%cz_top_mass(i)/Msun/s%star_mass
            q_bot = s%cz_bot_mass(i)/Msun/s%star_mass


            if (q_top - q_bot > q_max) then
                q_max = q_top - q_bot

            end if

        end if
    end do

    M_conv = s%mstar*q_max/Msun

end function compute_M_conv

real(dp) function compute_R_conv(s) result(R_conv)
    implicit none

    type(star_info), pointer :: s
    real(dp) :: R_bot, R_top
    real(dp) :: q_top, q_bot, q_max
    real(dp) :: q_top_max, q_bot_max
    integer :: k, i

    R_conv = 0

    ! --- first find q_top and q_bot

    q_max = 0
    q_top_max = 0
    q_bot_max = 0

    do i = 1, s%n_conv_regions
        if (s%cz_top_mass(i)/Msun - s%he_core_mass > 8d-3) then
            ! --- starts sufficiently close to the edge
            q_top = s%cz_top_mass(i)/Msun/s%star_mass
            q_bot = s%cz_bot_mass(i)/Msun/s%star_mass

            if (q_top - q_bot > q_max) then
                q_max = q_top - q_bot
                q_top_max = q_top
                q_bot_max = q_bot
            end if
        end if
    end do

    ! --- now find which rs this corresponds to
    R_top = 0d0

    do k = 1, s%nz
        if (s%m(k)/Msun/s%star_mass <= q_top_max) then
            R_top = s%r(k)
            exit
        end if
    end do

    R_bot = 0d0 ! for safety

    do k = k, s%nz
        if (s%m(k)/Msun/s%star_mass <= q_bot_max) then
            R_bot = s%r(k)
            exit
        end if
    end do

    R_conv = (R_top - R_bot)/Rsun

end function compute_R_conv

real(dp) function compute_t_conv(s) result(t_conv)
    implicit none

    type(star_info), pointer :: s
    real(dp) M_conv, R_conv, L
    M_conv = s%xtra(x_M_conv)
    R_conv = s%xtra(x_R_conv)

    t_conv = 0.4311d0*(3*M_conv*R_conv**2/(s%photosphere_L))**(1d0/3d0)*secyer

end function compute_t_conv
