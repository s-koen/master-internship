!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Extra routines for binary TPAGB modelling
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine save_model_on_radius_increase(id, s, ierr)

    integer, intent(in) :: id
    type(star_info), pointer :: s
    integer, intent(inout) :: ierr

    character(len=200) :: fname
    real(dp) :: R_now, R_last, center_h1

    ! the subroutine uses the current stellar radius 
    ! and the last saved stellar radius to determine
    ! if a model should be saved.
    R_now = s% photosphere_R
    R_last = s% xtra(x_last_saved_R)

    ! in order to prevent large pre-main sequence
    ! radii from "blocking" models on the main
    ! sequence, the central hydrogen abundance is
    ! used to get a crude estimate for the end of
    ! the pre main sequence.
    center_h1 = s% center_h1

    ! prevent NaNs.
    if (.not. (center_h1 == center_h1)) then
        center_h1 = 1d0
    end if

    ! if true: save a model

    ! this criterion produces many models, 
    ! but using this routine is only required once
    ! for every mass star!
    if (((R_now - R_last > 10d0) .or. (R_now > 1.25d0 * R_last)) .and. (center_h1 < 0.68d0)) then
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
