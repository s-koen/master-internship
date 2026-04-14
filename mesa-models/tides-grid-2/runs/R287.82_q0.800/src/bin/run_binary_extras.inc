subroutine extras_binary_controls(binary_id, ierr)
    integer :: binary_id
    integer, intent(out) :: ierr
    type(binary_info), pointer :: b
    ierr = 0

    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then
        write (*, *) 'failed in binary_ptr'
        return
    end if

    ! Set these function pointers to point to the functions you wish to use in
    ! your run_binary_extras. Any which are not set, default to a null_ version
    ! which does nothing.
    b%how_many_extra_binary_history_header_items => how_many_extra_binary_history_header_items
    b%data_for_extra_binary_history_header_items => data_for_extra_binary_history_header_items
    b%how_many_extra_binary_history_columns => how_many_extra_binary_history_columns
    b%data_for_extra_binary_history_columns => data_for_extra_binary_history_columns

    b%extras_binary_startup => extras_binary_startup
    b%extras_binary_start_step => extras_binary_start_step
    b%extras_binary_check_model => extras_binary_check_model
    b%extras_binary_finish_step => extras_binary_finish_step
    b%extras_binary_after_evolve => extras_binary_after_evolve


    b% other_jdot_ls => tides 
    b% other_jdot_ml => wind_loss 
    b% other_binary_wind_transfer => wind_accretion 

    ! Once you have set the function pointers you want, then uncomment this (or set it in your star_job inlist)
    ! to disable the printed warning message,
    b%warn_binary_extra = .false.

end subroutine extras_binary_controls

integer function how_many_extra_binary_history_header_items(binary_id)
    use binary_def, only: binary_info
    integer, intent(in) :: binary_id
    how_many_extra_binary_history_header_items = 0
end function how_many_extra_binary_history_header_items

subroutine data_for_extra_binary_history_header_items( &
    binary_id, n, names, vals, ierr)
    type(binary_info), pointer :: b
    integer, intent(in) :: binary_id, n
    character(len=maxlen_binary_history_column_name) :: names(n)
    real(dp) :: vals(n)
    integer, intent(out) :: ierr
    ierr = 0
    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then
        write (*, *) 'failed in binary_ptr'
        return
    end if
end subroutine data_for_extra_binary_history_header_items

integer function how_many_extra_binary_history_columns(binary_id)
    use binary_def, only: binary_info
    integer, intent(in) :: binary_id
    how_many_extra_binary_history_columns = 8
end function how_many_extra_binary_history_columns

subroutine data_for_extra_binary_history_columns(binary_id, n, names, vals, ierr)
    use const_def, only: dp
    type(binary_info), pointer :: b
    integer, intent(in) :: binary_id
    integer, intent(in) :: n
    character(len=maxlen_binary_history_column_name) :: names(n)
    real(dp) :: vals(n)
    integer, intent(out) :: ierr
    real(dp) :: beta
    integer :: k
    ierr = 0
    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then
        write (*, *) 'failed in binary_ptr'
        return
    end if

    k = 0
    call add_hist(names, vals, k, 'Omega_star', b% s_donor% xtra(x_omega))
    call add_hist(names, vals, k, 'Omega_orb', 2d0*pi/b% period)
    call add_hist(names, vals, k, 'beta_accretion', b% s_donor% xtra(x_beta))
    call add_hist(names, vals, k, 'eta_accretion', b% s_donor% xtra(x_eta))
    call add_hist(names, vals, k, 'f_conv', b% s_donor% xtra(x_f_conv))
    call add_hist(names, vals, k, 'vwind_over_vorbit', b% s_donor% xtra(x_vw_over_vorb))
    call add_hist(names, vals, k, 'angular_momentum_j_orbit', b% angular_momentum_j)
    call add_hist(names, vals, k, 'wind_mass_transfer', b% mdot_wind_transfer(b% d_i))

end subroutine data_for_extra_binary_history_columns

integer function extras_binary_startup(binary_id, restart, ierr)
    type(binary_info), pointer :: b
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr
    logical, intent(in) :: restart
    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then ! failure in  binary_ptr
        return
    end if

    call test_suite_startup(b, restart, ierr)
    b%s_donor% xtra(x_omega) = b%s_donor% x_ctrl(3)
    extras_binary_startup = keep_going
end function extras_binary_startup

integer function extras_binary_start_step(binary_id, ierr)
    type(binary_info), pointer :: b
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr

    extras_binary_start_step = keep_going
    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then ! failure in  binary_ptr
        return
    end if

end function extras_binary_start_step

!Return either keep_going, retry or terminate
integer function extras_binary_check_model(binary_id)
    type(binary_info), pointer :: b
    integer, intent(in) :: binary_id
    integer :: ierr
    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then ! failure in  binary_ptr
        return
    end if
    extras_binary_check_model = keep_going

end function extras_binary_check_model

! returns either keep_going or terminate.
! note: cannot request retry; extras_check_model can do that.
integer function extras_binary_finish_step(binary_id)
    type(binary_info), pointer :: b
    type(star_info), pointer :: s1
    integer, intent(in) :: binary_id
    integer :: ierr

    logical :: do_termination

    real(dp) :: cum_energy, tau_local, Mdot_crit, max_Mdot_crit
    integer :: k

    call binary_ptr(binary_id, b, ierr)
    call star_ptr(b%star_ids(1), s1, ierr)
    if (ierr /= 0) then ! failure in  binary_ptr
        return
    end if

    extras_binary_finish_step = keep_going

    do_termination = .false.

    cum_energy = 0d0
    max_Mdot_crit = -1d99

    do k = 1, s1%nz
        cum_energy = cum_energy + s1%dm(k)*s1%cp(k)*s1%T(k)

        if (s1%L(1) <= 0d0) cycle
        tau_local = cum_energy/s1%L(1)

        if (tau_local <= 0d0) cycle
        Mdot_crit = (s1%mstar - s1%m(k))/Msun/(tau_local/secyer)
        max_Mdot_crit = max(max_Mdot_crit, Mdot_crit)

    end do

    if (-s1%star_mdot > 3*max_Mdot_crit) then
        do_termination = .True.
    end if

    if (do_termination .eqv. .True.) then
        termination_code_str(t_xtra1) = "Critical mass loss rate exceeded"
        s1%termination_code = t_xtra1
        extras_binary_finish_step = terminate
    end if

end function extras_binary_finish_step

subroutine extras_binary_after_evolve(binary_id, ierr)
    type(binary_info), pointer :: b
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr
    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then ! failure in  binary_ptr
        return
    end if

    call test_suite_after_evolve(b, ierr)

end subroutine extras_binary_after_evolve

