subroutine toy_tides(binary_id, ierr)
    ! --- following the required input and output of default_jdot_ls
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr

    ! --- getting access to the binary object and star object
    type(binary_info), pointer :: b
    type(star_info), pointer :: s

    ! --- necessary other variables
    real(dp) :: omega, omega_orb, domega_dt
    real(dp) :: I_eff, Jdot_tides, dt
    real(dp) :: t_sync

    integer :: k ! for counting zones

    ierr = 0
    call binary_ptr(binary_id, b, ierr)
    s => b% s_donor 

    dt = s% dt

    ! --- get initial spin and orbital frequency
    omega = s% x_ctrl(3)
    omega_orb = 2d0*pi / b% period

    ! --- arbitrary sync timescale
    t_sync = 1d12

    ! --- simplified hut spin evolution
    domega_dt = (omega_orb - omega) / t_sync

    ! --- update spin
    omega = omega + dt * domega_dt
    s% x_ctrl(3) = omega 

    ! --- compute moment of inertia of solid body
    I_eff = 0
    do k=1,s% nz
        I_eff = I_eff + s% dm_bar(k) * (s% r(k)**2)
    end do
    Jdot_tides = I_eff * domega_dt

    ! --- change jdot orbit
    b% jdot_ls = - Jdot_tides

    ! --- write orbit and spin frequency
    write(*,*) "Omega_star = ", omega 
    write(*,*) "Omega_orb  = ", omega_orb
    write(*,*) "J_orb =", b% angular_momentum_j
    write(*,*) "a =", b% separation



end subroutine toy_tides


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


