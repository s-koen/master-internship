subroutine wind_loss(binary_id, ierr)
    ! --- following the required input and output of default_jdot_ml
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr

    ! --- getting access to the binary object and star object
    type (binary_info), pointer :: b
    type(star_info), pointer :: s_d
    real(dp) :: dt

    ierr = 0
    call binary_ptr(binary_id, b, ierr)
    s_d => b% s_donor 
    dt = s_d% dt

    ! --- MESA splits binary wind mass loss into 3 separate variables
    !   1. mdot_system_transfer(2) --- rate of mass loss from the system due to inefficient mass accretion

    b% jdot_ml = 0
end subroutine wind_loss

subroutine tides(binary_id, ierr)
    ! --- following the required input and output of default_jdot_ls
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr

    ! --- getting access to the binary object and star object
    type(binary_info), pointer :: b
    type(star_info), pointer :: s_d
    type(star_info), pointer :: s_a

    ! --- necessary other variables
    real(dp) :: omega, omega_orb, domega_dt
    real(dp) :: I_eff, Jdot_tides, dt
    real(dp) :: I_old, dlnI_dt, rg2
    real(dp) :: domega_dt_tides, q, R
    real(dp) :: a, k_over_T

    integer :: k ! for counting zones

    ierr = 0
    call binary_ptr(binary_id, b, ierr)
    s_d => b% s_donor 

    dt = s_d% dt

    ! --- compute mass ratio, R and a
    q = b% m(2) / b% m(1) 
    R = s_d% photosphere_r * Rsun ! in cm
    a = b% separation ! in cm


    ! --- compute k_over_T (including f_conv)
    k_over_T = compute_k_over_T(b, s_d)

    ! --- get initial spin and orbital frequency
    omega = s_d% x_ctrl(3)
    omega_orb = 2d0*pi / b% period

    ! --- compute moment of inertia of solid body and gyration radius
    I_eff = compute_I_eff(s_d)
    rg2 = compute_rg2(s_d, I_eff)

    ! --- compute dlnI/dt
    I_old = s_d% x_ctrl(4)
    if (I_old > 0d0 .and. dt > 0d0) then
        dlnI_dt = (log(I_eff) - log(I_old)) / dt
    else
        dlnI_dt = 0d0
    end if

    ! --- hut spin evolution
    domega_dt_tides = 3* k_over_T * q**2 / (rg2) * (R / a)**6 * (omega_orb - omega) 

    ! --- total spin evolution (structure + tides)
    domega_dt = domega_dt_tides - omega * dLnI_dt

    ! --- update spin
    omega = omega + dt * domega_dt
    s_d% x_ctrl(3) = omega 

    ! --- store I for next step
    s_d% x_ctrl(4) = I_eff


    ! --- change jdot orbit
    Jdot_tides = I_eff * domega_dt_tides
    b% jdot_ls = - Jdot_tides

    ! --- write orbit and spin frequency
    write(*,*) "Omega_star =", omega, " /s"
    write(*,*) "Omega_orb  =", omega_orb, " /s"
    write(*,*) "J_orb =", b% angular_momentum_j, " g cm^2 / s"
    write(*,*) "a     =", b% separation, " cm"
    write(*,*) "dt    =", dt, " s"
    write(*,*) "I_eff =", I_eff, " g cm^2"
    write(*,*) "rg2   = ", rg2
    write(*,*) ""

end subroutine tides

real(dp) function compute_I_eff(s) result(I_eff)
    implicit none

    type(star_info), pointer :: s
    integer :: k

    I_eff = 0d0
    do k = 1, s% nz
        I_eff = I_eff + (2d0/3d0) * s% dm_bar(k) * s% r(k)**2
    end do

end function compute_I_eff

real(dp) function compute_k_over_T(b, s) result(k_over_T)
    implicit none

    type(star_info), pointer :: s
    type(binary_info), pointer :: b
    real(dp) :: a, b1, c
    real(dp) :: log_z
    real(dp) :: R_conv, R
    real(dp) :: M_conv, M
    real(dp) :: t_conv
    real(dp) :: f_conv

    log_z = log10(0.014)
    a = 2.72 + 0.63 * log_z
    b1 = 0.68 - 0.219 * log_z
    c = 0.12 - 0.023 * log_z

    M_conv = compute_M_conv(s)
    M = s% star_mass

    R_conv = compute_R_conv(s)
    R = s% photosphere_r

    t_conv = compute_t_conv(s, M_conv, R_conv)
    f_conv = compute_f_conv(s, b, t_conv)

    k_over_T = (R_conv / R)**a * (M_conv / M)**b1 * c/t_conv * f_conv


    write(*,*) "M_conv=", M_conv, " Msun"
    write(*,*) "R_conv=", R_conv, " Rsun"
    write(*,*) "t_conv=", t_conv, " s"
    write(*,*) "t_conv=", t_conv / secyer * 12, " months"
    write(*,*) "k_over_T   =", k_over_t, " /s"


end function compute_k_over_T

real(dp) function compute_M_conv(s) result(M_conv)
    implicit none

    type(star_info), pointer :: s
    integer :: i
    real(dp) :: q_top, q_bot, q_max

    M_conv = 0
    q_max = 0
    do i = 1, s% n_conv_regions
        if (s% cz_top_mass(i) / Msun - s% he_core_mass > 8d-3) then
            ! --- starts sufficiently close to the edge 
            q_top = s% cz_top_mass(i) / Msun / s% star_mass
            q_bot = s% cz_bot_mass(i) / Msun / s% star_mass

            write(*,*) q_top, q_bot

            if (q_top - q_bot > q_max) then
                q_max = q_top - q_bot
                write(*,*) "found new q_max:" 
                write(*,*) q_top, q_bot, q_max

            end if

        end if
    end do
    write(*,*) "found M_conv:" 
    write(*,*) q_max, "gives M_conv = ", s% mstar * q_max / Msun

    M_conv = s% mstar * q_max / Msun

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

    do i = 1, s% n_conv_regions
        if (s% cz_top_mass(i) / Msun - s% he_core_mass > 8d-3) then
            ! --- starts sufficiently close to the edge 
            q_top = s% cz_top_mass(i) / Msun / s% star_mass
            q_bot = s% cz_bot_mass(i) / Msun / s% star_mass

            if (q_top - q_bot > q_max) then
                q_max = q_top - q_bot
                q_top_max = q_top
                q_bot_max = q_bot
            end if
        end if
    end do

    ! --- now find which rs this corresponds to
    R_top = 0d0 
 
    do k = 1, s% nz
        if (s% m(k) / Msun / s% star_mass <= q_top_max) then
            R_top = s% r(k)
            exit
        end if
    end do

    R_bot = 0d0 ! for safety

    do k = k, s% nz
        if (s% m(k) / Msun / s% star_mass <= q_bot_max) then
            R_bot = s% r(k)
            exit
        end if
    end do

    R_conv = (R_top - R_bot) / Rsun

end function compute_R_conv

real(dp) function compute_t_conv(s, M_conv, R_conv) result(t_conv)
    implicit none

    type(star_info), pointer :: s
    real(dp) M_conv, R_conv, L

    t_conv = 0.4311d0 * (3 * M_conv * R_conv**2 / (s% photosphere_L))**(1d0/3d0) * secyer

end function compute_t_conv

real(dp) function compute_f_conv(s, b, t_conv) result(f_conv)
    implicit none

    type(star_info), pointer :: s
    type(binary_info), pointer :: b
    real(dp) frac, t_conv, P_tid

    P_tid = abs(1d0/(b% period) - (2d0*pi)/(s% x_ctrl(3)))
    frac = P_tid / (2d0 * t_conv)
    f_conv = min(1d0, frac**2d0)

end function compute_f_conv

real(dp) function compute_rg2(s, I_eff) result(rg2)
    implicit none

    type(star_info), pointer :: s
    integer :: k
    real(dp) :: I_eff, M, R

    M = s% mstar
    R = s% r(1)

    if (M > 0d0 .and. R > 0d0) then
        rg2 = I_eff / (M * R * R)
    else
        rg2 = 0d0
    end if

end function compute_rg2


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

