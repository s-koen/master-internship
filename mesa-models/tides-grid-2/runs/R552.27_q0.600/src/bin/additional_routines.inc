! *****************************************************************
!             ██                       ██   ▀▀
! ▄█▀▀▀ ██ ██ ████▄ ████▄ ▄███▄ ██ ██ ▀██▀▀ ██  ████▄ ▄█▀█▄ ▄█▀▀▀
! ▀███▄ ██ ██ ██ ██ ██ ▀▀ ██ ██ ██ ██  ██   ██  ██ ██ ██▄█▀ ▀███▄
! ▄▄▄█▀ ▀██▀█ ████▀ ██    ▀███▀ ▀██▀█  ██   ██▄ ██ ██ ▀█▄▄▄ ▄▄▄█▀
!
! *****************************************************************

subroutine wind_loss(binary_id, ierr)
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr
    type(binary_info), pointer :: b
    real(dp) :: q, vw_over_vorb
    real(dp) :: beta, eta
    real(dp) :: omega, I_eff
    real(dp) :: domega_dt_wind

    ierr = 0
    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then
        write (*, *) 'failed in binary_ptr'
        return
    end if

    ! --- compute q == m_accretor / m_donor and v
    q = b%m(b%a_i)/b%m(b%d_i)
    vw_over_vorb = compute_vwind(b%s_donor)/compute_vorbit(b%m(1) + b%m(2), b%separation)
    b%s_donor%xtra(x_vw_over_vorb) = vw_over_vorb

    ! --- get beta and eta from Saladino
    beta = compute_beta_Sal(q, vw_over_vorb)/sqrt(1 - pow2(b%eccentricity))
    b%s_donor%xtra(x_beta) = beta
    eta = compute_eta_Sal(q, vw_over_vorb)
    b%s_donor%xtra(x_eta) = eta

    ! --- star angular momentum loss due to winds
    ! compute the change in spin frequency. this
    ! gets used in extra_binary_finish_step.
    omega = b%s_donor%xtra(x_omega)
    I_eff = b%s_donor%xtra(x_I_eff)
    domega_dt_wind = (b%mdot_system_wind(b%d_i) / (1 - beta) *(b%s_donor%photosphere_r*Rsun)**2*omega/1.5)/I_eff
    b%s_donor%xtra(x_domega_dt_wind) = domega_dt_wind

    ! --- mass lost from vicinity of donor
    b%jdot_ml = b%mdot_system_wind(b%d_i) &
                *eta &
                *(b%separation)**2 &
                *sqrt(1 - pow2(b%eccentricity)) &
                *2*pi/b%period

end subroutine wind_loss

subroutine wind_accretion(binary_id, s_i, ierr)
    integer, intent(in) :: binary_id, s_i ! s_i is index of the wind mass losing star
    integer, intent(out) :: ierr

    type(binary_info), pointer :: b
    type(star_info), pointer :: s

    real(dp) :: q, vw_over_vorb, beta

    call binary_ptr(binary_id, b, ierr)
    if (ierr /= 0) then
        write (*, *) 'failed in binary_ptr'
        return
    end if

    ! --- compute q == m_accretor / m_donor and v
    q = b%m(b%a_i)/b%m(b%d_i)
    vw_over_vorb = compute_vwind(b%s_donor)/compute_vorbit(b%m(1) + b%m(2), b%separation)

    ! --- get beta from Saladino
    beta = compute_beta_Sal(q, vw_over_vorb)/sqrt(1 - pow2(b%eccentricity))
    b%wind_xfer_fraction(s_i) = beta

end subroutine wind_accretion

subroutine tides(binary_id, ierr)
    ! --- following the required input and output of default_jdot_ls
    integer, intent(in) :: binary_id
    integer, intent(out) :: ierr

    ! --- getting access to the binary object and star object
    type(binary_info), pointer :: b
    type(star_info), pointer :: s_d
    type(star_info), pointer :: s_a

    ! --- necessary other variables
    real(dp) :: omega, omega_orb
    real(dp) :: I_eff, Jdot_tides, dt
    real(dp) :: rg2, a, k_over_T
    real(dp) :: q, R, domega_dt_tides

    ierr = 0
    call binary_ptr(binary_id, b, ierr)
    s_d => b%s_donor

    dt = b%dt

    ! --- compute mass ratio, R and a
    q = b%m(b%a_i)/b%m(b%d_i)
    R = s_d%photosphere_r*Rsun ! in cm
    a = b%separation ! in cm

    ! --- compute k_over_T (including f_conv)
    k_over_T = compute_k_over_T(b, s_d)

    ! --- get initial spin and orbital frequency
    omega = s_d%xtra(x_omega)
    omega_orb = 2d0*pi/b%period

    ! --- compute moment of inertia of solid body and gyration radius
    I_eff = s_d%xtra(x_I_eff)
    rg2 = s_d%xtra(x_rg2) 

    ! --- hut spin evolution
    domega_dt_tides = 3*k_over_T*q**2/(rg2)*(R/a)**6*(omega_orb - omega)
    s_d%xtra(x_domega_dt_tides) = domega_dt_tides

    ! --- change jdot orbit
    Jdot_tides = I_eff*domega_dt_tides
    b%jdot_ls = -Jdot_tides

end subroutine tides

! ****************************************************
!   ▄▄
!  ██                     ██   ▀▀
! ▀██▀ ██ ██ ████▄ ▄████ ▀██▀▀ ██  ▄███▄ ████▄ ▄█▀▀▀
!  ██  ██ ██ ██ ██ ██     ██   ██  ██ ██ ██ ██ ▀███▄
!  ██  ▀██▀█ ██ ██ ▀████  ██   ██▄ ▀███▀ ██ ██ ▄▄▄█▀
!
! ****************************************************

real(dp) function compute_vorbit(Mtot, a) result(vorbit)
    implicit none

    real(dp) :: Mtot, a
    vorbit = sqrt(standard_cgrav*Mtot/a)

end function compute_vorbit

real(dp) function compute_beta_Sal(q, v) result(beta)
    implicit none

    real(dp) :: q, v
    real(dp) :: c1, c2
    real(dp) :: alpha, beta_max

    c1 = 1.7 + 0.3/q
    c2 = 0.5 + 0.2/q
    alpha = 1.0/(c1 + (c2*v)**5.0) + 0.75
    beta_max = min(0.3d0, 1.4*q**2)
    beta = min(compute_beta_BHL(q, v, alpha), beta_max)

end function compute_beta_Sal

real(dp) function compute_beta_BHL(q, v, alpha) result(beta)
    implicit none

    real(dp) :: q, v, alpha
    real(dp) :: vrel

    vrel = sqrt(1 + v**2)
    beta = alpha/((1 + 1/q)**2*v*vrel**3)

end function compute_beta_BHL

real(dp) function compute_eta_Sal(q, v) result(eta)
    implicit none

    real(dp) :: q, v
    real(dp) :: eta_iso, c1, c2

    eta_iso = (1 + 1/q)**(-2)
    c1 = max(1/q, 0.6*(1/q)**1.7)
    c2 = 1.5 + 0.3/q
    eta = min(1.0/(c1 + (c2*v)**3) + eta_iso, 0.6d0)

end function compute_eta_Sal

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
    a = 2.72 + 0.63*log_z
    b1 = 0.68 - 0.219*log_z
    c = 0.12 - 0.023*log_z

    M_conv = s%xtra(x_M_conv)
    M = s%star_mass

    R_conv = s% xtra(x_R_conv) 
    R = s%photosphere_r

    t_conv = s%xtra(x_t_conv)
    f_conv = compute_f_conv(s, b, t_conv)
    s%xtra(x_f_conv) = f_conv

    k_over_T = (R_conv/R)**a*(M_conv/M)**b1*c/t_conv*f_conv

end function compute_k_over_T


real(dp) function compute_f_conv(s, b, t_conv) result(f_conv)
    implicit none

    type(star_info), pointer :: s
    type(binary_info), pointer :: b
    real(dp) frac, t_conv, o_orb, o_tid, p_tid

    o_orb = 2d0*pi/(b%period)
    o_tid = max(1d-10, abs(o_orb - s%xtra(x_omega)))
    p_tid = 2 * pi / o_tid

    frac = p_tid/(2d0*t_conv)
    f_conv = min(1d0, frac**2d0)

end function compute_f_conv
