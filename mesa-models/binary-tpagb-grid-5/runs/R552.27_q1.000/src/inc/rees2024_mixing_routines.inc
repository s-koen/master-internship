subroutine alpha_mlt_beta(id, ierr, A, B, inner_index)
   ! Rees & Izzard 2024
   ! increase mixing length parameter as a function of beta=Pgas/P to avoid the Fe-peak instability

   integer, intent(in) :: id, inner_index
   real(dp), intent(in) :: A, B
   integer, intent(out) :: ierr
   type (star_info), pointer :: s
   integer :: i
   real(dp) :: beta
   ierr = 0
   call star_ptr(id, s, ierr)
   if (ierr /= 0) return

   if (A>s%mixing_length_alpha) then
       do i=1,inner_index
           beta = s% Pgas(i)/s% Peos(i)
           s% alpha_mlt(i) = A -(A - s% mixing_length_alpha) /(1+10.0**(-6*(beta-B)))
       end do
   end if

end subroutine alpha_mlt_beta

subroutine calc_min_beta(s,min_beta)
   real(dp), intent(out) :: min_beta
   integer :: i
   type (star_info), pointer :: s
   !calculate minimum value of beta in envelope
   min_beta = 1.0
   do i=1, s% nz
       if (s% Pgas(i)/s% Peos(i) < min_beta) then
           min_beta = s% Pgas(i)/s% Peos(i)
       end if
   end do
end subroutine calc_min_beta

subroutine calc_alpha_mlt_max(s,alpha_mlt_max)
   real(dp),intent(out) :: alpha_mlt_max
   integer :: i
   type (star_info), pointer :: s

   alpha_mlt_max = 0
   do i=1, s% nz
       if (s% alpha_mlt(i) > alpha_mlt_max) then
           alpha_mlt_max = s% alpha_mlt(i)
       end if
   end do
end subroutine calc_alpha_mlt_max




     ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ! Andrassy et al. 2023
      !
      ! Adjust temperature gradient in the penetration zone to be adiabatic.
      ! Only works if conv_premix = .true. since the last iteration in a time
      ! step has NaNs in D_mix if conv_premix = .false.
      !
      ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      subroutine adjust_mlt_gradT_fraction_conv_pen(id, ierr)
        use utils_lib, only: is_bad
        integer, intent(in) :: id
        integer, intent(out) :: ierr
        type(star_info), pointer :: s
        integer :: k, max_mt

        ierr = 0
        call star_ptr(id, s, ierr)
        if (ierr /= 0) return

        if (s%num_conv_boundaries < 1) return

        if (s%model_number<10) return

        if(is_bad(s% D_mix(1))) return

        if (s% overshoot_scheme(1) == 'step') then
           do k= s%nz, 1, -1
              if (s%D_mix(k) <= s% min_D_mix) exit

              ! f is the fraction such that grad_T = f*grad_ad + (1-f)*grad_rad
              s% adjust_mlt_gradT_fraction(k) = 1.0_dp
           end do
        end if

     end subroutine adjust_mlt_gradT_fraction_conv_pen
