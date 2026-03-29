! ***********************************************************************
!
!   Copyright (C) 2012  Bill Paxton
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful,
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************
module run_binary_extras

    use star_lib
    use star_def
    use const_def
    use math_lib
    use binary_def

    implicit none

    include "binary_test_suite_extras_def.inc"

contains

    include "binary_test_suite_extras.inc"
    include 'bin/run_binary_extras.inc'
    include 'bin/additional_routines.inc'

end module run_binary_extras
