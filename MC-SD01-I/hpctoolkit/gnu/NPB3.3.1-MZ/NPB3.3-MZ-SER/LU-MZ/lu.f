!-------------------------------------------------------------------------!
!                                                                         !
!        N  A  S     P A R A L L E L     B E N C H M A R K S  3.3         !
!                                                                         !
!          S E R I A L    M U L T I - Z O N E    V E R S I O N            !
!                                                                         !
!                              L U - M Z                                  !
!                                                                         !
!-------------------------------------------------------------------------!
!                                                                         !
!    This benchmark is a serial version of the NPB LU code.               !
!    Refer to NAS Technical Reports 95-020 for details.                   !
!                                                                         !
!    Permission to use, copy, distribute and modify this software         !
!    for any purpose with or without fee is hereby granted.  We           !
!    request, however, that all derived work reference the NAS            !
!    Parallel Benchmarks 3.3. This software is provided "as is"           !
!    without express or implied warranty.                                 !
!                                                                         !
!    Information on NPB 3.3, including the technical report, the          !
!    original specifications, source code, results and information        !
!    on how to submit new results, is available at:                       !
!                                                                         !
!           http://www.nas.nasa.gov/Software/NPB/                         !
!                                                                         !
!    Send comments or suggestions to  npb@nas.nasa.gov                    !
!                                                                         !
!          NAS Parallel Benchmarks Group                                  !
!          NASA Ames Research Center                                      !
!          Mail Stop: T27A-1                                              !
!          Moffett Field, CA   94035-1000                                 !
!                                                                         !
!          E-mail:  npb@nas.nasa.gov                                      !
!          Fax:     (650) 604-3957                                        !
!                                                                         !
!-------------------------------------------------------------------------!

c---------------------------------------------------------------------
c
c Authors: S. Weeratunga
c          V. Venkatakrishnan
c          E. Barszcz
c          M. Yarrow
C          R.F. Van der Wijngaart
c
c---------------------------------------------------------------------

c---------------------------------------------------------------------
      program applu
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c
c   driver for the performance evaluation of the solver for
c   five coupled parabolic/elliptic partial differential equations.
c
c---------------------------------------------------------------------

      implicit none

      include 'header.h'

      integer   num_zones
      parameter (num_zones=x_zones*y_zones)

      integer   nx(num_zones), nxmax(num_zones), ny(num_zones), 
     $          nz(num_zones)

c---------------------------------------------------------------------
c   Define all field arrays as one-dimenional arrays, to be reshaped
c---------------------------------------------------------------------

      double precision u    (proc_max_size5),
     >                 rsd  (proc_max_size5),
     >                 frct (proc_max_size5),
     >                 qs   (proc_max_size ),
     >                 rho_i(proc_max_size ),
     >                 qbc  (proc_max_bcsize)

      common/cvar/ u, rsd, frct, qs, rho_i, qbc

c---------------------------------------------------------------------
c   2D auxiliary arrays are dimensioned to accommodate the largest
c   zone cross section
c---------------------------------------------------------------------

      double precision a(25*problem_size*problem_size), 
     $                 b(25*problem_size*problem_size), 
     $                 c(25*problem_size*problem_size), 
     $                 d(25*problem_size*problem_size),
     $                 phi1(problem_size*problem_size),
     $                 phi2(problem_size*problem_size)

      common/cjac/ a, b, c, d, phi1, phi2

      logical verified
      double precision mflops, nsur, navg, n3

      integer i, zone, step
      double precision t, tmax, timer_read, trecs(t_last),
     $                 rsdnm(5), rsdnm_aux(5), errnm(5), errnm_aux(5),
     $                 frc, frc_aux
      external timer_read
      character t_names(t_last)*8

c---------------------------------------------------------------------
c   read input data
c---------------------------------------------------------------------
      call read_input()

      if (timeron) then
         t_names(t_total)  = 'total'
         t_names(t_rhsx)   = 'rhsx'
         t_names(t_rhsy)   = 'rhsy'
         t_names(t_rhsz)   = 'rhsz'
         t_names(t_rhs)    = 'rhs'
         t_names(t_jacld)  = 'jacld'
         t_names(t_blts)   = 'blts'
         t_names(t_jacu)   = 'jacu'
         t_names(t_buts)   = 'buts'
         t_names(t_add)    = 'add'
         t_names(t_l2norm) = 'l2norm'
         t_names(t_rdis1)  = 'qbc_copy'
         t_names(t_rdis2)  = 'qbc_comm'
      endif

c---------------------------------------------------------------------
c   set up domain sizes
c---------------------------------------------------------------------
      call zone_setup(nx, nxmax, ny, nz)

c---------------------------------------------------------------------
c   set up coefficients
c---------------------------------------------------------------------
      call setcoeff()

      do i = 1, t_last
         call timer_clear(i)
      end do

      do zone = 1, num_zones

c---------------------------------------------------------------------
c   set the boundary values for dependent variables
c---------------------------------------------------------------------
        call setbv(u(start5(zone)),
     $             nx(zone), nxmax(zone), ny(zone), nz(zone))

c---------------------------------------------------------------------
c   set the initial values for dependent variables
c---------------------------------------------------------------------
        call setiv(u(start5(zone)),
     $          nx(zone), nxmax(zone), ny(zone), nz(zone))

c---------------------------------------------------------------------
c   compute the forcing term based on prescribed exact solution
c---------------------------------------------------------------------
        call erhs(frct(start5(zone)), rsd(start5(zone)),
     $            nx(zone), nxmax(zone), ny(zone), nz(zone))

c---------------------------------------------------------------------
c   compute the steady-state residuals
c---------------------------------------------------------------------
        call rhs(u(start5(zone)), rsd(start5(zone)), 
     $           frct(start5(zone)), qs(start1(zone)), 
     $           rho_i(start1(zone)), 
     $           nx(zone), nxmax(zone), ny(zone), nz(zone))

      end do

c---------------------------------------------------------------------
c   initialize a,b,c,d to zero (guarantees that page tables have been
c   formed, if applicable on given architecture, before timestepping).
c---------------------------------------------------------------------
      do i=1, 25*problem_size*problem_size
        a(i) = 0.d0
        b(i) = 0.d0
        c(i) = 0.d0
        d(i) = 0.d0
      enddo

c---------------------------------------------------------------------
c   perform one SSOR iteration to touch all data pages
c---------------------------------------------------------------------

      call exch_qbc(u, qbc, nx, nxmax, ny, nz)

      do zone = 1, num_zones
        call ssor(u(start5(zone)), rsd(start5(zone)), 
     $            frct(start5(zone)), qs(start1(zone)), 
     $            rho_i(start1(zone)), a, b, c, d, 
     $            nx(zone), nxmax(zone), ny(zone), nz(zone))
      end do

c---------------------------------------------------------------------
c   reset the boundary and initial values
c---------------------------------------------------------------------
      do zone = 1, num_zones

        call setbv(u(start5(zone)),
     $             nx(zone), nxmax(zone), ny(zone), nz(zone))

        call setiv(u(start5(zone)),
     $          nx(zone), nxmax(zone), ny(zone), nz(zone))

c---------------------------------------------------------------------
c   compute the steady-state residuals
c---------------------------------------------------------------------
        call rhs(u(start5(zone)), rsd(start5(zone)), 
     $           frct(start5(zone)), qs(start1(zone)), 
     $           rho_i(start1(zone)), 
     $           nx(zone), nxmax(zone), ny(zone), nz(zone))

      end do

c---------------------------------------------------------------------
c   begin pseudo-time stepping iterations
c---------------------------------------------------------------------

      do i = 1, t_last
         call timer_clear(i)
      end do
      call timer_start(1)

c---------------------------------------------------------------------
c   the timestep loop
c---------------------------------------------------------------------
      do step = 1, itmax

        if (mod(step,20) .eq. 0 .or. step .eq. 1 .or.
     >        step .eq. itmax) then
           write( *, 200) step
 200       format(' Time step ', i4)
        endif

        call exch_qbc(u, qbc, nx, nxmax, ny, nz)

c---------------------------------------------------------------------
c   perform the SSOR iterations
c---------------------------------------------------------------------

        do zone = 1, num_zones
          call ssor(u(start5(zone)), rsd(start5(zone)), 
     $              frct(start5(zone)), qs(start1(zone)), 
     $              rho_i(start1(zone)), a, b, c, d, 
     $              nx(zone), nxmax(zone), ny(zone), nz(zone))
        end do

      end do
 
      do i = 1, 5
         rsdnm(i) = 0.d0
         errnm(i) = 0.d0
      end do
      frc = 0.d0

c---------------------------------------------------------------------
c   compute the max-norms of newton iteration residuals
c---------------------------------------------------------------------
      if (timeron) call timer_start(t_l2norm)
      do zone = 1, num_zones
        call l2norm(rsd(start5(zone)), rsdnm_aux,
     $              nx(zone), nxmax(zone), ny(zone), nz(zone))
        do i = 1, 5
          rsdnm(i) = rsdnm(i) + rsdnm_aux(i)
        end do
      end do
      if (timeron) call timer_stop(t_l2norm)

      call timer_stop(1)
      maxtime= timer_read(1)

c---------------------------------------------------------------------
c   compute the solution error and surface integral
c---------------------------------------------------------------------
      do zone = 1, num_zones
        call error(u(start5(zone)), errnm_aux,
     $             nx(zone), nxmax(zone), ny(zone), nz(zone))
        call pintgr(u(start5(zone)), phi1, phi2, frc_aux,
     $              nx(zone), nxmax(zone), ny(zone), nz(zone))
        do i = 1, 5
          errnm(i) = errnm(i) + errnm_aux(i)
        end do
        frc = frc + frc_aux
      end do

c---------------------------------------------------------------------
c   verification test
c---------------------------------------------------------------------
      call verify ( rsdnm, errnm, frc, verified )

      mflops = 0.d0

      if (maxtime .ne. 0.d0) then
        do zone = 1, num_zones
          n3 = dble(nx(zone))*ny(zone)*nz(zone)
          navg = (nx(zone) + ny(zone) + nz(zone))/3.d0
          nsur = (nx(zone)*ny(zone) + nx(zone)*nz(zone) +
     >            ny(zone)*nz(zone))/3.d0
          mflops = mflops + float(itmax)*1.0d-6 *
     >       (1984.77d0 * n3 - 10923.3d0 * nsur
     >         + 27770.9d0 * navg - 144010.d0)
     >       / maxtime
        end do
      endif

      call print_results('LU-MZ', class, gx_size, gy_size, gz_size, 
     >  itmax, maxtime, mflops, 
     >  '          floating point', verified, npbversion, 
     >  compiletime, cs1, cs2, cs3, cs4, cs5, cs6, '(none)')

c---------------------------------------------------------------------
c      More timers
c---------------------------------------------------------------------
      if (.not.timeron) goto 999

      do i=1, t_last
         trecs(i) = timer_read(i)
      end do
      tmax = maxtime
      if ( tmax .eq. 0. ) tmax = 1.0

      write(*,800)
 800  format('  SECTION     Time (secs)')
      do i=1, t_last
         write(*,810) t_names(i), trecs(i), trecs(i)*100./tmax
         if (i.eq.t_rhs) then
            t = trecs(t_rhsx) + trecs(t_rhsy) + trecs(t_rhsz)
            write(*,820) 'sub-rhs', t, t*100./tmax
            t = trecs(i) - t
            write(*,820) 'rest-rhs', t, t*100./tmax
         elseif (i.eq.t_rdis2) then
            t = trecs(t_rdis1) + trecs(t_rdis2)
            write(*,820) 'exch_qbc', t, t*100./tmax
         endif
 810     format(2x,a8,':',f9.3,'  (',f6.2,'%)')
 820     format(5x,'--> total ',a8,':',f9.3,'  (',f6.2,'%)')
      end do

 999  continue
      end
