
c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

c---------------------------------------------------------------------
c The following include file is generated automatically by the
c "setparams" utility. It defines 
c      problem_size:  12, 64, 102, 162 (for class T, A, B, C)
c      dt_default:    default time step for this problem size if no
c                     config file
c      niter_default: default number of iterations for this problem size
c---------------------------------------------------------------------

      include 'npbparams.h'

      integer           npb_verbose
      logical           timeron
      common /global/   npb_verbose, timeron

      double precision  tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, 
     >                  dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, 
     >                  dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, 
     >                  ce(5,13), dxmax, dymax, dzmax, xxcon1, xxcon2, 
     >                  xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
     >                  dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
     >                  yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
     >                  zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
     >                  dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
     >                  dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2, 
     >                  c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, bt,
     >                  dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
     >                  c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
     >                  c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16

      common /constants/ tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3,
     >                  dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, 
     >                  dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, 
     >                  ce, dxmax, dymax, dzmax, xxcon1, xxcon2, 
     >                  xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
     >                  dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
     >                  yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
     >                  zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
     >                  dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
     >                  dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2, 
     >                  c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, bt,
     >                  dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
     >                  c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
     >                  c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16

      double precision cv(0:problem_size-1),   rhon(0:problem_size-1),
     >                 rhos(0:problem_size-1), rhoq(0:problem_size-1),
     >                 cuf(0:problem_size-1),  q(0:problem_size-1),
     >                 ue(0:problem_size-1,5), buf(0:problem_size-1,5)
      common /work_1d/ cv, rhon, rhos, rhoq, cuf, q, ue, buf
!$OMP THREADPRIVATE(/work_1d/)

      double precision lhs (5,0:problem_size,0:problem_size),
     >                 lhsp(5,0:problem_size,0:problem_size),
     >                 lhsm(5,0:problem_size,0:problem_size)
      common /work_lhs/ lhs, lhsp, lhsm
!$OMP THREADPRIVATE(/work_lhs/)

      integer   max_zones
      parameter (max_zones=x_zones*y_zones)
      integer   x_start(x_zones), x_end(x_zones), x_size(x_zones),
     >          y_start(y_zones), y_end(y_zones), y_size(y_zones),
     >          iz_west (max_zones), iz_east (max_zones),
     >          iz_south(max_zones), iz_north(max_zones)
      common /zones/ x_start, x_end, x_size, y_start, y_end, y_size,
     >               iz_west, iz_east, iz_south, iz_north

      dimension start1(max_zones), start5(max_zones),
     $          qstart_west (max_zones), qstart_east (max_zones),
     $          qstart_south(max_zones), qstart_north(max_zones)
      common /array_start/ start1, start5, qstart_west, qstart_east,
     $                     qstart_south, qstart_north, qoffset

c-----------------------------------------------------------------------
c   Timer constants
c-----------------------------------------------------------------------
      integer t_rhsx,t_rhsy,t_rhsz,t_xsolve,t_ysolve,t_zsolve,
     >        t_rdis1,t_rdis2,t_tzetar,t_ninvr,t_pinvr,t_add,
     >        t_rhs,t_txinvr,t_last,t_total
      parameter (t_total = 1)
      parameter (t_rhsx = 2)
      parameter (t_rhsy = 3)
      parameter (t_rhsz = 4)
      parameter (t_rhs = 5)
      parameter (t_xsolve = 6)
      parameter (t_ysolve = 7)
      parameter (t_zsolve = 8)
      parameter (t_rdis1 = 9)
      parameter (t_rdis2 = 10)
      parameter (t_txinvr = 11)
      parameter (t_pinvr = 12)
      parameter (t_ninvr = 13)
      parameter (t_tzetar = 14)
      parameter (t_add = 15)
      parameter (t_last = 15)

