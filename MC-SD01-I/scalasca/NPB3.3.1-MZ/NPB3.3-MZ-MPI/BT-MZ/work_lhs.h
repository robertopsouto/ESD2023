c---------------------------------------------------------------------
c---------------------------------------------------------------------
c
c  work_lhs.h
c
c---------------------------------------------------------------------
c---------------------------------------------------------------------

      double precision fjac(5, 5, 0:problem_size),
     >                 njac(5, 5, 0:problem_size),
     >                 lhs (5, 5, 3, 0:problem_size),
     >                 rtmp(5, 0:problem_size),
     >                 tmp1, tmp2, tmp3
      common /work_lhs/ fjac, njac, lhs, rtmp, tmp1, tmp2, tmp3
!$OMP THREADPRIVATE(/work_lhs/)
