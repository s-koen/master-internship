set term postscript color enhanced 24
set output "print.ps"
set pointsize 1.8
#set size square
set style line 1 lt 1 lw 4 pt 1
set style line 2 lt 3 lw 4 pt 9
set style line 3 lt 1 lw 4 pt 6
set style line 4 lt 2 lw 4 pt 4
set style line 5 lt 9 lw 4 pt 7
set style line 6 lt 8 lw 4 pt 8
set style line 7 lt 5 lw 4 pt 7
set style line 8 lt 9 lw 4 pt 3
set style line 9 lt 0 lw 8 pt 2
#set label "Y=0.26" at 2.2, 3.0 font "Arial,16"
set label "(b)" at 1.5, 1.74 font "Arial, 36
set label "Y=0.32" at 3.1, 1.2 font "Arial,16"
set label "Y=0.30" at 3.1, 1.6 font "Arial,16"
set label "Y=0.35" at 3.1, 0.8 font "Arial,16"
set label "Y=0.40" at 3.1, 0.28 font "Arial,16"
#set label "Y=0.24" at 4.15, 2.30 font "Arial,16"
set label "C/O = 1" at 7.0, 1.2 font "Arial,20"
set xlabel "Initial mass (M_{sun})"; set ylabel "C/O ratio"
plot [0.8:8.2][0:2] "z03.dat" using 1:3 title "Z = 0.03, Y=0.30" with lp ls 1,\
"m3z03.dat" using 1:3 notitle with lp ls 5,\
"m3.5z03.dat" using 1:3 notitle with lp ls 5,\
"m4z03.dat" using 1:3 notitle with lp ls 5,\
"m5z03.dat" using 1:3 notitle with lp ls 5,\
"co1.dat" using 1:2 notitle with l ls 9,\
"y0.35.dat" using 1:3 title "Y=0.35" with l ls 5

#"m3z014.dat" using 1:3 notitle with lp ls 5,\
#"m4z014.dat" using 1:3 notitle with lp ls 5,\



#"m3z014helium.dat" using 1:3 notitle with lp ls 5,\

#"z03ov.dat" using 1:3 notitle with lp ls 5,\
#"z014ov.dat" using 1:3 notitle with lp ls 5,\

#"z03.dat" using 1:3 title "Z = 0.03" with lp ls 1,\