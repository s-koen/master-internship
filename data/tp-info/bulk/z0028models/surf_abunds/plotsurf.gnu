set term postscript color enhanced 24
set output "print.ps"
set pointsize 1.8
#set size square
set style line 1 lt 4 lw 4 pt 7
set style line 2 lt 3 lw 4 pt 7
#set style line 3 lt 1 lw 4 pt 7
set style line 3 lc rgb "red"  lw 4 pt 7
set style line 4 lt 4 lw 4 pt 7
set style line 5 lt 5 lw 4 pt 7
set style line 6 lt 8 lw 4 pt 8
set style line 7 lt 5 lw 4 pt 7
set style line 8 lt 9 lw 4 pt 3
set style line 9 lt 0 lw 8 pt 2
set key at 60, 2.95
set xlabel "Atomic number, Z"; set ylabel "[X/Fe]"
plot [25:85] "m2z0028.dat" using 2:5 title "2Msun, [Fe/H] = -0.7" with lp ls 2,\
"tp10.dat"  using 2:5 title "TP=10" with lp ls 3,\
"tp15.dat"  using 2:5 title "TP=15" with lp ls 4,\
"tp20.dat"  using 2:5 title "TP=20" with lp ls 5

#"m3z014.dat" using 1:3 notitle with lp ls 5,\
#"m4z014.dat" using 1:3 notitle with lp ls 5,\



#"m3z014helium.dat" using 1:3 notitle with lp ls 5,\

#"z03ov.dat" using 1:3 notitle with lp ls 5,\
#"z014ov.dat" using 1:3 notitle with lp ls 5,\

#"z03.dat" using 1:3 title "Z = 0.03" with lp ls 1,\