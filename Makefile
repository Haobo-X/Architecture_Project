all: project

project: project.c myblockmm.o baseline.o
	gcc -O3 -msse4.1 -msse4.2 -mavx -mavx2 project.c myblockmm.o baseline.o -o project -lpthread

myblockmm.o: myblockmm.c
	gcc -O3 -msse4.1 -msse4.2 -mavx -mavx2 myblockmm.c -c -lpthread

baseline.o: baseline.c
	gcc -O3 -msse4.1 -msse4.2 -mavx -mavx2 baseline.c -c -lpthread

clean:
		rm -f project *.o
