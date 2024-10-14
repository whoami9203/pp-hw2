CXX = mpicxx
CC  = mpicc
CPPFLAGS = -I/work/samuel21119/PP24/hw2/lodepng -I/work/samuel21119/PP24/hw2/glm/include
LDFLAGS = -pthread -fopenmp -lm  -L/work/samuel21119/PP24/hw2/glm/lib64/libglm.a
CXXFLAGS = -std=c++17 
TARGET = hw2
HW2 = hw2.cpp
LODEPNG = /work/samuel21119/PP24/hw2/lodepng/lodepng.cpp

all: $(TARGET)


$(TARGET): $(HW2) $(LODEPNG)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(HW2) $(LODEPNG) -o $(TARGET) -pg

.PHONY: clean
clean:
	rm -r hw2	
