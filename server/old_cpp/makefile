NAME := server

GCC := g++

BUILD_DIR := ./build
SRC       := main.cpp
EXEC      := $(NAME)

LDFLAGS   := -lcurl

$(NAME): 
	mkdir -p $(BUILD_DIR)
	$(GCC) $(SRC) $(LDFLAGS) -o $(BUILD_DIR)/$(NAME)

clean:
	rm -rf build