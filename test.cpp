#include <dlfcn.h>
#include <stdio.h>

int main() {
    void *handle = dlopen("libopencv_video.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    
    void *symbol = dlsym(handle, "_ZN2cv12KalmanFilterC1Eiiii");
    if (!symbol) {
        fprintf(stderr, "Symbol not found: %s\n", dlerror());
    } else {
        printf("Symbol found at %p\n", symbol);
    }
    
    dlclose(handle);
    return 0;
}